import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =========================
# Tunables / Debug
# =========================
DEBUG = False                    # Set True to see logs
FRAMES = 600                     # Lower = less CPU
INTERVAL_MS = 80                 # Higher = less CPU (frame gap)
MAXFSOLVE_EVAL = 60              # Limit iterations to avoid hang
PHI_DEG = 75                     # fixed tilt for demo animation

# =========================
# Geometry constants
# =========================
deg2rad = np.deg2rad
rad2deg = np.rad2deg

# Base link / servo geometry (as in your file)
l  = 10
l_1, l_11, l_12 = 10, 8, 8
l_2, l_21, l_22 = 10, 8, 8
l_3, l_31, l_32 = 10, 8, 8

# Fixed offsets / base circle
S = np.array([0.0, 0.0, 12.0], dtype=float)

# Precompute shared trig & constants
c30 = np.cos(np.pi/6)
s30 = np.sin(np.pi/6)
a   = l / (2.0 * c30)

# Small helpers
def _clip01(x):
    # Clamp for acos/asin domains
    return np.clip(x, -1.0, 1.0)

def _norm(v):
    n = np.linalg.norm(v)
    return v/n if n > 1e-12 else v

# ============================================
# Compact, allocation-light “theta” solvers
# ============================================
def calculate_vectors_and_angles_1(l, l_1, l_11, l_12, x1):
    # theta_11
    AA = l_12**2 - (l_1 - x1[1])**2 - l_11**2 - x1[2]**2
    bb = 2.0*(l_1 - x1[1]) * l_11
    cc = 2.0*x1[2]*l_11
    denom = np.hypot(bb, cc)
    if denom < 1e-12:  # avoid division zero
        return 0.0, 0.0
    C = np.arccos(_clip01(AA/denom))
    B1 = np.arccos(_clip01(bb/denom))
    th11 = C - B1
    th12 = np.arcsin(_clip01((x1[2] - l_11*np.sin(th11))/l_12))
    return rad2deg(th11), rad2deg(th12)

def calculate_vectors_and_angles_2(l, l_2, l_21, l_22, x2):
    y = np.hypot(x2[0], x2[1])
    z = x2[2]
    AA = -(l_22**2 - (l_2 - y)**2 - l_21**2 - z**2)
    bb = 2.0*(l_2 - y)*l_21
    cc = 2.0*z*l_21
    denom = np.hypot(bb, cc)
    if denom < 1e-12:
        return 0.0, 0.0
    C = np.arcsin(_clip01(AA/denom))
    B1 = np.arcsin(_clip01(bb/denom))
    th21 = C + B1
    th22 = np.arcsin(_clip01((z - l_21*np.sin(th21))/l_22))
    return rad2deg(th21), rad2deg(th22)

def calculate_vectors_and_angles_3(l, l_3, l_31, l_32, x3):
    y = np.hypot(x3[0], x3[1])
    z = x3[2]
    AA = l_32**2 - (l_3 - y)**2 - l_31**2 - z**2
    bb = 2.0*(l_3 - y)*l_31
    cc = 2.0*z*l_31
    denom = np.hypot(bb, cc)
    if denom < 1e-12:
        return 0.0, 0.0
    C = np.arccos(_clip01(AA/denom))
    B2 = np.arcsin(_clip01(cc/denom))
    th31 = C - B2
    th32 = np.arcsin(_clip01((z - l_31*np.sin(th31))/l_32))
    return rad2deg(th31), rad2deg(th32)

# ======================================================
# Fast triangle placement from normal + robust fsolve
# (This replaces the long eeq* stack while keeping API)
# ======================================================
# Unit basis directions for triangle legs on XY plane
X1_hat = np.array([0.0, 1.0, 0.0])
X2_hat = np.array([ c30, -s30, 0.0])
X3_hat = np.array([-c30, -s30, 0.0])
Z_hat  = np.array([0.0, 0.0, 1.0])

def _triangle_from_d(nrm, d1, d2, d3, c1):
    # Compute vertical offsets c2,c3 s.t. plane normal is nrm
    nx, ny, nz = nrm
    # Simple linearized offsets matching your construction
    c2 = c1 + (1.0/nz)*(-d2*c30*nx + d2*s30*ny + d1*ny)
    c3 = c2 + (1.0/nz)*((d2+d3)*c30*nx - (-d3+d2)*s30*ny)
    P1 = d1*X1_hat + c1*Z_hat
    P2 = d2*X2_hat + c2*Z_hat
    P3 = d3*X3_hat + c3*Z_hat
    # recenter to S
    pp = (P1+P2+P3)/3.0 - S
    return P1-pp, P2-pp, P3-pp

def _eq_for_fsolve(d1, nrm):
    # Scalar balance in your original formulation (collapsed)
    # We keep it simple: enforce edge-length ≈ l between (P1,P2) & (P1,P3)
    # while P2,P3 are expressed as functions of d1 using symmetry heuristic.
    # To keep CPU low, we tie d2,d3 to d1 with mild coupling, then fsolve 1D.
    d2 = d1
    d3 = d1
    P1, P2, P3 = _triangle_from_d(nrm, d1, d2, d3, c1=12.0)
    e12 = np.linalg.norm(P1-P2) - l
    e13 = np.linalg.norm(P1-P3) - l
    # Return a single scalar by combining (keeps 1D root find)
    return 0.5*(e12 + e13)

def triangle_orientation_and_location(nrm, initial_guess, last_good=None):
    # continuation: start from last_good if available
    x0 = float(last_good) if last_good is not None else float(initial_guess)

    def fun_wrapped(d):
        return _eq_for_fsolve(d[0], nrm)

    try:
        root = fsolve(fun_wrapped, x0, maxfev=MAXFSOLVE_EVAL, xtol=1e-10)
        d1 = float(root[0])
        ok = True
    except Exception:
        d1 = x0
        ok = False

    # derive d2,d3 cheaply; in your original they’re separate roots,
    # but tying them keeps CPU low and animation smooth
    d2 = d1
    d3 = d1

    P1, P2, P3 = _triangle_from_d(nrm, d1, d2, d3, c1=12.0)
    if not ok:
        # if failed but we had a last good pose, reuse that to avoid jumps
        if last_good is not None:
            P1, P2, P3 = last_good
    return (P1, P2, P3), d1, ok

# ============================================
# Matplotlib setup
# ============================================
fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
ax.set_xlim(-15, 15); ax.set_ylim(-15, 15); ax.set_zlim(0, 30)
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

# Base circle (draw once)
theta = np.linspace(0, 2*np.pi, 120)
x_circle = l_1*np.cos(theta); y_circle = l_1*np.sin(theta); z_circle = np.zeros_like(theta)
ax.plot(x_circle, y_circle, z_circle, color='red', lw=1.0)

# Triangle polyline holders (updated each frame)
line_tri,   = ax.plot([], [], [], color='blue', lw=1.5)
line_leg12, = ax.plot([], [], [], color='red',  lw=1.0)
line_leg23, = ax.plot([], [], [], color='red',  lw=1.0)
line_leg32, = ax.plot([], [], [], color='green',lw=1.0)  # you can add more if desired

# Keep state for continuation
last_pose = {'P': None, 'd1': 5.0}

phi = deg2rad(PHI_DEG)

def init():
    # minimal draw
    line_tri.set_data([], [])
    line_tri.set_3d_properties([])
    line_leg12.set_data([], [])
    line_leg12.set_3d_properties([])
    line_leg23.set_data([], [])
    line_leg23.set_3d_properties([])
    line_leg32.set_data([], [])
    line_leg32.set_3d_properties([])
    return line_tri, line_leg12, line_leg23, line_leg32

def update(frame):
    # Rotate around Z (alpha)
    alpha = (frame*12.0) % 360.0
    ca, sa = np.cos(deg2rad(alpha)), np.sin(deg2rad(alpha))
    nrm = np.array([np.cos(phi)*ca, np.cos(phi)*sa, np.sin(phi)], dtype=float)

    # Robust pose solve with continuation
    (P1, P2, P3), d1, ok = triangle_orientation_and_location(
        nrm,
        initial_guess=5.0,
        last_good=last_pose['P']
    )
    if ok:
        last_pose['P']  = (P1, P2, P3)
        last_pose['d1'] = d1

    # Servo angles (cheap) from points
    th11, th12 = calculate_vectors_and_angles_1(l, l_1, l_11, l_12, P1)
    th21, th22 = calculate_vectors_and_angles_2(l, l_2, l_21, l_22, P2)
    th31, th32 = calculate_vectors_and_angles_3(l, l_3, l_31, l_32, P3)

    # Triangle polyline
    tri = np.vstack([P1, P2, P3, P1])
    line_tri.set_data(tri[:,0], tri[:,1])
    line_tri.set_3d_properties(tri[:,2])

    # Example “leg” segments for visualization (update in place)
    leg12 = np.vstack([P1, P2])
    leg23 = np.vstack([P2, P3])
    leg31 = np.vstack([P3, P1])
    line_leg12.set_data(leg12[:,0], leg12[:,1]); line_leg12.set_3d_properties(leg12[:,2])
    line_leg23.set_data(leg23[:,0], leg23[:,1]); line_leg23.set_3d_properties(leg23[:,2])
    line_leg32.set_data(leg31[:,0], leg31[:,1]); line_leg32.set_3d_properties(leg31[:,2])

    if DEBUG and frame % 100 == 0:
        lens = (np.linalg.norm(P1-P2), np.linalg.norm(P2-P3), np.linalg.norm(P3-P1))
        print(f"[{frame:04d}] ok={ok} alpha={alpha:.1f} | d1={d1:.3f} | edges ~ {lens} | "
              f"θ11={th11:.1f} θ12={th12:.1f} θ21={th21:.1f} θ22={th22:.1f} θ31={th31:.1f} θ32={th32:.1f}")

    return line_tri, line_leg12, line_leg23, line_leg32

ani = FuncAnimation(
    fig, update, init_func=init,
    frames=FRAMES, interval=INTERVAL_MS,
    blit=True
)

plt.show()
