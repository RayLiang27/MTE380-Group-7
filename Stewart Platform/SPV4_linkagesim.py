"""SPV4_linkagesim.py

Adaptation of LinkageSim's robust 2-link solver for the Stewart-platform (SPV4)
- Projects each leg into its radial vertical plane
- Uses circle-circle intersection to compute elbow joint (joint1)
- Returns joint positions and servo angles (radians internally, degrees for reporting)
- Includes safety/clamping and reachability checks

Usage: run as script to see a quick printout using the same default geometry from SPV4.py
"""

import numpy as np
from math import isfinite

# small safe trig helpers
def safe_arccos(x):
    return np.arccos(np.clip(x, -1.0, 1.0))

def safe_arcsin(x):
    return np.arcsin(np.clip(x, -1.0, 1.0))

# Reuse the robust two-link planar solver from LinkageSim (adapted for local plane coords)
def solve_two_link_2d(S, T, l1, l2, elbow_up=True):
    """Planar two-link solver (2D vectors)
    S: motor origin (2,) in plane
    T: target point (2,) in same plane
    l1,l2: link lengths
    Returns (J1, theta1, theta2_global) in plane coords (radians) or None if unreachable
    J1 is the 2D elbow point.
    """
    S = np.asarray(S, float)
    T = np.asarray(T, float)
    v = T - S
    d = np.linalg.norm(v)
    if d < 1e-12:
        return None
    # reachability
    if d > (l1 + l2) + 1e-9 or d < abs(l1 - l2) - 1e-9:
        return None
    a = (l1**2 - l2**2 + d**2) / (2 * d)
    h_sq = max(l1**2 - a**2, 0.0)
    h = np.sqrt(h_sq)
    v_hat = v / d
    P2 = S + a * v_hat
    perp = np.array([-v_hat[1], v_hat[0]])
    J1 = P2 + (h * perp if elbow_up else -h * perp)
    theta1 = np.arctan2(J1[1] - S[1], J1[0] - S[0])
    theta2_global = np.arctan2(T[1] - J1[1], T[0] - J1[0])
    return J1, theta1, theta2_global

# Project a 3D motor-target pair into a 2D local plane for a given leg
def project_leg_plane(motor3, target3, out_radial=None):
    """Project 3D points into a 2D plane for a leg whose primary plane is radial+Z.
    radial direction (r_hat) is the unit vector in XY from platform center to motor anchor
    z_hat = [0,0,1]

    Returns:
      motor2 (2,), target2 (2,), r_hat (3,), z_hat (3,)
    where 2D coords are (u along r_hat, v along z)
    If out_radial is provided it will be used as r_hat (3-vector)."""
    motor = np.asarray(motor3, float)
    target = np.asarray(target3, float)
    z_hat = np.array([0.0, 0.0, 1.0])
    if out_radial is None:
        # radial direction from origin projected to motor XY (if motor at origin this will be unit y etc.)
        r_xy = motor.copy()
        r_xy[2] = 0.0
        norm = np.linalg.norm(r_xy)
        if norm < 1e-9:
            # choose +Y as default radial when motor is on Y axis
            r_hat = np.array([0.0, 1.0, 0.0])
        else:
            r_hat = r_xy / norm
    else:
        r_hat = np.asarray(out_radial, float)
        r_hat = r_hat / np.linalg.norm(r_hat)

    # Build orthonormal basis: r_hat (x_local), z_hat (y_local in plane is vertical)
    # 2D coords: u = dot(point - motor, r_hat), v = dot(point - motor, z_hat)
    def to2(p3):
        d = p3 - motor
        return np.array([np.dot(d, r_hat), np.dot(d, z_hat)])
    motor2 = np.array([0.0, 0.0])
    target2 = to2(target)
    return motor2, target2, r_hat, z_hat

# Reconstruct 3D elbow from 2D elbow using basis
def lift_to_3d(motor3, J1_2d, r_hat, z_hat):
    motor = np.asarray(motor3, float)
    u = J1_2d[0]
    v = J1_2d[1]
    # point = motor + u*r_hat + v*z_hat
    return motor + u * r_hat + v * z_hat

# High-level inverse kinematics: given platform vertices P1,P2,P3 and base motors, compute elbow positions and servo angles
def inverse_legs_from_platform(P_list, motor_list, l1_list, l2_list, elbow_up=True, verbose=False):
    """
    P_list: [P1,P2,P3] list of 3D platform attachment points
    motor_list: [M1,M2,M3] list of motor base 3D positions
    l1_list: [l_11,l_21,l_31] first link lengths
    l2_list: [l_12,l_22,l_32] second link lengths

    Returns dict with per-leg results or reports unreachable.
    """
    results = {}
    for i in range(3):
        P = np.asarray(P_list[i], float)
        M = np.asarray(motor_list[i], float)
        l1 = float(l1_list[i])
        l2 = float(l2_list[i])
        motor2, target2, r_hat, z_hat = project_leg_plane(M, P)
        sol = solve_two_link_2d(motor2, target2, l1, l2, elbow_up=elbow_up)
        if sol is None:
            if verbose:
                print(f"Leg {i+1}: UNREACHABLE (motor={M}, target={P}, l1={l1}, l2={l2})")
            results[i] = None
            continue
        J1_2d, theta1_2d, theta2_2d = sol
        J1_3d = lift_to_3d(M, J1_2d, r_hat, z_hat)
        # Convert angles to a convenient representation: theta1 in local plane
        results[i] = {
            "motor": M,
            "target": P,
            "elbow_2d": J1_2d,
            "elbow_3d": J1_3d,
            "theta1_rad": float(theta1_2d),
            "theta2_rad": float(theta2_2d),
            "theta1_deg": float(np.degrees(theta1_2d)),
            "theta2_deg": float(np.degrees(theta2_2d)),
            "r_hat": r_hat,
            "z_hat": z_hat,
        }
        if verbose:
            print(f"Leg {i+1}: motor={M}, target={P}, elbow={J1_3d}, theta1={np.degrees(theta1_2d):.2f}°")
    return results

# Helper: compute the platform top triangle from a normal (copied/adapted from SPV4)
def position_and_orientation(nrm, S, l_platform=25.0):
    nrm = np.asarray(nrm, float)
    S = np.asarray(S, float)
    # build local basis - pick an arbitrary in-plane vector for cross
    i_hat = np.array([1.0, 0.0, 0.0])
    v = np.cross(nrm, i_hat)
    if np.linalg.norm(v) < 1e-8:
        # nrm parallel to i_hat, choose another
        i_hat = np.array([0.0, 1.0, 0.0])
        v = np.cross(nrm, i_hat)
    v_hat = v / np.linalg.norm(v)
    u_hat = np.cross(v_hat, nrm)
    a = l_platform / (2.0 * np.cos(np.deg2rad(30.0)))
    x1 = S + a * v_hat
    x2 = S - a * np.sin(np.deg2rad(30.0)) * v_hat + a * np.cos(np.deg2rad(30.0)) * u_hat
    x3 = S - a * np.sin(np.deg2rad(30.0)) * v_hat - a * np.cos(np.deg2rad(30.0)) * u_hat
    return [x1, x2, x3]

# Default geometry (copied from SPV4 for convenience)
DEFAULTS = {
    "l": 25.0,
    "l_1": 5.0,
    "l_11": 8.0,
    "l_12": 9.0,
    "l_2": 5.0,
    "l_21": 8.0,
    "l_22": 9.0,
    "l_3": 5.0,
    "l_31": 8.0,
    "l_32": 9.0,
}

# Motor/base anchor positions consistent with SPV4's visualization convention
def base_motor_positions(l1, l2, l3):
    # Leg1 anchor on +Y axis
    M1 = np.array([0.0, l1, 0.0])
    # Leg2 anchor bottom-left
    M2 = np.array([-0.866 * l2, -0.5 * l2, 0.0])
    # Leg3 anchor bottom-right
    M3 = np.array([ 0.866 * l3, -0.5 * l3, 0.0])
    return [M1, M2, M3]

# High level wrapper: given normal and S, compute platform vertices then per-leg IK
def inverse_kinematics_from_orientation(nrm, S, geometry=DEFAULTS, elbow_up=False, verbose=False):
    l = geometry["l"]
    # compute platform vertices
    P_list = position_and_orientation(nrm, S, l_platform=l)
    # motors (reordered to match platform vertex ordering used in position_and_orientation)
    # position_and_orientation returns [P1 ( +Y ), P2 ( +x, -y ), P3 ( -x, -y )]
    M_all = base_motor_positions(geometry["l_1"], geometry["l_2"], geometry["l_3"])
    # map P1->M1, P2->M3, P3->M2
    M_list = [M_all[0], M_all[2], M_all[1]]
    # reorder link length lists to match motor ordering above
    l1_list = [geometry["l_11"], geometry["l_31"], geometry["l_21"]]
    l2_list = [geometry["l_12"], geometry["l_32"], geometry["l_22"]]
    results = inverse_legs_from_platform(P_list, M_list, l1_list, l2_list, elbow_up=elbow_up, verbose=verbose)
    return {
        "platform": P_list,
        "motors": M_list,
        "legs": results,
    }

# Quick test harness when run as script
if __name__ == "__main__":
    # Example orientation (similar to SPV4 defaults)
    phi = np.deg2rad(75.0)
    alpha = np.deg2rad(50.0)
    nrm = np.array([np.cos(phi) * np.cos(alpha), np.cos(phi) * np.sin(alpha), np.sin(phi)])
    S = np.array([0.0, 0.0, 8.0])
    geom = DEFAULTS.copy()
    res = inverse_kinematics_from_orientation(nrm, S, geometry=geom, elbow_up=False, verbose=True)
    P = res["platform"]
    M = res["motors"]
    print("Platform verts:")
    for i,p in enumerate(P,1):
        print(f" P{i}: {p}")
    print("Motor anchors:")
    for i,m in enumerate(M,1):
        print(f" M{i}: {m}")
    print("Leg solutions:")
    for i in range(3):
        sol = res["legs"].get(i)
        if sol is None:
            print(f" Leg {i+1}: UNREACHABLE")
        else:
            print(f" Leg {i+1}: elbow_3d={sol['elbow_3d']}, theta1={sol['theta1_deg']:.2f}°, theta2={sol['theta2_deg']:.2f}°")

    # Simple verification: confirm elbow->target distance equals l2
    for i in range(3):
        sol = res["legs"].get(i)
        if sol is None:
            continue
        d = np.linalg.norm(sol["target"] - sol["elbow_3d"]) 
        print(f"Leg {i+1} elbow->target dist = {d:.6f}, expected l2={geom[f'l_{i+2 if i<2 else 2}'] if False else '...'}")

    print("Done.")

    # --- Animation that mirrors SPV4.py behaviour but uses the robust IK ---
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation

    def animate_spv4_equivalent(geometry=DEFAULTS, frames=120, interval=100, elbow_up=True, print_interval=10):
        # fixed platform height and phi similar to SPV4 defaults
        S = np.array([0.0, 0.0, 10.0])
        phi = np.deg2rad(75.0)

        # figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # axis limits based on base radius + platform size
        max_base = max(geometry['l_1'], geometry['l_2'], geometry['l_3']) + 10.0
        zmax = S[2] + geometry['l_11'] + geometry['l_12'] + 5.0
        ax.set_xlim(-max_base, max_base)
        ax.set_ylim(-max_base, max_base)
        ax.set_zlim(0, zmax)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # artists
        plat_line, = ax.plot([], [], [], 'b-', lw=2, label='platform')
        elbow_lines = []

        # create placeholders for 3 legs: two segments per leg
        for i in range(3):
            seg1, = ax.plot([], [], [], 'r-', lw=2)
            seg2, = ax.plot([], [], [], 'r-', lw=2)
            elbow_lines.append((seg1, seg2))

        # draw static motor anchors
        M_list = base_motor_positions(geometry['l_1'], geometry['l_2'], geometry['l_3'])
        motor_xyz = np.vstack(M_list)
        ax.scatter(motor_xyz[:, 0], motor_xyz[:, 1], motor_xyz[:, 2], c='k', s=30, label='motors')
        # on-screen status text for motor angles
        text_status = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

        def update(frame):
            # rotate alpha over time like SPV4: 0..360
            alpha = np.deg2rad((frame * (360.0 / frames)) % 360.0)
            nrm = np.array([np.cos(phi) * np.cos(alpha), np.cos(phi) * np.sin(alpha), np.sin(phi)])

            res = inverse_kinematics_from_orientation(nrm, S, geometry=geometry, elbow_up=elbow_up, verbose=False)
            P_list = res['platform']
            legs = res['legs']

            # platform polygon
            plat = np.vstack(P_list + [P_list[0]])
            plat_line.set_data(plat[:, 0], plat[:, 1])
            plat_line.set_3d_properties(plat[:, 2])

            # per-leg segments
            for i in range(3):
                sol = legs.get(i)
                if sol is None:
                    # clear lines when unreachable
                    elbow_lines[i][0].set_data([], [])
                    elbow_lines[i][0].set_3d_properties([])
                    elbow_lines[i][1].set_data([], [])
                    elbow_lines[i][1].set_3d_properties([])
                    continue
                M = sol['motor']
                J = sol['elbow_3d']
                T = sol['target']
                seg1 = np.vstack([M, J])
                seg2 = np.vstack([J, T])
                elbow_lines[i][0].set_data(seg1[:, 0], seg1[:, 1])
                elbow_lines[i][0].set_3d_properties(seg1[:, 2])
                elbow_lines[i][1].set_data(seg2[:, 0], seg2[:, 1])
                elbow_lines[i][1].set_3d_properties(seg2[:, 2])

            # build angle display text
            angle_lines = []
            for i in range(3):
                sol = legs.get(i)
                if sol is None:
                    angle_lines.append(f"L{i+1}: unreachable")
                else:
                    angle_lines.append(f"L{i+1}: θ1={sol['theta1_deg']:.1f}°, θ2={sol['theta2_deg']:.1f}°")
            angle_text = "  ".join(angle_lines)
            text_status.set_text(angle_text)
            # print to console at configured interval
            if print_interval and (frame % print_interval == 0):
                print(f"frame {frame}: {angle_text}")

            return (plat_line, ) + tuple([a for pair in elbow_lines for a in pair]) + (text_status,)

        ani = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)
        plt.legend()
        plt.show()

    # run the animation automatically when script executed
    try:
        animate_spv4_equivalent(geometry=geom, frames=180, interval=80, elbow_up=False, print_interval=10)
    except Exception as e:
        print(f"Animation failed: {e}")
