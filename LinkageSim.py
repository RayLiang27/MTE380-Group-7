import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def solve_two_link(motor, target, l1, l2, elbow_up=True):
    """
    Given motor point (S), target point (T), and link lengths l1, l2,
    return (joint1, theta_11, theta_12_global). joint2 == target by construction.
    theta_11 is the absolute angle of link1 from motor.
    theta_12_global is the absolute angle of link2 from joint1 (both in radians).
    """
    S = np.asarray(motor, float)
    T = np.asarray(target, float)
    v = T - S
    d = np.linalg.norm(v)
    # Reachability check
    if d > (l1 + l2) or d < abs(l1 - l2) or d < 1e-9:
        return None  # unreachable

    # Circle–circle intersection for joint1
    a = (l1**2 - l2**2 + d**2) / (2 * d)
    h_sq = max(l1**2 - a**2, 0.0)
    h = np.sqrt(h_sq)
    v_hat = v / d
    P2 = S + a * v_hat  # base point along S->T
    perp = np.array([-v_hat[1], v_hat[0]])
    J1 = P2 + (h * perp if elbow_up else -h * perp)

    # Angles (absolute/global)
    theta_11 = np.arctan2(J1[1] - S[1], J1[0] - S[0])
    theta_12_global = np.arctan2(T[1] - J1[1], T[0] - J1[0])

    return J1, theta_11, theta_12_global

def beam_points(fulcrum, l, alpha_deg, tie_offset=None):
    alpha = np.radians(alpha_deg)
    u = np.array([np.cos(alpha), np.sin(alpha)])
    # Endpoints of the beam (for drawing)
    left  = fulcrum - l * u
    right = fulcrum + l * u
    # Tie point along the beam (defaults to the right end)
    if tie_offset is None:
        tie = right.copy()
    else:
        tie = fulcrum + float(tie_offset) * u
    return left, right, tie

# -- NEW: sweep alpha and record required servo angle theta1 (deg)
def sweep_theta1_for_alpha(w, h, l1, l2, l, alpha_max=10.0, alpha_min=-10.0,
                           n=81, elbow_up=True, tie_offset=None, plot=True):
    fulcrum = np.array([0.0, float(h)])
    motor   = np.array([float(w), 0.0])

    alphas = np.linspace(alpha_max, alpha_min, n)  # +10° -> -10°
    thetas = np.full_like(alphas, np.nan, dtype=float)

    for i, a in enumerate(alphas):
        _, _, tie = beam_points(fulcrum, l, a, tie_offset)
        sol = solve_two_link(motor, tie, l1, l2, elbow_up=elbow_up)
        if sol is None:
            continue
        _, theta1, _ = sol
        thetas[i] = np.degrees(theta1)

    # Finite-diff slope d(alpha)/d(theta1) near alpha≈0 (precision metric)
    valid = np.isfinite(thetas)
    gain = None
    if valid.any():
        # interpolate θ1 at alpha=0 and neighbors
        try:
            theta_vs_alpha = np.interp(0.0, alphas[valid], thetas[valid])
            # small symmetric window around 0°
            eps = 0.5
            th_p = np.interp(eps,  alphas[valid], thetas[valid])
            th_m = np.interp(-eps, alphas[valid], thetas[valid])
            # dα/dθ ≈ (α_p - α_m) / (θ_p - θ_m)
            if np.isfinite(th_p) and np.isfinite(th_m) and abs(th_p - th_m) > 1e-9:
                gain = ( eps - (-eps) ) / (th_p - th_m)  # deg/deg at α≈0
        except Exception:
            pass

    if plot:
        plt.figure(figsize=(6,4))
        plt.plot(alphas[valid], thetas[valid], lw=2)
        plt.gca().invert_xaxis()  # visual: left->right corresponds +10 to -10
        plt.grid(True)
        plt.xlabel("Beam angle α (deg)")
        plt.ylabel("Servo angle θ₁ (deg)")
        plt.title("Required servo angle vs beam angle")
        plt.show()

    # Print a small report
    if valid.any():
        theta_span = np.nanmax(thetas) - np.nanmin(thetas)
        print(f"Valid α samples: {valid.sum()}/{n}")
        print(f"θ₁ span over α∈[{alpha_min:+.1f},{alpha_max:+.1f}]°: {theta_span:.2f} deg")
        if gain is not None:
            print(f"Precision near α≈0: dα/dθ₁ ≈ {gain:.4f} deg/deg "
                  f"(smaller = finer control)")
        else:
            print("Precision near α≈0: (not enough samples to estimate)")
    else:
        print("No reachable α in requested range (check lengths/geometry).")

    return alphas, thetas, gain

def plot_beam_balancer(w, h, l1, l2, l, elbow_up=True, theta=None):
    # θ (beam angle) may be provided; otherwise leave default
    theta = 0.0 if theta is None else np.radians(theta)

    fulcrum = np.array([0.0, float(h)])
    motor   = np.array([float(w), 0.0])

    # Beam endpoints (centered at fulcrum, rotated by theta)
    beam_left  = fulcrum + l * np.array([np.cos(theta + np.pi), np.sin(theta + np.pi)])
    beam_right = fulcrum + l * np.array([np.cos(theta),         np.sin(theta)])

    # IK: place joint2 exactly at beam_right
    sol = solve_two_link(motor, beam_right, l1, l2, elbow_up=elbow_up)
    if sol is None:
        # Unreachable – draw what we can and warn
        joint1 = motor  # collapsed
        joint2 = beam_right
        theta_11 = theta_12 = None
        unreachable = True
    else:
        joint1, theta_11, theta_12 = sol
        joint2 = beam_right
        unreachable = False

    # Plotting
    plt.figure(figsize=(200, 190))
    # Beam
    plt.plot([beam_left[0], beam_right[0]], [beam_left[1], beam_right[1]], 'b-', lw=4, label='Beam')
    # Fulcrum & motor
    plt.plot(fulcrum[0], fulcrum[1], 'ko', label='Fulcrum')
    plt.plot(motor[0], motor[1], 'ro', label='Motor')

    # Linkage 1: S -> J1
    if theta_11 is not None:
        plt.plot([motor[0], joint1[0]], [motor[1], joint1[1]], 'g-', lw=2, label='Linkage 1')
        plt.plot(joint1[0], joint1[1], 'go')
    else:
        plt.plot([motor[0]], [motor[1]], 'gx', label='Linkage 1 (unreachable)')

    # Linkage 2: J1 -> T (beam_right)
    if theta_12 is not None:
        plt.plot([joint1[0], joint2[0]], [joint1[1], joint2[1]], 'm-', lw=2, label='Linkage 2')
    else:
        plt.plot([beam_right[0]], [beam_right[1]], 'mx', label='Linkage 2 (unreachable)')
    plt.plot(joint2[0], joint2[1], 'mo')

    # Labels & cosmetics
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    title = 'Beam Balancer Linkage (Elbow Up)' if elbow_up else 'Beam Balancer Linkage (Elbow Down)'
    if unreachable:
        title += ' — UNREACHABLE (adjust lengths/positions)'
    plt.title(title)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.show()

def animate_mechanism(w, h, l1, l2, l,
                      alpha_path=None, elbow_up=True, tie_offset=None,
                      interval_ms=30, save_path=None):
    """
    Animate the linkage while the beam angle alpha follows alpha_path (deg).
    If alpha_path is None, we auto-build a sweep: +10 -> -10 -> +10.
    interval_ms = frame delay; save_path: '.gif' (Pillow) or '.mp4' (ffmpeg) optional.
    """
    fulcrum = np.array([0.0, float(h)])
    motor   = np.array([float(w), 0.0])

    if alpha_path is None:
        forward = np.linspace(+10.0, -10.0, 121)
        backward = forward[::-1]
        alpha_path = np.concatenate([forward, backward])

    # Build a figure and artists we will update
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_title("Beam Balancer – Animation")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')

    # Static points
    fulc_pt, = ax.plot([fulcrum[0]], [fulcrum[1]], 'ko', label='Fulcrum')
    motor_pt, = ax.plot([motor[0]], [motor[1]], 'ro', label='Motor')

    # Dynamic artists (placeholders; we set data in init)
    beam_ln,  = ax.plot([], [], 'b-', lw=4, label='Beam')
    link1_ln, = ax.plot([], [], 'g-', lw=2, label='Linkage 1')
    link2_ln, = ax.plot([], [], 'm-', lw=2, label='Linkage 2')
    j1_pt,    = ax.plot([], [], 'go')
    tie_pt,   = ax.plot([], [], 'mo')

    text_status = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    # Set axis limits once based on a rough sweep bounding box
    # (use the path extremes to get generous bounds)
    def rough_bounds():
        samples = np.linspace(np.min(alpha_path), np.max(alpha_path), 7)
        xs, ys = [], []
        for a in samples:
            left, right, tie = beam_points(fulcrum, l, a, tie_offset)
            xs += [left[0], right[0], motor[0], fulcrum[0], tie[0]]
            ys += [left[1], right[1], motor[1], fulcrum[1], tie[1]]
        pad = 0.15 * max(1.0, (max(xs) - min(xs)))
        ax.set_xlim(min(xs)-pad, max(xs)+pad)
        ax.set_ylim(min(ys)-pad, max(ys)+pad)

    rough_bounds()
    ax.legend(loc='best')

    # Per-frame update
    def update(frame):
        a_deg = float(alpha_path[frame])
        left, right, tie = beam_points(fulcrum, l, a_deg, tie_offset)
        sol = solve_two_link(motor, tie, l1, l2, elbow_up=elbow_up)

        # Beam line
        beam_ln.set_data([left[0], right[0]], [left[1], right[1]])

        if sol is None:
            # Unreachable – clear link lines, but still show beam & tie
            link1_ln.set_data([], [])
            link2_ln.set_data([], [])
            j1_pt.set_data([], [])
            tie_pt.set_data([tie[0]], [tie[1]])
            text_status.set_text(f"α={a_deg:+.1f}°  UNREACHABLE")
        else:
            j1, th1, th2 = sol
            # Link 1
            link1_ln.set_data([motor[0], j1[0]], [motor[1], j1[1]])
            # Link 2
            link2_ln.set_data([j1[0], tie[0]], [j1[1], tie[1]])
            # Points
            j1_pt.set_data([j1[0]], [j1[1]])
            tie_pt.set_data([tie[0]], [tie[1]])
            text_status.set_text(f"α={a_deg:+.1f}°  θ₁={np.degrees(th1):.1f}°")

        return (beam_ln, link1_ln, link2_ln, j1_pt, tie_pt, text_status)

    ani = FuncAnimation(fig, update, frames=len(alpha_path),
                        interval=interval_ms, blit=False, repeat=True)

    # Optional saving (GIF/MP4)
    if save_path:
        ext = str(save_path).lower()
        if ext.endswith(".gif"):
            try:
                from matplotlib.animation import PillowWriter
                ani.save(save_path, writer=PillowWriter(fps=max(1, int(1000/interval_ms))))
            except Exception as e:
                print(f"GIF save failed: {e}")
        elif ext.endswith(".mp4"):
            try:
                ani.save(save_path, writer='ffmpeg', fps=max(1, int(1000/interval_ms)))
            except Exception as e:
                print(f"MP4 save failed: {e}")
        else:
            print("Unsupported save extension (use .gif or .mp4).")

    plt.show()
    return ani


def evaluate_combo(w, h, l, l1, l2, tie_offset=None, elbow_up=True):
    alphas, thetas, gain0 = sweep_theta1_for_alpha(
        w, h, l1, l2, l,
        alpha_max=5.0, alpha_min=-5.0, n=101,
        elbow_up=elbow_up, tie_offset=tie_offset, plot=False
    )
    valid = np.isfinite(thetas)
    coverage = valid.mean()                # % of α in range that’s reachable (0..1)
    if coverage == 0:
        return {"score": -1e9, "coverage": 0}

    # Monotonicity: θ1(α) should be monotone (no reversals) for predictable control
    th = thetas[valid]
    al = alphas[valid]
    sign = np.sign(np.diff(th))
    rev = np.count_nonzero(np.diff(sign) != 0)  # reversals count (0 is best)

    # Servo span required for α±10°
    theta_span = float(np.nanmax(th) - np.nanmin(th)) if th.size else np.inf

    # Precision near center (smaller |dα/dθ1| is finer control; 0.08–0.15 good)
    precision = gain0  # deg/deg at α≈0 (may be None)
    target = 0.12
    prec_score = 0.0 if precision is None else np.exp(-((precision - target)**2) / (2*(0.04**2)))

    # Build a simple score (tweak weights to taste)
    score = (
        3.0 * coverage                           # must reach most of ±10°
        - 1.5 * rev                               # penalize reversals
        + 1.2 * prec_score                        # prefer target precision
        # + 0.6 * np.tanh((30.0 - theta_span) / 10) # avoid huge servo span
    )
    return {
        "score": float(score),
        "coverage": float(coverage),
        "reversals": int(rev),
        "theta_span_deg": float(theta_span),
        "gain_center_deg_per_deg": float(precision) if precision is not None else None,
        "elbow_up": bool(elbow_up),
        "tie_offset": None if tie_offset is None else float(tie_offset),
        "L1": float(l1), "L2": float(l2),
    }

def search_lengths(w, h, l, 
                   L1_vals=np.linspace(20, 120, 11),
                   L2_vals=np.linspace(20, 120, 11),
                   tie_offsets=(None,),             # or e.g., (None, 50) in mm from fulcrum
                   elbows=(True, False),
                   top_k=10):
    results = []
    for l1 in L1_vals:
        for l2 in L2_vals:
            for t in tie_offsets:
                for e in elbows:
                    m = evaluate_combo(w, h, l, l1, l2, tie_offset=t, elbow_up=e)
                    m["combo"] = {"L1": float(l1), "L2": float(l2), "elbow_up": e, "tie_offset": t}
                    results.append(m)
    results.sort(key=lambda d: d["score"], reverse=True)
    return results[:top_k], results

def print_top(results):
    print("Top candidates:")
    for i, r in enumerate(results, 1):
        g = r["gain_center_deg_per_deg"]
        gtxt = f"{g:.3f}" if g is not None else "n/a"
        print(f"[{i:02}] score={r['score']:+.2f}  L1={r['L1']:.1f}  L2={r['L2']:.1f}  "
              f"elbow_up={r['elbow_up']}  tie={r['tie_offset']}  "
              f"coverage={100*r['coverage']:.0f}%  reversals={r['reversals']}  "
              f"θspan={r['theta_span_deg']:.1f}°  dα/dθ₁@0={gtxt}")


# Example usage
if __name__ == "__main__":
    w = 20.0   # motor offset from center (x)
    h = 100.0   # fulcrum height (y)
    l1 = 46.0  # linkage 1 length
    l2 = 85.0  # linkage 2 length
    l  = 50.0  # half beam length
    theta = 5.0  # beam angle (deg)

        # Quick pass (coarse grid). Refine ranges around winners after first run.
    L1_vals = np.linspace(30, 60, 30)   # try small grids first
    L2_vals = np.linspace(70, 100, 30)

    top, allres = search_lengths(
        w, h, l, 
        L1_vals=L1_vals,
        L2_vals=L2_vals,
        tie_offsets=(None,),          # or (None, 40.0) to tie before the tip
        elbows=(False,),
        top_k=10
    )
    print_top(top)

    # Visualize θ1(α) for the winner to sanity-check:
    best = top[0]
    sweep_theta1_for_alpha(
        w, h, best["L1"], best["L2"], l,
        alpha_max=theta, alpha_min=-theta, n=101,
        elbow_up=best["elbow_up"], tie_offset=best["tie_offset"], plot=True
    )
    forward = np.linspace(theta, -theta, 120)
    back    = forward[::-1]
    alpha_path = np.concatenate([forward, back])

    animate_mechanism(w, h, best["L1"], best["L2"], l,
                    alpha_path=alpha_path,
                    elbow_up=False,
                    tie_offset=None,
                    interval_ms=30,
                    save_path=None) 