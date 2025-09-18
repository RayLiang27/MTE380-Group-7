import numpy as np

import matplotlib.pyplot as plt


def solve_two_link(motor, target, l1, l2, elbow_up=True):
    """
    Given motor point (S), target point (T), and link lengths l1, l2,
    return (joint1, theta_11, theta_12_global). joint2 == target by construction.
    theta_11 is the absolute angle of link1 from motor.
    theta_12_global is the absolute angle of link2 from joint1 (both in radians).
    """
    S = motor
    T = target
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


# Example usage
if __name__ == "__main__":
    w = 20.0   # motor offset from center (x)
    h = 100.0   # fulcrum height (y)
    l1 = 46.0  # linkage 1 length
    l2 = 85.0  # linkage 2 length
    l  = 50.0  # half beam length
    theta = 10.0  # beam angle (deg)

    plot_beam_balancer(w, h, l1, l2, l, elbow_up=False, theta=theta)
    plot_beam_balancer(w, h, l1, l2, l, elbow_up=False, theta=-theta)