"""Lightweight kinematics helpers extracted from SPV4.py.

This module contains only the pure kinematics functions needed at run-time
and avoids executing plotting or top-level script code from the original file.
"""
import numpy as np
from scipy.optimize import fsolve

# Geometry constants (copied from SPV4.py defaults). If you have different
# geometry, set them when calling functions or modify these defaults.
l = 10
l_1 = 10
l_11 = 8
l_12 = 8
l_2 = 10
l_21 = 8
l_22 = 8
l_3 = 10
l_31 = 8
l_32 = 8


def position_and_orientation(nrm, S, l_val=l):
    """Compute platform vertex positions P1,P2,P3 from a normal `nrm` and center S.

    Returns dict with x1,x2,x3 as numpy arrays.
    """
    vector_i = np.array([1, 0, 0])
    cross_product = np.cross(nrm, vector_i)
    anorm = np.linalg.norm(cross_product)
    v_hat = cross_product / anorm
    a = l_val / (2 * np.cos(30 * np.pi / 180))
    cross_product = np.cross(v_hat, nrm)
    anorm = np.linalg.norm(cross_product)
    u_hat = cross_product / anorm
    x1 = S + a * v_hat
    x2 = S - a * np.sin(30. * np.pi / 180.0) * v_hat + a * np.cos(30. * np.pi / 180.) * u_hat
    x3 = S - a * np.sin(30. * np.pi / 180.0) * v_hat - a * np.cos(30. * np.pi / 180.) * u_hat
    return {"x1": x1, "x2": x2, "x3": x3}


def calculate_vectors_and_angles_1(l_val, l_1_val, l_11_val, l_12_val, x1, x2, x3):
    AA_tmp = l_12_val ** 2 - (l_1_val - x1[1]) ** 2 - l_11_val ** 2 - x1[2] ** 2
    bb_tmp = (2 * (l_1_val - x1[1]) * l_11_val)
    cc_tmp = (2 * x1[2] * l_11_val)
    denom_AA = np.sqrt(bb_tmp ** 2 + cc_tmp ** 2)
    cc = np.arccos((AA_tmp / denom_AA))
    bb1 = np.arccos(bb_tmp / denom_AA)
    theta_11 = cc - bb1
    theta_12 = np.arcsin((x1[2] - l_11_val * np.sin(theta_11)) / l_12_val)
    return {"theta_11": theta_11 * 180 / np.pi, "theta_12": theta_12 * 180 / np.pi}


def calculate_vectors_and_angles_2(l_val, l_2_val, l_21_val, l_22_val, x1, x2, x3):
    y_tmp = np.sqrt(x2[0] ** 2 + x2[1] ** 2)
    z_tmp = x2[2]
    AA_tmp = -(l_22_val ** 2 - (l_2_val - y_tmp) ** 2 - l_21_val ** 2 - z_tmp ** 2)
    bb_tmp = (2 * (l_2_val - y_tmp) * l_21_val)
    cc_tmp = (2 * z_tmp * l_21_val)
    denom_AA = np.sqrt(bb_tmp ** 2 + cc_tmp ** 2)
    cc = np.arcsin(AA_tmp / denom_AA)
    bb1 = np.arcsin(bb_tmp / denom_AA)
    theta_21 = cc + bb1
    theta_22 = np.arcsin((x2[2] - l_21_val * np.sin(theta_21)) / l_22_val)
    return {"theta_21": theta_21 * 180 / np.pi, "theta_22": theta_22 * 180 / np.pi}


def calculate_vectors_and_angles_3(l_val, l_3_val, l_31_val, l_32_val, x1, x2, x3):
    y_tmp = np.sqrt(x3[0] ** 2 + x3[1] ** 2)
    z_tmp = x3[2]
    AA_tmp = l_32_val ** 2 - (l_3_val - y_tmp) ** 2 - l_31_val ** 2 - z_tmp ** 2
    bb_tmp = (2 * (l_3_val - y_tmp) * l_31_val)
    cc_tmp = (2 * z_tmp * l_31_val)
    denom_AA = np.sqrt(bb_tmp ** 2 + cc_tmp ** 2)
    cc = np.arccos(AA_tmp / denom_AA)
    bb2 = np.arcsin(cc_tmp / denom_AA)
    theta_31 = cc - bb2
    theta_32 = np.arcsin((z_tmp - l_31_val * np.sin(theta_31)) / l_32_val)
    return {"theta_31": theta_31 * 180 / np.pi, "theta_32": theta_32 * 180 / np.pi}


def triangle_orientation_and_location(nrm1, S, initial_guess=5.0,
                                      l_val=l, l_1_val=l_1, l_11_val=l_11, l_12_val=l_12,
                                      l_2_val=l_2, l_21_val=l_21, l_22_val=l_22,
                                      l_3_val=l_3, l_31_val=l_31, l_32_val=l_32):
    """Solve for P1,P2,P3 and servo angles given a unit normal and center S.

    Returns a dict with P1,P2,P3 and theta_11, theta_12, theta_21, theta_22, theta_31, theta_32 (degrees).
    """
    n_x = nrm1[0]
    n_y = nrm1[1]
    n_z = nrm1[2]

    # Root solve for d_1 using one of the eeq functions structure
    def eeq1_local(d_1):
        A3 = 1 + (-n_x * np.sqrt(3) / 2 - n_y / 2) ** 2 / n_z ** 2
        BBB = d_1 - 2 * n_y * d_1 * (-n_x * np.sqrt(3) / 2 - n_y / 2) / n_z ** 2
        CCC = d_1 ** 2 + n_y ** 2 * d_1 ** 2 / n_z ** 2 - l_val ** 2
        A2 = 1 + (n_x * np.sqrt(3) / 2 - n_y / 2) ** 2 / n_z ** 2
        BB = d_1 - 2 * n_y * d_1 * (n_x * np.sqrt(3) / 2 - n_y / 2) / n_z ** 2
        CC = d_1 ** 2 + n_y ** 2 * d_1 ** 2 / n_z ** 2 - l_val ** 2
        s2 = np.sqrt(-4 * A2 * CC + BB ** 2)
        s3 = np.sqrt(-4 * A3 * CCC + BBB ** 2)
        eq1 = (n_x * n_y * ((-BBB + s3) * A2 + A3 * (BB - s2)) * ((-BBB + s3) * A2 - A3 * (BB - s2)) * np.sqrt(3) + 2 * (-4 * l_val ** 2 * n_z ** 2 * A3 ** 2 + (-BBB + s3) ** 2 * (n_z ** 2 + 0.3e1 / 0.4e1 * n_x ** 2 + n_y ** 2 / 4)) * A2 ** 2 - 2 * (BB - s2) * (n_z ** 2 + 0.3e1 / 0.2e1 * n_x ** 2 - n_y ** 2 / 2) * (-BBB + s3) * A3 * A2 + 2 * (BB - s2) ** 2 * A3 ** 2 * (n_z ** 2 + 0.3e1 / 0.4e1 * n_x ** 2 + n_y ** 2 / 4)) / n_z ** 2 / A2 ** 2 / A3 ** 2 / 8
        return eq1

    root = fsolve(eeq1_local, initial_guess)
    d_1 = float(root[0])

    A3 = 1 + (-n_x * np.sqrt(3) / 2 - n_y / 2) ** 2 / n_z ** 2
    BBB = d_1 - 2 * n_y * d_1 * (-n_x * np.sqrt(3) / 2 - n_y / 2) / n_z ** 2
    CCC = d_1 ** 2 + n_y ** 2 * d_1 ** 2 / n_z ** 2 - l_val ** 2
    A2 = 1 + (n_x * np.sqrt(3) / 2 - n_y / 2) ** 2 / n_z ** 2
    BB = d_1 - 2 * n_y * d_1 * (n_x * np.sqrt(3) / 2 - n_y / 2) / n_z ** 2
    CC = d_1 ** 2 + n_y ** 2 * d_1 ** 2 / n_z ** 2 - l_val ** 2
    s2 = np.sqrt(-4 * A2 * CC + BB ** 2)
    s3 = np.sqrt(-4 * A3 * CCC + BBB ** 2)
    d_2 = (-BB + s2) / (2 * A2)
    d_3 = (-BBB + s3) / (2 * A3)

    c_1 = 12
    c_2 = c_1 + (1 / n_z) * (-d_2 * np.cos(30 * np.pi / 180) * n_x + d_2 * np.sin(30 * np.pi / 180) * n_y + d_1 * n_y)
    c_3 = c_2 + (1 / n_z) * ((d_2 + d_3) * np.cos(30 * np.pi / 180) * n_x - (-d_3 + d_2) * np.sin(30 * np.pi / 180) * n_y)

    X1_hat = np.array([0, 1, 0])
    X2_hat = np.array([np.cos(30 * np.pi / 180), -np.sin(30 * np.pi / 180), 0])
    X3_hat = np.array([-np.cos(30 * np.pi / 180), -np.sin(30 * np.pi / 180), 0])
    z_hat = np.array([0, 0, 1])

    P1 = d_1 * X1_hat + c_1 * z_hat
    P2 = d_2 * X2_hat + c_2 * z_hat
    P3 = d_3 * X3_hat + c_3 * z_hat

    pp = (P1 + P2 + P3) / 3 - S
    P1 = P1 - pp
    P2 = P2 - pp
    P3 = P3 - pp

    # compute thetas
    r1 = calculate_vectors_and_angles_1(l_val, l_1_val, l_11_val, l_12_val, P1, P2, P3)
    r2 = calculate_vectors_and_angles_2(l_val, l_2_val, l_21_val, l_22_val, P1, P2, P3)
    r3 = calculate_vectors_and_angles_3(l_val, l_3_val, l_31_val, l_32_val, P1, P2, P3)

    return {
        "P1": P1,
        "P2": P2,
        "P3": P3,
        "theta_11": r1["theta_11"],
        "theta_12": r1["theta_12"],
        "theta_21": r2["theta_21"],
        "theta_22": r2["theta_22"],
        "theta_31": r3["theta_31"],
        "theta_32": r3["theta_32"],
    }


def inverse_kinematics(l_val, l_1_val, l_11_val, l_12_val, l_2_val, l_21_val, l_22_val, l_3_val, l_31_val, l_32_val, nrm, S):
    """Compatibility wrapper similar to original SPV4.inverse_kinematics."""
    return triangle_orientation_and_location(nrm, S, initial_guess=5.0,
                                             l_val=l_val, l_1_val=l_1_val, l_11_val=l_11_val, l_12_val=l_12_val,
                                             l_2_val=l_2_val, l_21_val=l_21_val, l_22_val=l_22_val,
                                             l_3_val=l_3_val, l_31_val=l_31_val, l_32_val=l_32_val)
