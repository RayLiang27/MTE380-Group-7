"""Stewart Platform Kinematics

This module implements the inverse kinematics for a 3-DOF Stewart Platform,
mapping desired platform orientation to servo angles while respecting the
physical constraints of the mechanism.

Coordinate System:
- Origin at base center
- Z-axis vertical up
- X-axis toward servo 1 (front)
- Y-axis follows right-hand rule

Physical Setup:
- Three servos mounted at 120° intervals
- Each leg has two links connected by an intermediate joint
- Servos have 50° neutral position and 0-70° range
- Platform is an equilateral triangle

Angle Convention:
- Kinematics frame: 0° is horizontal, positive up
- Servo frame: 70° is fully down, 50° neutral, 0° is up
- Function inputs/outputs use servo frame (0-70°)
- IMPORTANT: Servo angles decrease as arm moves up!
"""
import numpy as np
from scipy.optimize import fsolve
from typing import Tuple, List

# Physical dimensions (mm)
PLATFORM_SIDE_LENGTH = 160.0  # Side length of platform triangle
BASE_RADIUS = 120.0          # Radius to servo mounts
SERVO_HEIGHT = 60.0          # Height of servo from base
SERVO_OFFSET = 20.0          # Horizontal offset from center
LINK_1_LENGTH = 40.0         # First link length
LINK_2_LENGTH = 90.0         # Second link length

# Servo parameters 
SERVO_NEUTRAL = 50  # Degrees, neutral position
SERVO_MIN = 0      # Degrees, maximum up position
SERVO_MAX = 70     # Degrees, maximum down position

def kinematic_to_servo_angle(angle: float) -> float:
    """Convert from kinematic frame (0° horizontal, positive up) to servo frame.
    
    Args:
        angle: Angle in degrees from kinematics calculation (positive = up)
        
    Returns:
        Servo angle (70° = down, 50° = neutral, 0° = up)
        Lower angles correspond to higher arm positions
    """
    # Invert angle since servo angles decrease as arm moves up
    servo_angle = SERVO_NEUTRAL - angle
    return np.clip(servo_angle, SERVO_MIN, SERVO_MAX)

def servo_to_kinematic_angle(angle: float) -> float:
    """Convert from servo frame (70° down, 0° up) to kinematic frame.
    
    Args:
        angle: Servo angle in degrees (70° = down, 50° = neutral, 0° = up)
        
    Returns:
        Kinematic angle where 0° is horizontal and positive = up
    """
    # Invert the difference since servo angles work in opposite direction
    return SERVO_NEUTRAL - angle


def position_and_orientation(nrm: np.ndarray, S: np.ndarray, platform_side: float = PLATFORM_SIDE_LENGTH) -> dict:
    """Compute platform vertex positions from normal vector and center point.
    
    Given a unit normal vector and center point, computes the vertices of the
    equilateral triangle platform. The first vertex (P1) faces the x-axis
    when the platform is level.
    
    Args:
        nrm: Unit normal vector [nx, ny, nz] defining platform orientation
        S: Platform center position [x, y, z]
        platform_side: Side length of equilateral platform triangle (mm)
        
    Returns:
        dict with keys:
            x1, x2, x3: Vertex positions as numpy arrays [x,y,z]
            Note: Vertices are ordered counterclockwise when viewed from above
    """
    # Verify normal is unit length
    if not np.isclose(np.linalg.norm(nrm), 1.0, rtol=1e-3):
        raise ValueError("Normal vector must be unit length")
        
    # Calculate platform radius (center to vertex)
    radius = platform_side / (2 * np.cos(np.pi/6))  # 30° = π/6 rad
    
    # Find orthonormal basis for platform plane
    x_axis = np.array([1, 0, 0])
    v_hat = np.cross(nrm, x_axis)
    v_hat = v_hat / np.linalg.norm(v_hat)  # Normalize
    u_hat = np.cross(v_hat, nrm)  # Already unit length since v_hat ⊥ nrm
    
    # Calculate vertices using 120° rotations
    x1 = S + radius * v_hat  # Front vertex
    angle = np.pi/6  # 30°
    x2 = S - radius * (np.sin(angle) * v_hat - np.cos(angle) * u_hat)  # Back right
    x3 = S - radius * (np.sin(angle) * v_hat + np.cos(angle) * u_hat)  # Back left
    
    return {
        "x1": x1,
        "x2": x2,
        "x3": x3,
    }


def calculate_servo1_angles(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> dict:
    """Calculate angles for servo 1 (front servo) given platform vertex positions.
    
    This computes the two angles needed for the 2-link leg connecting the
    base to vertex x1. The angles are calculated in the kinematics frame
    (0° = horizontal) and then converted to physical servo angles.
    
    Args:
        x1, x2, x3: Platform vertex positions from position_and_orientation()
        
    Returns:
        dict with:
            theta_1: First joint angle in servo frame (0-70° range)
            theta_2: Second joint angle in servo frame (0-70° range)
    """
    # Get target point relative to servo mount point
    servo_mount = np.array([0, SERVO_OFFSET, SERVO_HEIGHT])
    p = x1 - servo_mount
    
    # Law of cosines to find first angle
    r_xy = np.sqrt(p[0]**2 + p[1]**2)  # Horizontal distance
    r = np.sqrt(r_xy**2 + p[2]**2)     # Total distance
    
    # Check if point is reachable
    if r > (LINK_1_LENGTH + LINK_2_LENGTH):
        raise ValueError(
            f"Point {x1} too far to reach with links "
            f"{LINK_1_LENGTH}, {LINK_2_LENGTH}"
        )
    if r < abs(LINK_1_LENGTH - LINK_2_LENGTH):
        raise ValueError(
            f"Point {x1} too close to reach with links "
            f"{LINK_1_LENGTH}, {LINK_2_LENGTH}"
        )
    
    # Calculate angles in kinematic frame
    cos_theta2 = (LINK_1_LENGTH**2 + LINK_2_LENGTH**2 - r**2) / (2 * LINK_1_LENGTH * LINK_2_LENGTH)
    theta2 = np.arccos(np.clip(cos_theta2, -1, 1))
    
    beta = np.arccos(np.clip(
        (LINK_1_LENGTH**2 + r**2 - LINK_2_LENGTH**2) / (2 * LINK_1_LENGTH * r), -1, 1
    ))
    alpha = np.arctan2(p[2], r_xy)
    theta1 = alpha + beta
    
    # Convert to degrees and map to servo frame
    theta1_deg = np.rad2deg(theta1)
    theta2_deg = np.rad2deg(theta2)
    
    # Map to physical servo angles (includes validation)
    return {
        "theta_1": kinematic_to_servo_angle(theta1_deg),
        "theta_2": kinematic_to_servo_angle(theta2_deg)
    }





class StewartPlatform:
    """Helper class for Stewart platform kinematics calculations."""

    def __init__(self):
        """Initialize platform parameters from constants."""
        # Physical dimensions
        self.platform_side = PLATFORM_SIDE_LENGTH 
        self.base_radius = BASE_RADIUS
        self.servo_height = SERVO_HEIGHT
        self.servo_offset = SERVO_OFFSET
        self.link1 = LINK_1_LENGTH
        self.link2 = LINK_2_LENGTH

        # Derived dimensions
        self.platform_radius = self.platform_side / (2 * np.cos(np.pi/6))
        
        # Servo positions (120° spacing)
        self.servo_angles = np.array([0, 120, 240]) * np.pi/180
        
    def get_platform_vertices(self, normal: np.ndarray, center: np.ndarray) -> dict:
        """Calculate platform vertex positions given orientation and center.
        
        Args:
            normal: Unit normal vector [nx, ny, nz] defining platform orientation
            center: Platform center position [x, y, z]
            
        Returns:
            dict with vertex positions P1, P2, P3 as numpy arrays
            
        Raises:
            ValueError: If normal vector is not unit length
        """
        # Verify normal is unit length
        if not np.isclose(np.linalg.norm(normal), 1.0, rtol=1e-3):
            raise ValueError("Normal vector must be unit length")
            
        # Find orthonormal basis for platform plane
        x_axis = np.array([1, 0, 0]) 
        v_hat = np.cross(normal, x_axis)
        v_hat = v_hat / np.linalg.norm(v_hat)
        u_hat = np.cross(v_hat, normal)
        
        # Calculate vertex positions
        vertices = {}
        angle = np.pi/6  # 30°
        
        vertices["P1"] = center + self.platform_radius * v_hat
        vertices["P2"] = center - self.platform_radius * (
            np.sin(angle) * v_hat - np.cos(angle) * u_hat
        )
        vertices["P3"] = center - self.platform_radius * (
            np.sin(angle) * v_hat + np.cos(angle) * u_hat
        )
        
        return vertices
        
    def get_servo_angles(self, vertices: dict) -> dict:
        """Calculate servo angles for given platform vertex positions.
        
        Args:
            vertices: dict with platform vertex positions (P1, P2, P3)
            
        Returns:
            dict with servo angles (in servo frame 0-70°):
                servo1, servo2: angles for first leg
                servo3, servo4: angles for second leg
                servo5, servo6: angles for third leg
        """
        angles = {}
        
        # Calculate angles for each leg
        leg1 = self.calculate_leg_angles(vertices["P1"])
        leg2 = self.calculate_leg_angles(
            self._rotate_point(vertices["P2"], -2*np.pi/3)
        )
        leg3 = self.calculate_leg_angles(
            self._rotate_point(vertices["P3"], 2*np.pi/3)
        )
        
        angles.update({
            "servo1": leg1["theta_1"],
            "servo2": leg1["theta_2"], 
            "servo3": leg2["theta_1"],
            "servo4": leg2["theta_2"],
            "servo5": leg3["theta_1"], 
            "servo6": leg3["theta_2"]
        })
        
        return angles
        
    def calculate_leg_angles(self, target: np.ndarray) -> dict:
        """Calculate servo angles to reach target point.
        
        Args:
            target: Target point [x, y, z] for leg endpoint
            
        Returns:
            dict with:
                theta_1: First joint angle in servo frame (0-70°)
                theta_2: Second joint angle in servo frame (0-70°)
                
        Raises:
            ValueError: If point is unreachable
        """
        # Get point relative to servo mount
        servo_mount = np.array([0, self.servo_offset, self.servo_height])
        p = target - servo_mount
        
        # Check reach
        r_xy = np.sqrt(p[0]**2 + p[1]**2)
        r = np.sqrt(r_xy**2 + p[2]**2)
        
        if r > (self.link1 + self.link2):
            raise ValueError(f"Target {target} too far")
        if r < abs(self.link1 - self.link2):
            raise ValueError(f"Target {target} too close")
            
        # Law of cosines for angles
        cos_theta2 = (self.link1**2 + self.link2**2 - r**2) / (
            2 * self.link1 * self.link2
        )
        theta2 = np.arccos(np.clip(cos_theta2, -1, 1))
        
        beta = np.arccos(np.clip(
            (self.link1**2 + r**2 - self.link2**2) / (
                2 * self.link1 * r
            ), -1, 1
        ))
        alpha = np.arctan2(p[2], r_xy)
        theta1 = alpha + beta
        
        # Convert to degrees and map to servo frame
        theta1_deg = np.rad2deg(theta1) 
        theta2_deg = np.rad2deg(theta2)
        
        return {
            "theta_1": kinematic_to_servo_angle(theta1_deg),
            "theta_2": kinematic_to_servo_angle(theta2_deg)
        }
        
    def _rotate_point(self, point: np.ndarray, angle: float) -> np.ndarray:
        """Rotate point around z-axis by given angle."""
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return R @ point
        
def setup_platform() -> StewartPlatform:
    """Create and configure a StewartPlatform instance.
    
    Returns:
        Configured StewartPlatform object
    """
    return StewartPlatform()

def solve_platform_kinematics(normal: np.ndarray, center: np.ndarray
) -> tuple[dict, dict]:
    """Solve inverse kinematics for desired platform pose.
    
    This is the main entry point for kinematics calculations. Given a desired
    platform orientation and position, it computes both vertex positions and
    the required servo angles.
    
    Args:
        normal: Unit normal vector [nx, ny, nz] defining platform orientation
        center: Platform center position [x, y, z]
        
    Returns:
        tuple of:
            vertices: dict with platform vertex positions
            angles: dict with servo angles in servo frame (0-70°)
            
    Raises:
        ValueError: If solution cannot be found or angles exceed limits
    """
    platform = setup_platform()
    vertices = platform.get_platform_vertices(normal, center)
    angles = platform.get_servo_angles(vertices)
    return vertices, angles
    
    # Calculate other heights using geometric relationships
    c_2 = c_1 + (1/n_z) * (
        -d_2 * np.cos(30 * np.pi/180) * n_x +
        d_2 * np.sin(30 * np.pi/180) * n_y +
        d_1 * n_y
    )
    c_3 = c_2 + (1/n_z) * (
        (d_2 + d_3) * np.cos(30 * np.pi/180) * n_x -
        (-d_3 + d_2) * np.sin(30 * np.pi/180) * n_y
    )

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


def legacy_inverse_kinematics(nrm: np.ndarray, S: np.ndarray) -> dict:
    """Legacy compatibility function for old SPV4 interface.
    
    This function maintains compatibility with code that expects the original
    SPV4.inverse_kinematics interface. It converts the new interface back to
    the old format.
    
    Args:
        nrm: Unit normal vector [nx, ny, nz]
        S: Platform center position [x, y, z]
        
    Returns:
        dict with original SPV4 format angles (theta_11 through theta_32)
    """
    vertices, angles = solve_platform_kinematics(nrm, S)
    return {
        "theta_11": angles["servo1"],
        "theta_12": angles["servo2"],
        "theta_21": angles["servo3"],
        "theta_22": angles["servo4"],
        "theta_31": angles["servo5"],
        "theta_32": angles["servo6"]
    }


# Example usage with actual dimensions:
if __name__ == "__main__":
    # Create a platform instance
    platform = setup_platform()
    
    # Test a simple orientation - platform tilted 10° around x-axis
    angle = 10 * np.pi/180
    normal = np.array([np.sin(angle), 0, np.cos(angle)])
    center = np.array([0, 0, 130])  # 130mm above base
    
    try:
        vertices, angles = solve_platform_kinematics(normal, center)
        print("\nPlatform vertices:")
        for name, pos in vertices.items():
            print(f"{name}: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]")
        
        print("\nServo angles:")
        for name, angle in angles.items():
            print(f"{name}: {angle:.1f}°")
            
    except ValueError as e:
        print(f"Error: {e}")
