"""Stewart Platform Physical Dimensions and Conventions

This file documents the physical dimensions and angle conventions for the Stewart
Platform, serving as the central reference for all kinematics calculations.

Coordinate System:
- Origin: Center of base plate
- Z-axis: Vertical up
- X-axis: Points toward servo 1
- Y-axis: Follows right-hand rule

Platform Geometry:
- Equilateral triangle arrangement
- Three servos mounted at 120° intervals on base
- Each leg consists of two links connected by an intermediate joint

Physical Dimensions (mm):
"""
import numpy as np

# Platform dimensions
PLATFORM_SIDE_LENGTH = 120.0  # Side length of equilateral top platform

# Base geometry
BASE_RADIUS = 100.0  # Distance from center to servo mount points
SERVO_HEIGHT = 60.0  # Height of servo center above base (z)
SERVO_OFFSET = 20.0  # Horizontal offset of servo from mount point (y)

# Link lengths
LINK_1_LENGTH = 40.0  # Length of first link (servo to intermediate joint)
LINK_2_LENGTH = 90.0  # Length of second link (intermediate to platform)

# Servo parameters
SERVO_NEUTRAL_ANGLE = 50.0  # Degrees
SERVO_MIN_ANGLE = 0.0       # Physical limit
SERVO_MAX_ANGLE = 70.0      # Physical limit
SERVO_RANGE = SERVO_MAX_ANGLE - SERVO_MIN_ANGLE

# Conversion helpers
def kinematic_to_servo_angle(kinematic_angle_deg):
    """Convert kinematic angle to physical servo angle.
    
    Args:
        kinematic_angle_deg: Angle from kinematics calculation where:
            0° = horizontal position
            positive = up from horizontal
            negative = down from horizontal
            
    Returns:
        Physical servo angle (0-70°) where:
            50° = neutral position
            0° = maximum down position
            70° = maximum up position
    """
    # First convert kinematic angle to servo frame
    servo_frame = SERVO_NEUTRAL_ANGLE - kinematic_angle_deg
    
    # Clamp to physical limits
    return np.clip(servo_frame, SERVO_MIN_ANGLE, SERVO_MAX_ANGLE)

def servo_to_kinematic_angle(servo_angle_deg):
    """Convert physical servo angle to kinematic angle.
    
    Args:
        servo_angle_deg: Physical servo angle (0-70°)
            
    Returns:
        Angle in kinematics frame where 0° is horizontal
    """
    # Clamp input to valid range
    servo_angle = np.clip(servo_angle_deg, SERVO_MIN_ANGLE, SERVO_MAX_ANGLE)
    
    # Convert to kinematics frame
    return SERVO_NEUTRAL_ANGLE - servo_angle