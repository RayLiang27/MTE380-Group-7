# Stewart Platform Kinematics Solver (SPV4.py)

## Overview
This script provides a mathematical model and visualization tools for a 3-DOF (Degrees of Freedom) Stewart Platform. It calculates both forward and inverse kinematics for positioning the platform and includes 3D visualization capabilities.

## Features
- Forward Kinematics: Calculate platform position from actuator angles
- Inverse Kinematics: Calculate required actuator angles for desired position
- 3D Visualization: Real-time visualization of the platform's state
- Position and Orientation Solver: Compute leg positions for given platform orientation

## Usage

### 1. Platform Configuration
Set up the physical dimensions of your Stewart Platform:
```python
# Platform dimensions (in mm or your preferred unit)
l = 10    # Base radius
l_1 = 10  # Leg 1 length
l_11 = 8  # Leg 1 segment 1
l_12 = 8  # Leg 1 segment 2
# ... similarly for legs 2 and 3
```

### 2. Position and Orientation
Define the desired platform position and orientation:
```python
# Example orientation
phi = 70 * np.pi / 180    # Elevation angle (radians)
alpha = 50 * np.pi / 180  # Azimuth angle (radians)
nrm = np.array([np.cos(phi)*np.cos(alpha), 
                np.cos(phi)*np.sin(alpha), 
                np.sin(phi)])  # Normal vector
S = np.array([0, 0, 17.05])   # Platform center position
```

### 3. Solve Kinematics
```python
# Get leg positions
result = position_and_orientation(nrm, S)
x1 = result["x1"]  # Position of leg 1
x2 = result["x2"]  # Position of leg 2
x3 = result["x3"]  # Position of leg 3

# Calculate angles
angles = calculate_vectors_and_angles_1(l, l_1, l_11, l_12, x1, x2, x3)
```

### 4. Visualization
The script includes built-in visualization functions that will show:
- Base platform
- Moving platform
- Leg positions and orientations
- Real-time movement animations

## Key Functions

### `position_and_orientation(nrm, S)`
Calculates the position of each leg attachment point given:
- `nrm`: Normal vector of the platform orientation
- `S`: Position vector of the platform center
Returns: Dictionary with positions x1, x2, x3

### `calculate_vectors_and_angles_1(l, l_1, l_11, l_12, x1, x2, x3)`
Calculates the required actuator angles given:
- `l`: Base radius
- `l_1`: Leg length
- `l_11`, `l_12`: Leg segment lengths
- `x1`, `x2`, `x3`: Leg positions
Returns: Dictionary with angles and vectors for each leg

## Dependencies
- NumPy: For numerical computations
- SciPy: For optimization and special functions
- Matplotlib: For 3D visualization

## Tips for Usage
1. Always work in consistent units (all lengths in mm or all in meters)
2. Angles are handled in radians internally
3. Check the physical limits of your actuators when using inverse kinematics
4. The visualization can help debug unexpected platform behaviors

## Error Handling
- The script includes checks for:
  - Physical constraints of the legs
  - Reachable workspace limits
  - Valid angle ranges

## Performance Notes
- The forward kinematics solver uses numerical optimization
- For real-time control, cache results when possible
- The visualization can be disabled for faster computation