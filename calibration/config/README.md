# Calibration Configuration Files

Configuration files for hand-eye calibration.

## Available Configurations

### Eye-in-Hand (Camera on Robot)

- **`eye_in_hand_fr3_charuco.yaml`**: FR3 robot with ChArUco board
  - Camera mounted on end-effector
  - Solves for T_ee_camera (end-effector to camera transform)

### Eye-to-Hand (Fixed Camera)

- **`eye_to_hand_fr3_charuco.yaml`**: FR3 robot with external camera
  - Camera fixed in world
  - Robot moves calibration board
  - Solves for T_base_camera (robot base to camera transform)

## Configuration Structure

```yaml
robot_motion_config: "path/to/robot_motion_cfg.yaml"

calibration:
  type: "eye_in_hand" or "eye_to_hand"

  board:
    type: "charuco" or "aruco"
    square_length: 0.020  # meters
    marker_length: 0.015  # meters
    board_size: [14, 9]   # [cols, rows]
    aruco_dict: "DICT_5X5_250"

  camera:
    name: "camera_name_in_robot_motion_config"
    min_corner_count: 8
    max_reprojection_error: 2.0

  workspace:
    center: [x, y, z]  # meters in base frame
    grid_size: [nx, ny, nz]
    spacing: [dx, dy, dz]  # meters
    orientation_randomness: 30  # degrees

  solver:
    method: "Tsai"  # or Park, Horaud, Andreff, Daniilidis
    min_samples: 10

  data_save_path: "path/to/save/directory"
```

## Creating Custom Configurations

### For Different Robots

1. Copy an existing config file
2. Update `robot_motion_config` to point to your robot's config
3. Adjust `workspace` parameters for your robot's reachable space
4. Update `camera.name` to match your camera configuration

### For Different Calibration Boards

#### ChArUco Board
```yaml
board:
  type: "charuco"
  square_length: 0.020  # Measure your board
  marker_length: 0.015
  board_size: [14, 9]   # Count squares (not markers!)
  aruco_dict: "DICT_5X5_250"
```

#### ArUco Marker
```yaml
board:
  type: "aruco"
  marker_length: 0.100  # Measure your marker
  aruco_dict: "DICT_6X6_250"
  marker_id: 0  # Optional: specific marker to use
```

### Workspace Tuning

**Eye-in-Hand:**
- `center`: Position where calibration board will be placed
- `grid_size`: [3, 3, 3] gives 27 poses (good starting point)
- `spacing`: 0.05-0.10m provides good coverage
- `orientation_randomness`: 20-40° ensures diverse orientations

**Eye-to-Hand:**
- `center`: Position visible to fixed camera
- `grid_size`: [4, 4, 3] or larger (48+ poses recommended)
- Keep board facing camera: reduce `orientation_randomness` to 15-25°

## Usage Examples

### Automatic Collection
```bash
cd /path/to/HIROLRobotPlatform/calibration
python hand_eye_calibration.py --config config/eye_in_hand_fr3_charuco.yaml
```

### Manual Collection
```bash
python hand_eye_calibration.py --config config/eye_in_hand_fr3_charuco.yaml --manual
# Move robot manually, press 'r' to record samples
```

### Verify Existing Data
```bash
python hand_eye_calibration.py --config config/eye_in_hand_fr3_charuco.yaml --verify-only
```

## Troubleshooting

### "Camera not found"
- Check `camera.name` matches RobotMotion config
- Ensure camera is initialized in RobotMotion

### "Insufficient samples"
- Increase `grid_size` or reduce `solver.min_samples`
- Check detection quality (low reprojection errors)

### "High condition number"
- Increase workspace coverage (larger `spacing` or `grid_size`)
- Ensure diverse orientations (`orientation_randomness` > 20°)

### Poor calibration accuracy
- Increase `camera.min_corner_count` (8-12 for ChArUco)
- Decrease `camera.max_reprojection_error` (1.0-1.5 for high accuracy)
- Collect more samples (20-30 recommended)
- Ensure stable robot motion (increase wait time between poses)

## References

- ChArUco boards: https://docs.opencv.org/4.x/df/d4a/tutorial_charuco_detection.html
- Hand-eye calibration: Tsai & Lenz (1989), "A New Technique for Fully Autonomous and Efficient 3D Robotics Hand/Eye Calibration"
