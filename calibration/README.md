# Hand-Eye Calibration

Unified hand-eye calibration framework for HIROLRobotPlatform.

## Features

- ✅ **Robot-Agnostic**: Works with FR3, Monte01, UnitreeG1, and any robot supported by RobotMotion
- ✅ **Multi-Camera Support**: Select specific camera from multiple cameras in RobotMotion
- ✅ **Dual Calibration Types**: Eye-in-Hand and Eye-to-Hand
- ✅ **Multiple Board Types**: ChArUco (recommended) and ArUco markers
- ✅ **Automatic Collection**: Grid-based workspace sampling with random orientations
- ✅ **Manual Collection**: Interactive mode with real-time detection feedback
- ✅ **Quality Metrics**: Reprojection error, condition number, residual analysis
- ✅ **Multiple Solvers**: Tsai, Park, Horaud, Andreff, Daniilidis methods

## Quick Start

### 1. Prepare Calibration Board

**ChArUco Board (Recommended)**:
- Download and print CharuCO- board (12×9 squares, 30mm square, 22.5mm marker)
- Or generate custom board using OpenCV
- Mount board on rigid, flat surface

📘 **See [ChArUco Board Guide](config/CHARUCO_BOARDS.md) for detailed specifications, printing guidelines, and custom board generation**

### 2. Configure Robot and Cameras

Ensure your robot and cameras are configured in RobotMotion:

```yaml
# Example: factory/components/motion_configs/fr3_with_franka_hand_ik.yaml
sensor_dicts:
  cameras:
    - name: "wrist_camera"              # Camera 1 (for eye-in-hand)
      type: "realsense"
      cfg: !include hardware/sensors/cameras/config/d435i_ee_1.yaml

    - name: "external_camera"           # Camera 2 (for eye-to-hand)
      type: "realsense"
      cfg: !include hardware/sensors/cameras/config/d435i_external.yaml
```

**Note**: If you have multiple cameras, the calibration system will use **exact name matching** to select the correct one. See [Multi-Camera Setup Guide](config/MULTI_CAMERA_SETUP.md) for details.

### 3. Run Calibration

```bash
cd /path/to/HIROLRobotPlatform/calibration

# Automatic collection (recommended)
python hand_eye_calibration.py --config config/eye_in_hand_fr3_charuco.yaml

# Manual collection
python hand_eye_calibration.py --config config/eye_in_hand_fr3_charuco.yaml --manual
```

### 4. Results

Calibration results saved to:
```
calibration_data/
├── images/
│   ├── sample_0000.png
│   ├── sample_0001.png
│   └── ...
├── calibration_data.json       # Raw samples
└── calibration_result.json     # Final transformation
```

## Directory Structure

```
calibration/
├── README.md                         # This file
├── TODO.md                           # Future enhancements roadmap
├── hand_eye_calibration.py           # Main program
├── config/
│   ├── README.md                     # Configuration guide
│   ├── CHARUCO_BOARDS.md             # ChArUco board specifications and printing guide
│   ├── MULTI_CAMERA_SETUP.md         # Multi-camera setup guide
│   ├── eye_in_hand_fr3_charuco.yaml  # Eye-in-hand config
│   └── eye_to_hand_fr3_charuco.yaml  # Eye-to-hand config
├── board_detectors/
│   ├── base_detector.py              # Abstract detector interface
│   ├── charuco_detector.py           # ChArUco board detector
│   └── aruco_detector.py             # ArUco marker detector
├── utils/
│   ├── workspace_generator.py        # Pose generation
│   ├── calibration_solver.py         # AX=XB solver
│   └── data_manager.py               # Data persistence
└── tests/
    └── (unit tests)
```

## Usage Examples

### Example 1: Eye-in-Hand with Auto Collection

```bash
python calibration/hand_eye_calibration.py \
  --config calibration/config/eye_in_hand_fr3_charuco.yaml
```

**What happens**:
1. Robot moves to 27 grid positions (3×3×3)
2. At each position, detects ChArUco board
3. Records (robot_pose, board_pose) pairs
4. Solves for T_ee_camera using Tsai method
5. Saves result to `calibration_data/eye_in_hand_fr3_charuco/`

### Example 2: Eye-to-Hand with Manual Collection

```bash
python hand_eye_calibration.py \
  --config config/eye_to_hand_fr3_charuco.yaml \
  --manual
```

**What happens**:
1. Shows live camera feed with detection overlay
2. User moves robot manually (via teach pendant or code)
3. Press 'r' when board is detected to record sample
4. Repeat until sufficient samples collected (>12)
5. Press 'q' to finish collection
6. Automatically solves and saves calibration

### Example 3: Verify Existing Calibration

```bash
python hand_eye_calibration.py \
  --config config/eye_in_hand_fr3_charuco.yaml \
  --verify-only
```

## Calibration Types

### Eye-in-Hand

**Setup**: Camera mounted on robot end-effector

**Use Case**: Mobile manipulation, visual servoing

**Solves For**: `T_ee_camera` (end-effector to camera transform)

**Equation**: `T_base_board[i] = T_base_ee[i] @ T_ee_camera @ T_camera_board[i]`

### Eye-to-Hand

**Setup**: Camera fixed in world, robot moves board

**Use Case**: Pick-and-place, bin picking with fixed camera

**Solves For**: `T_base_camera` (robot base to camera transform)

**Equation**: `T_base_board[i] = T_base_camera @ T_camera_board[i]`

## Board Types

### ChArUco (Recommended)

**Advantages**:
- Subpixel corner accuracy
- Partial occlusion tolerance
- More stable pose estimation

**Configuration**:
```yaml
board:
  type: "charuco"
  square_length: 0.020  # 20mm
  marker_length: 0.015  # 15mm
  board_size: [14, 9]
  aruco_dict: "DICT_5X5_250"
```

📘 **For complete ChArUco board specifications, supported dictionaries, printing guidelines, and custom board generation, see [ChArUco Board Guide](config/CHARUCO_BOARDS.md)**

### ArUco

**Advantages**:
- Simpler setup (single marker)
- Faster detection

**Configuration**:
```yaml
board:
  type: "aruco"
  marker_length: 0.100  # 100mm
  aruco_dict: "DICT_6X6_250"
  marker_id: 0  # Optional
```

## Quality Guidelines

### Good Calibration Indicators

✅ Reprojection error < 1.5 pixels
✅ Condition number < 50
✅ Mean residual < 3mm
✅ 15-30 samples collected
✅ Wide spatial distribution
✅ Diverse orientations (±30°)

### Troubleshooting

**"Detection failed at multiple poses"**
- Check lighting conditions
- Ensure board is in focus
- Reduce `max_reprojection_error` threshold
- Print board at higher resolution

**"High condition number (>100)"**
- Increase workspace coverage
- Larger `spacing` or `grid_size`
- Ensure robot moves in all 3 axes
- Add more orientation diversity

**"Calibration unstable (large residuals)"**
- Collect more samples (>20)
- Increase `min_corner_count` for ChArUco
- Wait longer between poses (motion settling)
- Check for board/camera vibration

## Multi-Camera Scenarios

### When You Have Multiple Cameras

If your RobotMotion is configured with **multiple cameras** (e.g., 4 cameras), you can select which one to use for calibration.

**Step 1: Configure all cameras in motion_config**:
```yaml
# factory/components/motion_configs/your_robot_cfg.yaml
sensor_dicts:
  cameras:
    - name: "wrist_camera"           # Camera 1
      type: "realsense"
      cfg: !include hardware/sensors/cameras/config/d435i_ee_1.yaml

    - name: "external_camera_left"   # Camera 2
      type: "realsense"
      cfg: !include hardware/sensors/cameras/config/d435i_ext_left.yaml

    - name: "external_camera_right"  # Camera 3
      type: "realsense"
      cfg: !include hardware/sensors/cameras/config/d435i_ext_right.yaml

    - name: "overhead_camera"        # Camera 4
      type: "realsense"
      cfg: !include hardware/sensors/cameras/config/d435i_overhead.yaml
```

**Step 2: Select specific camera in calibration_config**:
```yaml
# calibration/config/eye_in_hand_fr3_charuco.yaml
calibration:
  camera:
    name: "wrist_camera"  # Exact match with motion_config
```

**Step 3: Run calibration**:
```bash
python hand_eye_calibration.py --config config/eye_in_hand_fr3_charuco.yaml

# Log output will show:
# INFO: Available cameras in RobotMotion: ['wrist_camera', 'external_camera_left', 'external_camera_right', 'overhead_camera']
# INFO: Selected camera: 'wrist_camera' (exact match)
```

**Important**:
- Camera names must **exactly match** (case-sensitive)
- See [Multi-Camera Setup Guide](config/MULTI_CAMERA_SETUP.md) for complete examples and troubleshooting

## Advanced Usage

### Custom Workspace Generation

Edit config file:
```yaml
workspace:
  center: [0.5, 0.0, 0.3]    # Grid center
  grid_size: [4, 4, 3]        # 48 poses
  spacing: [0.10, 0.10, 0.08] # 10cm XY, 8cm Z
  orientation_randomness: 35  # ±35° random rotation
```

### Different Solver Methods

Available methods (in `solver.method`):
- **Tsai**: Most robust, recommended (default)
- **Park**: Dual-quaternion formulation
- **Horaud**: Non-linear optimization
- **Andreff**: Screw motion based
- **Daniilidis**: Quaternion-based, rotation-first

### Integration with Other Robots

1. Create RobotMotion config for your robot
2. Copy and modify calibration config:
   ```yaml
   robot_motion_config: "path/to/your_robot_cfg.yaml"
   camera:
     name: "your_camera_name"
   workspace:
     # Adjust for your robot's reachable space
   ```
3. Run calibration (no code changes needed!)

## API Usage (Programmatic)

```python
from calibration.hand_eye_calibration import HandEyeCalibration

# Initialize
calib = HandEyeCalibration("config/eye_in_hand_fr3_charuco.yaml")

# Collect data
num_samples = calib.collect_data(auto=True)

# Solve
T_ee_camera, diagnostics = calib.calibrate()

# Verify
verification = calib.verify(T_ee_camera)

# Save
calib.save_result(T_ee_camera, diagnostics)

# Clean up
calib.close()
```

## Future Enhancements

This calibration system currently supports **single camera calibration**.

**Planned features** (see [TODO.md](TODO.md) for details):
- 🔲 Multi-camera parallel calibration (calibrate multiple cameras simultaneously)
- 🔲 Online calibration verification (real-time accuracy check)
- 🔲 Calibration board generator (auto-generate printable PDFs)
- 🔲 Semi-automatic collection mode
- 🔲 Real-time quality assessment
- 🔲 Support for AprilTag and other board types

Contributions are welcome! See [TODO.md](TODO.md) for implementation details.

## References

- **Tsai & Lenz (1989)**: "A New Technique for Fully Autonomous and Efficient 3D Robotics Hand/Eye Calibration"
- **OpenCV Hand-Eye Calibration**: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b
- **ChArUco Detection**: https://docs.opencv.org/4.x/df/d4a/tutorial_charuco_detection.html

## License

Part of HIROLRobotPlatform. See main repository for license.
