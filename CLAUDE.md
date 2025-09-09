# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HIROLRobotPlatform is a unified robot control framework enabling cross-vendor robot collaboration through hardware abstraction and centralized task planning. It supports multiple robot hardware platforms, simulation environments, control algorithms, and learning/inference capabilities.

## Setup and Installation

```bash
# Initialize git submodules (required for robot SDKs)
git submodule update --init --recursive

# Install the platform in editable mode
pip install -e .
```

**Note**: The requirements.txt has merge conflict markers (lines 170-206) that need resolution.

## Core Architecture

### Factory Pattern Design
- **Central Factory**: `factory/components/robot_factory.py` - RobotFactory class manages all robot system components
- **Hardware Abstraction**: Base classes (ArmBase, ToolBase, SimBase, CameraBase) provide unified interfaces
- **Configuration-Driven**: YAML configs control system behavior (e.g., `learning/config/`, `controller/config/`)

### Asynchronous Control Architecture
- **Dual-Thread Design**: Main thread (10-50Hz planning) + Background thread (800Hz control)
- **Smooth Control**: Decouples high-level planning from low-level control for smooth motion
- **Auto-Enable**: Set `auto_enable_async_control: true` in config
- **Documentation**: `docs/async_control_architecture.md`

## Development Commands

### Running Tests
```bash
# Run specific test modules
python test/test_end_to_end_integration.py
python test/test_act_integration.py
python test/test_cartesian_impedance_controller.py

# Test learning inference
python test/test_learning_inference_factory.py
```

### Common Operations
```bash
# Check hardware connection (example for FR3)
python hardware/fr3/test_connection.py

# Run ACT inference demo
python learning/demo/act_fr3_inference.py

# Test simulation environments
python simulation/mujoco/test_mujoco_sim.py
```

## Key Components and Locations

### Hardware Interfaces
- `hardware/fr3/` - Franka FR3 robot (uses panda-py submodule)
- `hardware/monte01/` - Monte01 custom robot
- `hardware/agibot_g1/` - Agibot G1 humanoid
- `hardware/unitreeG1/` - Unitree G1 robot

### Control Algorithms
- `controller/` - Impedance, admittance controllers
- `motion/` - IK solvers (Levenberg-Marquardt, DLS, Gaussian-Newton)
- `smoother/` - Trajectory smoothing (critical damping, adaptive, Ruckig)
- `trajectory/` - Trajectory planning and generation

### Learning/Inference
- `learning/models/` - Neural network models (ACT, DETR)
- `learning/demo/` - Inference demonstrations
- `learning/data/` - Data adapters for unified learning interface
- Supports temporal aggregation for ACT inference

### Simulation
- `simulation/mujoco/` - MuJoCo 3.3.0 simulation
- `simulation/fr3_isaaclab/` - Isaac Lab simulation

## Critical Implementation Notes

### Error Handling
- Avoid broad try-except blocks
- Always re-raise exceptions or log and re-raise for debugging
- Use assertions for parameter validation where appropriate

### Async Control Usage
When implementing robot control:
1. Use `set_joint_commands()` for target setting
2. Background thread handles smoothing and command sending at 800Hz
3. Check `auto_enable_async_control` flag in robot config

### Configuration Management
- Hardware configs: `hardware/{robot_name}/config/`
- Controller configs: `controller/config/`
- Learning configs: `learning/config/`
- Always validate YAML structure before runtime

### Testing New Hardware
1. Implement base class interfaces (ArmBase, ToolBase, etc.)
2. Create hardware config YAML
3. Add to RobotFactory registry
4. Write connection test in `hardware/{robot_name}/test_connection.py`

## Dependencies and SDKs

### Git Submodules (must be initialized)
- `dependencies/panda-py` - Franka SDK
- `dependencies/monte01_sdk` - Monte01 SDK
- `dependencies/unitree_sdk2*` - Unitree SDKs

### Key Python Dependencies
- **Core**: Python ≥3.10, NumPy 1.26.4, PyYAML 6.0.2
- **Robotics**: Pinocchio (dynamics), Pin 3.7.0 (kinematics), Ruckig 0.14.0
- **ML/DL**: PyTorch (install per CUDA version), JAX 0.6.2
- **Vision**: OpenCV 4.10.0, RealSense 2.56.4
- **Simulation**: MuJoCo 3.3.0

## Performance Considerations

- **Control Frequency**: Async mode achieves stable 800Hz control
- **Smoothing**: Critical damping smoother reduces jerk by ~70%
- **IK Solving**: Levenberg-Marquardt solver optimal for most cases
- **Multi-camera**: Use threading for parallel camera capture