# Kinematics Module Usage Guide

This guide demonstrates how to use the kinematics module for robot inverse and forward kinematics calculations.

## Quick Start

### Initialize Kinematics Model

```python
from motion.kinematics import PinocchioKinematicsModel

# Initialize kinematics model
model = PinocchioKinematicsModel(
    urdf_path="assets/franka_fr3/fr3.urdf",
    base_link="fr3_link0",
    end_effector_link="fr3_hand"
)

print(f"Model initialized with {model.n_joints} joints")
print(f"Joint limits: {model.joint_lower_limit} to {model.joint_upper_limit}")
```

## Forward Kinematics

Calculate end-effector pose from joint angles:

```python
import numpy as np

# Define joint configuration
joint_angles = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

# Compute forward kinematics
pose = model.fk(joint_angles)

print("End-effector pose (4x4 transformation matrix):")
print(pose)
print(f"Position: {pose[:3, 3]}")
print(f"Rotation matrix:\n{pose[:3, :3]}")
```

## Inverse Kinematics

### Standard IK (Gauss-Newton Method)

```python
# Define target pose
target_pose = np.eye(4)
target_pose[:3, 3] = [0.5, 0.2, 0.4]  # Target position

# Optional: provide initial seed
seed = np.zeros(model.n_joints)

# Solve inverse kinematics
converged, solution = model.ik(
    target_pose=target_pose,
    seed=seed,
    max_iter=5000,
    tol=1e-6,
    step_size=1.0
)

if converged:
    print(f"IK converged! Solution: {solution}")
    # Verify solution
    achieved_pose = model.fk(solution)
    error = np.linalg.norm(achieved_pose[:3, 3] - target_pose[:3, 3])
    print(f"Position error: {error:.6f} m")
else:
    print("IK failed to converge")
```

### PyRoki IK (Alternative Solver)

```python
# Solve using PyRoki library (if available)
converged, solution = model.ik_pyroki(
    target_pose=target_pose,
    tol=1e-6
)

if converged:
    print(f"PyRoki IK converged! Solution: {solution}")
    # Verify solution
    achieved_pose = model.fk(solution)
    error = np.linalg.norm(achieved_pose[:3, 3] - target_pose[:3, 3])
    print(f"Position error: {error:.6f} m")
else:
    print("PyRoki IK failed to converge")
```

## Complete Example

```python
import numpy as np
from motion.kinematics import PinocchioKinematicsModel

def main():
    # Initialize model
    model = PinocchioKinematicsModel(
        urdf_path="assets/franka_fr3/fr3.urdf",
        base_link="fr3_link0",
        end_effector_link="fr3_hand"
    )
    
    # Generate a random reachable target pose using FK
    random_joints = np.random.uniform(
        model.joint_lower_limit,
        model.joint_upper_limit
    )
    target_pose = model.fk(random_joints)
    print(f"Target pose:\n{target_pose}")
    
    # Test both IK methods
    methods = ['ik', 'ik_pyroki']
    
    for method_name in methods:
        if method_name == 'ik_pyroki' and not hasattr(model, '_pyroki_robot'):
            print(f"Skipping {method_name} (PyRoki not available)")
            continue
            
        print(f"\n--- Testing {method_name} ---")
        method = getattr(model, method_name)
        
        import time
        start_time = time.time()
        converged, solution = method(target_pose)
        solve_time = time.time() - start_time
        
        if converged:
            achieved_pose = model.fk(solution)
            position_error = np.linalg.norm(achieved_pose[:3, 3] - target_pose[:3, 3])
            print(f"✓ Converged in {solve_time:.4f}s")
            print(f"  Solution: {solution}")
            print(f"  Position error: {position_error:.6f} m")
        else:
            print(f"✗ Failed to converge (time: {solve_time:.4f}s)")

if __name__ == "__main__":
    main()
```

## Advanced Usage

### Custom Joint Limits

```python
# Define custom joint limits
lower_limits = np.array([-2.8, -1.7, -2.8, -3.0, -2.8, -0.0, -2.8])
upper_limits = np.array([2.8, 1.7, 2.8, -0.1, 2.8, 3.7, 2.8])

converged, solution = model.ik(
    target_pose=target_pose,
    joint_limits=(lower_limits, upper_limits)
)
```

### Multiple Target Testing

```python
def test_multiple_targets(model, num_tests=100):
    """Test IK solver on multiple random targets."""
    success_count = 0
    total_time = 0
    
    for i in range(num_tests):
        # Generate random reachable pose
        random_joints = np.random.uniform(
            model.joint_lower_limit,
            model.joint_upper_limit
        )
        target_pose = model.fk(random_joints)
        
        # Solve IK
        start_time = time.time()
        converged, solution = model.ik(target_pose)
        solve_time = time.time() - start_time
        
        total_time += solve_time
        if converged:
            success_count += 1
    
    success_rate = (success_count / num_tests) * 100
    avg_time = total_time / num_tests
    
    print(f"Success rate: {success_rate:.1f}% ({success_count}/{num_tests})")
    print(f"Average time: {avg_time:.4f}s")

# Run test
test_multiple_targets(model, num_tests=50)
```

## Performance Comparison

To compare performance between different IK methods, use the provided test script:

```bash
cd /home/hanyu/HIROL/HIROLRobotPlatform/motion/test_scripts
python test_ik_comparison.py --num-tests 100 --robot-config panda
```

## Troubleshooting

### PyRoki Not Available
If you see warnings about PyRoki not being available:
```bash
pip install pyroki pyroki_snippets
```

### URDF Loading Issues
Ensure the URDF path is relative to the project root:
```python
# Correct path
model = PinocchioKinematicsModel("assets/franka_fr3/fr3.urdf", ...)

# Incorrect (absolute path)
model = PinocchioKinematicsModel("/full/path/to/urdf", ...)
```

### IK Convergence Issues
- Try different initial seeds
- Adjust tolerance and maximum iterations
- Ensure target pose is within robot workspace
- Use reachable poses generated by forward kinematics for testing

## API Reference

### PinocchioKinematicsModel

#### Methods

- `fk(joint_positions)` - Forward kinematics
- `ik(target_pose, seed=None, joint_limits=None, max_iter=5000, tol=1e-6, step_size=1.0)` - Standard IK
- `ik_pyroki(target_pose, seed=None, tol=1e-6)` - PyRoki-based IK

#### Properties

- `n_joints` - Number of joints
- `joint_lower_limit` - Lower joint limits
- `joint_upper_limit` - Upper joint limits
- `ee_frame_name` - End-effector frame name