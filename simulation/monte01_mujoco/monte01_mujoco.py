import time
import mujoco
import mujoco.viewer
from threading import Thread, Lock
import numpy as np
import glog as log
from simulation.mujoco_env_creator import MujocoEnvCreator
from motion.kinematics import PinocchioKinematicsModel as KinematicsModel

# Gripper joint names
LEFT_GRIPPER_JOINT = "left_drive_gear_joint"
RIGHT_GRIPPER_JOINT = "right_drive_gear_joint"

class Monte01Mujoco:
    def __init__(self, xml_path=None, config_path="simulation/scene_config/example.yaml", simulate_dt=0.001, viewer_dt=0.02):
        t0 = time.time()
        self.simulate_dt = simulate_dt
        self.viewer_dt = viewer_dt
        self.locker = Lock()

        if xml_path is not None:
            # Use static XML file (legacy mode)
            self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
            self.mj_data = mujoco.MjData(self.mj_model)
            log.info(f"Loaded static XML from: {xml_path}")
        else:
            # Use dynamic scene generation
            env_creator = MujocoEnvCreator(config_path=config_path)
            self.mj_model, self.mj_data = env_creator.create_model()
            log.info(f"Generated dynamic scene from config: {config_path}")
        self.viewer = None

        self.mj_model.opt.timestep = self.simulate_dt
        self.num_motor_ = self.mj_model.nu
        self.dim_motor_sensor_ = 3 * self.num_motor_
        self.last_print_time = -1

        joint_names = [self.mj_model.joint(i).name for i in range(self.mj_model.njnt)]
        self._joint_name_to_id = {name: i for i, name in enumerate(joint_names)}
        
        # Create mapping from joint IDs to actuator IDs
        # MuJoCo actuators may not correspond 1:1 with joints
        self._joint_id_to_actuator_id = {}
        for i in range(self.mj_model.nu):
            actuator_joint_id = self.mj_model.actuator_trnid[i, 0]  # Get joint ID for this actuator
            if actuator_joint_id >= 0:  # Valid joint
                self._joint_id_to_actuator_id[actuator_joint_id + 1] = i  # +1 because we use 1-based joint IDs
        
        log.info(f"MuJoCo model info: nq={self.mj_model.nq}, nu={self.mj_model.nu}, njnt={self.mj_model.njnt}")
        log.info(f"Joint ID to Actuator ID mapping: {self._joint_id_to_actuator_id}")

        # mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)

        mujoco.mj_forward(self.mj_model, self.mj_data)

        self.jp_prev = np.zeros_like(self.mj_data.qpos)
        self.jp_prev[:] = self.mj_data.qpos[:]

        self.b_set_jp = False # fixed drop in first a few frames.
        
        # Gripper state tracking
        self.gripper_target_position = 0.0  # Target gripper position

        t1 = time.time()
        log.info(f"Monte01Mujoco __init__耗时: {t1 - t0:.4f} 秒")

    def get_joint_positions(self, joint_names: list) -> np.ndarray:
        """
        Gets the positions of the specified joints using vectorized operations.
        This method assumes the lock is already held by the calling thread.
        """
        with self.locker:
            qpos_addrs = np.array([self.mj_model.joint(name).qposadr[0] for name in joint_names])
            positions = self.mj_data.qpos[qpos_addrs]
            log.debug(f"Joint names\t{joint_names}\nJoint positions:\t{positions}")
            return positions

    def hold_joint_positions(self, ids):
        # Convert joint IDs to 0-based indices for qpos/qvel
        qpos_indices = np.array(ids) - 1
        self.mj_data.qpos[qpos_indices] = self.jp_prev[qpos_indices]
        self.mj_data.qvel[qpos_indices] = 0
        
        # Set actuator controls for joints that have actuators
        for joint_id in ids:
            if joint_id in self._joint_id_to_actuator_id:
                actuator_id = self._joint_id_to_actuator_id[joint_id]
                self.mj_data.ctrl[actuator_id] = 0

    def set_joint_positions(self, id2positions):
        """
        Sets the target positions for the specified joints using a PD controller.
        This method assumes the lock is already held by the calling thread.
        """
        # TODO: Add hardware API call here
        with self.locker:
            # 假设 self._all_joint_ids 包含了所有你希望控制的关节
            # 2. 为需要移动的关节设置目标位置
            for joint_id, target_pos in id2positions.items():
                # 检查该关节是否有对应的 'position' 执行器
                if joint_id in self._joint_id_to_actuator_id:
                    actuator_id = self._joint_id_to_actuator_id[joint_id]
                    # 对于 position 类型的执行器，ctrl 值就是目标位置
                    self.mj_data.ctrl[actuator_id] = target_pos
                else:
                    # Only log warning once per joint to reduce noise
                    if not hasattr(self, '_warned_joints'):
                        self._warned_joints = set()
                    if joint_id not in self._warned_joints:
                        joint_name = self.mj_model.joint(joint_id - 1).name
                        log.warning(f"Joint {joint_id} ({joint_name}) has no position actuator, cannot set target.")
                        self._warned_joints.add(joint_id)
                    pass

                self.jp_prev[joint_id - 1] = target_pos # check

            # 3. 对于未被显式控制的关节，不进行任何干预以避免动力学耦合干扰  
            # 特别是身体和头部关节，应该保持自然的动力学状态
            # 只对明确被控制的关节设置目标位置
            controlled_joints = set(id2positions.keys())
            log.debug(f"Controlling joints: {controlled_joints}")
            
            # 不再对未控制的关节应用“保持”逻辑，让它们自然动作
            
            self.b_set_jp = True


    def _simulation_thread(self):
        """Main simulation loop."""
        print("Simulation thread started.")
        while self.viewer.is_running():
            try:
                step_start = time.perf_counter()

                with self.locker:
                    # if self.b_set_jp:
                    mujoco.mj_step(self.mj_model, self.mj_data)
                    # else:
                    #     time.sleep(0.1)
                    

                time_until_next_step = self.mj_model.opt.timestep - (time.perf_counter() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
            except Exception as e:
                print(f"Error in simulation thread: {e}")
                break

    def get_time(self):
        with self.locker:
            return self.mj_data.time
        
    def should_print(self):
        with self.locker:
            if int(self.mj_data.time) > self.last_print_time:
                self.last_print_time = int(self.mj_data.time)
                return True
            return False

    def start_viewer_only(self):
        """Starts only the viewer and simulation thread, but doesn't run the main viewer loop."""
        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        
        sim_thread = Thread(target=self._simulation_thread)
        sim_thread.start()
        return sim_thread

    def start(self):
        """Starts the simulation and viewer with main loop."""
        sim_thread = self.start_viewer_only()

        # Main rendering loop
        print("Viewer (main) thread started.")
        while self.viewer.is_running():
            with self.locker:
                self.viewer.sync()
            time.sleep(self.viewer_dt)

        sim_thread.join()

    def set_end_effector_pose(self, body_name: str, target_pose: np.ndarray, joint_ids: list, arm_kinematics:KinematicsModel=None) -> bool:
        """
        Set end-effector pose using inverse kinematics.
        
        Args:
            body_name: Name of the end-effector body (e.g., 'left_arm_link_7')
            target_pose: Target pose as 4x4 transformation matrix in chest frame
            joint_ids: List of joint IDs to control
            arm_kinematics: Optional kinematics model from the arm component
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            log.debug(f"set_end_effector_pose called for {body_name} with joint_ids: {joint_ids}")
            
            # If we have a kinematics model, use IK to solve for joint positions
            if arm_kinematics is not None:
                # Get current joint positions as seed for IK
                current_qpos_indices = np.array(joint_ids) - 1  # Convert to 0-based indices
                current_joint_positions = self.mj_data.qpos[current_qpos_indices]
                
                # Solve IK
                success, joint_positions = arm_kinematics.ik(target_pose, seed=current_joint_positions)
                
                # Only set joint positions if IK converged successfully
                if success and joint_positions is not None:
                    # Create joint position mapping
                    id2positions = {}
                    for i, joint_id in enumerate(joint_ids):
                        id2positions[joint_id] = joint_positions[i]
                    
                    # Set the computed joint positions
                    self.set_joint_positions(id2positions)
                    log.debug(f"IK converged and solution applied for {body_name}")
                    return True
                else:
                    log.warning(f"IK failed to converge for {body_name}, keeping current joint positions")
                    return False
            else:
                # Simple mode: just return True to indicate method exists
                # The arm component should handle IK externally and call set_joint_positions
                log.debug(f"No kinematics model provided, assuming external IK handling")
                return True
            
        except Exception as e:
            log.error(f"Error in set_end_effector_pose: {e}")
            return False


if __name__ == "__main__":
    sim = Monte01Mujoco()
    sim.start()
