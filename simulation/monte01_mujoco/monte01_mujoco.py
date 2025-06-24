import time
import mujoco
import mujoco.viewer
from threading import Thread, Lock
import numpy as np
import glog as log
import time

# Kp = np.array([1,1,1,1,1,
#     # 1800, 1700, 1600, 1500, 1400, 1300, 1200, # left
#     1800, 0,0,0,0,0,0, # left
#     800, 700, 600, 500, 400, 300, 200, # right
#     ])
# Kd = Kp * 0.5

# joints_fixed = [
#     "body_joint_1",
#     "body_joint_2",
#     "body_joint_3",

#     "head_joint_1",
#     "head_joint_2",
# ]

# joints_all = [
#     "body_joint_1",
#     "body_joint_2",
#     "body_joint_3",

#     "head_joint_1",
#     "head_joint_2",

#     "left_arm_joint_1", 
#     "left_arm_joint_2", 
#     "left_arm_joint_3", 
#     "left_arm_joint_4", 
#     "left_arm_joint_5", 
#     "left_arm_joint_6", 
#     "left_arm_joint_7",

#     "right_arm_joint_1", 
#     "right_arm_joint_2", 
#     "right_arm_joint_3",
#     "right_arm_joint_4",
#     "right_arm_joint_5",
#     "right_arm_joint_6",
#     "right_arm_joint_7",
# ]
joint_ids_all = [1,2,3,4,5,                              # Body + Head (5)
                 6,7,8,9,10,11,12,                          # Left Arm (7)
                 13,14,15,16,17,18,19,                      # Left Gripper (7) 
                 20,21,22,23,24,25,26,                      # Right Arm (7)
                 27,28,29,30,31,32,33]                      # Right Gripper (7)

# Gripper joint names
LEFT_GRIPPER_JOINT = "left_drive_gear_joint"
RIGHT_GRIPPER_JOINT = "right_drive_gear_joint"

class Monte01Mujoco:
    def __init__(self, xml_path="assets/monte_01/urdf/scene_monte01.xml", simulate_dt=0.001, viewer_dt=0.02):
        t0 = time.time()
        self.simulate_dt = simulate_dt
        self.viewer_dt = viewer_dt
        self.locker = Lock()

        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
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

        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)

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
            log.info(f"Joint names\t{joint_names}\nJoint positions:\t{positions}")
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
                    log.warning(f"Joint {joint_id} has no position actuator, cannot set target.")
                    pass

                self.jp_prev[joint_id - 1] = target_pos # check

            # 3. 为其他关节设置保持当前位置的控制目标
            # 这可以防止它们因动力学耦合而漂移
            holding_joints = set({1,2,3,4,5})
            for joint_id in holding_joints:
                if joint_id in self._joint_id_to_actuator_id:
                    actuator_id = self._joint_id_to_actuator_id[joint_id]
                    # 目标是当前位置，即保持不动 (joint_id is 1-based, qpos is 0-based)
                    self.mj_data.ctrl[actuator_id] = self.mj_data.qpos[joint_id - 1]
            
            self.b_set_jp = True


    def _simulation_thread(self):
        """Main simulation loop."""
        print("Simulation thread started.")
        while self.viewer.is_running():
            try:
                step_start = time.perf_counter()

                with self.locker:
                    if self.b_set_jp:
                        mujoco.mj_step(self.mj_model, self.mj_data)
                    else:
                        time.sleep(0.1)
                    

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

    def start(self):
        """Starts the simulation and viewer."""
        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)

        sim_thread = Thread(target=self._simulation_thread)
        sim_thread.start()

        # Main rendering loop
        print("Viewer (main) thread started.")
        while self.viewer.is_running():
            with self.locker:
                self.viewer.sync()
            time.sleep(self.viewer_dt)

        sim_thread.join()


if __name__ == "__main__":
    sim = Monte01Mujoco()
    sim.start()
