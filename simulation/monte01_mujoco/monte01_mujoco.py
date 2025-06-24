import time
import mujoco
import mujoco.viewer
from threading import Thread, Lock
import numpy as np
import xml.etree.ElementTree as ET
import glog as log

Kp = np.array([1,1,1,1,1,
    # 1800, 1700, 1600, 1500, 1400, 1300, 1200, # left
    1800, 0,0,0,0,0,0, # left
    800, 700, 600, 500, 400, 300, 200, # right
    ])
Kd = Kp * 0.5

joints_fixed = [
    "body_joint_1",
    "body_joint_2",
    "body_joint_3",

    "head_joint_1",
    "head_joint_2",
]

joints_all = [
    "body_joint_1",
    "body_joint_2",
    "body_joint_3",

    "head_joint_1",
    "head_joint_2",

    "left_arm_joint_1", 
    "left_arm_joint_2", 
    "left_arm_joint_3", 
    "left_arm_joint_4", 
    "left_arm_joint_5", 
    "left_arm_joint_6", 
    "left_arm_joint_7",

    "right_arm_joint_1", 
    "right_arm_joint_2", 
    "right_arm_joint_3",
    "right_arm_joint_4",
    "right_arm_joint_5",
    "right_arm_joint_6",
    "right_arm_joint_7",
]
joint_ids_all = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]

class Monte01Mujoco:
    def __init__(self, xml_path="assets/monte_01/urdf/scene_monte01.xml", simulate_dt=0.001, viewer_dt=0.02):
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

        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)

        mujoco.mj_forward(self.mj_model, self.mj_data)

        self.jp_prev = np.zeros_like(self.mj_data.qpos)
        self.jp_prev[:] = self.mj_data.qpos[:]

        self.b_set_jp = False # fixed drop in first a few frames.

    def get_joint_positions(self, joint_names) -> np.ndarray:
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
        ids = np.array(ids) - 1
        self.mj_data.qpos[ids] = self.jp_prev[ids]
        self.mj_data.qvel[ids] = 0
        self.mj_data.ctrl[ids] = 0

    def set_joint_positions(self, id2positions):
        """
        Sets the target positions for the specified joints using a PD controller.
        This method assumes the lock is already held by the calling thread.
        """
        # TODO: Add hardware API call here
        with self.locker:
            ids = list(id2positions.keys())
            ids = np.array(ids) - 1

            mujoco.mj_inverse(self.mj_model, self.mj_data)

            # log.info(f"ids:{ids}")
            target_pos = list(id2positions.values())
            # log.info(f"target_pos:{target_pos}")
            fixed_joints = set(joint_ids_all) - set(ids)
            self.hold_joint_positions(tuple(fixed_joints))
            
            # x = np.array(tuple(fixed_joints))-1
            # self.mj_data.ctrl[x] = self.mj_data.qfrc_inverse[x]

            # gravity_comp = self.mj_data.qfrc_inverse[ids]
            # tau = Kp[ids] * (target_pos - self.mj_data.qpos[ids]) - Kd[ids] * self.mj_data.qvel[ids] + gravity_comp

            # # torque_range = self.mj_model.actuator_ctrlrange[ids]
            # # tau = np.clip(tau, torque_range[:, 0], torque_range[:, 1])
            # self.mj_data.ctrl[ids] = tau

            self.mj_data.qpos[ids] = target_pos #self.mj_data.qpos[ids] + (target_pos - self.mj_data.qpos[ids])*0.02
            self.mj_data.qvel[ids] = 0
            self.mj_data.ctrl[ids] = 0
            self.b_set_jp = True

            # log.info(f"target_pos == {target_pos}")
            
            self.jp_prev[ids]=target_pos

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
        time.sleep(0.2)

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
