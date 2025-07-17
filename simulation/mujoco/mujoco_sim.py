import mujoco
import mujoco.viewer
import numpy as np
import threading
import time
from hardware.base.utils import RobotJointState, transform_pose, negate_pose
import copy
from simulation.base.sim_base import SimBase
import warnings
import os
from simulation.mujoco.mujoco_env_creator import MujocoEnvCreator
from xml.etree import ElementTree as ET

class MujocoSim(SimBase):
    def __init__(self, config):
        super().__init__(config)
        # self._use_hardware = config['use_hardware']
        self._quat_sequence = config.get('quat_sequence', 'xyzw')
        self._dt = config['dt']
        self._model = None
        self._data = None
        self._joint_names = config['joint_names']
        self._actuator_names = config["actuator_names"]
        self._dof = config.get('dof', [len(self._actuator_names)])
        self._actuator_mode = config['actuator_mode']
        
        # Body joint configuration for hardware-simulation sync
        self._body_joint_names = config.get('body_joint_names', [])
        self._body_actuator_names = config.get('body_actuator_names', [])
        self._body_actuator_mode = config.get('body_actuator_mode', [])
        self.end_effector_site_name = config.get('ee_site_name', None)
        self.base_body_name = config["base_body"]
        #  key: sensor name, value: sensor data dim
        self._sensor_dict = config.get('sensor_dict', None)
        self._cam_renderer = []
        self._extra_render = config["extra_render"]
        self.use_custom_key_frame = config.get("use_custom_key_frame", False)
        
        # sim state feedback
        self._ee_site_pose = None  # [x, y, z, qx, qy, qz, qw]
        self._sensor_data = dict()
        
        # parse model
        self.parse_config()
        self.viewer = None
        
        # start mujoco thread
        self.lock = threading.Lock()
        thread = threading.Thread(target=self.sim_thread)
        thread.start()
        time.sleep(1)
    
    def sim_thread(self):
        if self._model is None or self._data is None:
            raise RuntimeError("Mujoco model and data are not initialized.")
        
        # main simulation loop
        print("Starting Mujoco simulation main loop...")
        self._model.opt.timestep = self._dt
        with mujoco.viewer.launch_passive(self._model, self._data, show_right_ui=True) as viewer:
            # Enable site frame visualization.
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
            viewer.user_scn.ngeom = 0
            viewer.sync()
            if self.viewer is None:
                self.viewer = viewer
                for i in range(self._traj_max_len):
                    self._add_geometry(self.viewer, [0,0,0], i)
            viewer.user_scn.ngeom = self._traj_max_len
            viewer.sync()
            while viewer.is_running():
                step_start = time.time()
                mujoco.mj_step(self._model, self._data)
                if self._extra_render:
                    self.render()
                viewer.sync()                
               
                # print('update simulation state!!!')
                self.lock.acquire()
                self.update_simulation_states()
                self.lock.release()

                used_time = time.time() - step_start
                time_until_next_step = self._dt - used_time
                if time_until_next_step > 0:
                    time.sleep(0.2*time_until_next_step)
                elif time_until_next_step > 1.2 * self._dt:
                    warnings.warn(f"Mujoco node frequency is not enough, "
                                  f"actual: {used_time}, expected: {self._dt}")

    def update_simulation_states(self):
        """Update the joint states from the Mujoco simulation."""
        if self._model is None or self._data is None:
            raise RuntimeError("Mujoco model and data are not initialized.")
        
        # joint state update
        for i, joint_name in enumerate(self._joint_names):
            joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f"Joint '{joint_name}' not found in the Mujoco model,"
                                 "please check your mujoco config & xml file.")
            qpos_adr = self._model.jnt_qposadr[joint_id]
            self._joint_states._positions[i] = self._data.qpos[qpos_adr]
            qvel_adr = self._model.jnt_dofadr[joint_id]
            self._joint_states._velocities[i] = self._data.qvel[qvel_adr]
            self._joint_states._accelerations[i] = self._data.qacc[qvel_adr]
            actuator_id = self._model.actuator(self._actuator_names[i]).id
            self._joint_states._torques[i] = self._data.qfrc_actuator[actuator_id]
        # print(f'joint position: {self._joint_states._positions}')
        # ee pose update
        if self.end_effector_site_name is not None:
            ee_site_pose = self.get_site_pose(self.end_effector_site_name, self._quat_sequence)
            base_pose = self.get_body_pose(self.base_body_name, self._quat_sequence)
            base2world_pose = negate_pose(base_pose)
            # base 2 ee pose
            ee_site_pose = transform_pose(base2world_pose, ee_site_pose)
            if not ee_site_pose is None:
                self._ee_site_pose = ee_site_pose

    def get_joint_states(self) -> RobotJointState:
        self.lock.acquire()
        cur_joint_states = copy.deepcopy(self._joint_states)
        self.lock.release()
        return cur_joint_states
    
    def get_tcp_pose(self) -> np.ndarray | None:
        if self._ee_site_pose is None:
            return None
        
        self.lock.acquire()
        cur_tcp_pose = copy.deepcopy(self._ee_site_pose)
        self.lock.release()
        return cur_tcp_pose

    def set_joint_command(self, mode: list[str], actuator_action:np.ndarray):
        # assertion check
        if len(actuator_action) != len(mode):
            raise ValueError(f'the command for the action has different dimension with the mode, '
                             f'action dim: {len(actuator_action)}, mode dim: {len(mode)}')
        
        if len(actuator_action) != len(self._actuator_mode):
            raise ValueError(f"Action length {len(actuator_action)} does not match the number of actuators {len(self._actuator_mode)}.")    
        
        for i, target in enumerate(actuator_action):
            if mode[i] != self._actuator_mode[i]:
                warnings.warn(f"The mode for {i} th actuator differs from the command!, "
                              f"expected: {self._actuator_mode[i]}, actual:{mode[i]}")
                break
            
            joint_name = self._joint_names[i]
            joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f"Joint '{joint_name}' not found in the Mujoco model.")
            
            # command execution
            if mode[i] == 'position':
                qpos_adr = self._model.jnt_qposadr[joint_id]
                self._data.qpos[qpos_adr] = target
            elif mode[i] == "velocity":
                qvel_adr = self._model.jnt_dofadr[joint_id]
                self._data.qvel[qvel_adr] = target
            elif mode[i] == "torque":
                actuator_id = self._model.actuator(self._actuator_names[i]).id
                self._data.ctrl[actuator_id] = target
            else:
                raise ValueError(f"Unsupported mode for {i}th actuator '{mode[i]}'. Supported modes are 'position', 'velocity', and 'torque'.")
    
    def set_body_joint_command(self, body_positions: np.ndarray) -> bool:
        """
        Set body joint positions in simulation
        
        Args:
            body_positions: 5D array [body_joint_1, body_joint_2, body_joint_3, head_joint_1, head_joint_2]
            
        Returns:
            bool: Success status
        """
        if len(self._body_joint_names) == 0:
            warnings.warn("[BodySync] No body joints configured in simulation")
            return False
            
        if len(body_positions) != len(self._body_joint_names):
            warnings.warn(f"[BodySync] Body position count mismatch: expected {len(self._body_joint_names)}, got {len(body_positions)}")
            return False
            
        try:
            for i, (joint_name, position) in enumerate(zip(self._body_joint_names, body_positions)):
                joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id < 0:
                    warnings.warn(f"[BodySync] Body joint '{joint_name}' not found in Mujoco model")
                    continue
                    
                qpos_adr = self._model.jnt_qposadr[joint_id]
                self._data.qpos[qpos_adr] = position
                
            return True
            
        except Exception as e:
            warnings.warn(f"[BodySync] Failed to set body joint positions: {e}")
            return False
    
    def get_site_pose(self, site_name, quat_seq):
        """
            Get the pose in [x,y,z,qx.qy,qz,qw] of the site
            @params:
                site_name
                quat_seq: sequence of the quat in pose, ['xyzw', 'wxyz']
        """
        if self._model is None:
            warnings.warn("The model for the mujoco simulation is not correctly configured")
            return None
        
        site_id = self._model.site(site_name).id
        if site_id < 0:
            warnings.warn("The specific site could not be found from the mujoco model!!!")
            return None
        
        pose = np.zeros(7)
        pose[:3] = self._data.site_xpos[site_id].copy()
        pose_mat = self._data.site_xmat[site_id].copy()
        # print(f'pose: {pose[:3]}, mat: {pose_mat}')
        mujoco.mju_mat2Quat(pose[3:], pose_mat)
        
        if quat_seq == 'xyzw':
            # Convert mujoco wxyz to xyzw
            pose[3:] = [pose[4], pose[5], pose[6], pose[3]]
        return pose
    
    def get_body_pose(self, body_name, quat_seq = "xyzw"):
        """
            Get the pose in [x,y,z,qx.qy,qz,qw] of the body
            @params:
                body_name
                quat_seq: sequence of the quat in pose, ['xyzw', 'wxyz']
        """
        if self._model is None:
            warnings.warn("The model for the mujoco simulation is not correctly configured")
            return None

        body_id = self._model.body(body_name).id
        if body_id < 0:
            warnings.warn(f"The specific body {body_name} could not be found from the mujoco model!!!")
            return None
        
        pose = np.zeros(7)
        pose[:3] = self._data.body(body_id).xpos
        pose[3:] = self._data.body(body_id).xquat
        if quat_seq == "xyzw":
            pose[3:] = [pose[4], pose[5], pose[6], pose[3]]
        return pose
        
    def add_camera(self, camera_name, resolution):
        render = mujoco.Renderer(self._model, height=resolution[0], width=resolution[1])
        camera_render_dict = {'name': camera_name, 'render': render}
        self._cam_renderer.append(camera_render_dict)
        
    def get_camera_img(self, camera_name) -> None | np.ndarray:
        img = None
        for camera_render_dict in self._cam_renderer:
            if camera_name == camera_render_dict['name']:
                render = camera_render_dict['render']
                img = render.render()
                return img
             
    def parse_relative_path(self, relative_path):
        cur_path = os.path.dirname(os.path.abspath(__file__))
        abs_path = os.path.join(cur_path, '../..', relative_path)
        return abs_path
        
    def parse_config(self) -> bool:
        """parse config to get the Mujoco model and data."""
        # @TODO: Require modification to achieve dynamic loading 
        model_path = self.parse_relative_path(self._config['base_xml_file'])
        print(f'model path: {model_path}')
        self._model = mujoco.MjModel.from_xml_path(model_path)
        self._data = mujoco.MjData(self._model)
        
        # set init actuator position
        if self.use_custom_key_frame:
            robot_xml_path = self.parse_relative_path(self._config['robot_xml'])
            self._init_pose = self.extract_custom_params(robot_xml_path, 'init_pos')
            self.set_joint_position(self._init_pose)
        
        # model creation based on mujoco env config
        # env_cfg = self._config["env_config"]
        # env_template = self._config["env_template"]
        # mujoco_env_creator = MujocoEnvCreator(env_cfg, env_template)
        # self._model, self._data = mujoco_env_creator.create_model()
        
        # init robot state data 
        nv = len(self._joint_names)
        self._joint_states._positions  = np.zeros(nv)
        self._joint_states._velocities  = np.zeros(nv)
        self._joint_states._accelerations  = np.zeros(nv)
        self._joint_states._torques  = np.zeros(nv)
        
        # sensor
        if self._sensor_dict is not None:
            for idx, (key, value) in enumerate(self._sensor_dict.items()):
                # cameras
                if 'cam' in key:
                    self.add_camera(camera_name=key, resolution=value['resolution'])
        # dynamic object spawn
        
        return True

    def set_joint_position(self, values):
        if len(values) != len(self._joint_names):
            warnings.warn("The target position dim does not match with defined joint names")
            
        for i, target in enumerate(values):
            joint_name = self._joint_names[i]
            joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f"Joint '{joint_name}' not found in the Mujoco model.")
            
            qpos_adr = self._model.jnt_qposadr[joint_id]
            self._data.qpos[qpos_adr] = target

    def extract_custom_params(self, xml_path, param_name):
        """
            @brief: extract the user-defined parameters from xml
            :@params: 
                xml_path: the specified xml file 
                param_name: the name of the user defined param
            :@return: parameter values
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # find 'custom' tag
        custom_elem = root.find('custom')
        if custom_elem is None:
            raise ValueError("Could not find custom tag in the specified xml path")
        
        # obtain specific param data
        data = []
        name = None
        for numeric in custom_elem.findall('numeric'):
            name = numeric.get('name')
            if name == param_name:
                data = [float(x) for x in numeric.get('data').split()]
                break
        
        if not data:
            raise ValueError(f"Could not find data element of the {param_name} with {name}")
        else:
            print(f"data: {data} for {param_name}'s {name}")
        return data
    
    def render(self):
        """Render the current state of the simulation."""
        # visualize the trajectory
        if len(self._visulize_traj_data) == 0:
            return 
        
        traj_data = self._visulize_traj_data.popleft()[:3]
        self._update_geometry_position(self.viewer, traj_data, self._cur_traj_index)
        self._cur_traj_index += 1
        if self._cur_traj_index == self._traj_max_len:
            self._cur_traj_index = 0
        
    def _add_geometry(self, viewer, position, index):
        mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[index],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.016, 0, 0],
                        pos=position,
                        mat=np.eye(3).flatten(),
                        rgba=np.array([1, 0, 0, 1])
                        )
    
    def _update_geometry_position(self, viewer, position, index: int):
        viewer.user_scn.geoms[index].pos = position
        
    def set_target_mocap_rotation(self, mocap_name, quat, quat_seq="xyzw"):
        mocap_id = self._model.body(mocap_name).mocapid[0]
        if quat_seq == "xyzw":
            quat = [quat[3], quat[0], quat[1], quat[2]]
        self._data.mocap_quat[mocap_id] = quat
        
    def set_target_mocap_position(self, mocap_name, position):
        mocap_id = self._model.body(mocap_name).mocapid[0]
        self._data.mocap_pos[mocap_id] = position
        
    def set_target_mocap_pose(self, mocap_name, pose, quat_seq="xyzw"):
        """
            @brief: visualize the target pose
            @ params:
                mocap_name: the mocap site name to visualize the pose
                pose: 7D pose with format [x,y,z,qx,qy,qz,qw]
        """
        self.set_target_mocap_rotation(mocap_name, pose[3:], quat_seq)
        self.set_target_mocap_position(mocap_name, pose[:3])
        
    def get_dof(self):
        return self._dof
    
if __name__ == '__main__':
    import yaml
    import os
    config = None
    cur_path = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(cur_path, '../config', 'mujoco_fr3_cfg.yaml')
    print(f'cfg file name: {cfg_file}')
    with open(cfg_file, 'r') as stream:
        config = yaml.safe_load(stream)
    print(f'yaml data: {config}')
    print(config)
    sensor_dict = config['mujoco']['sensor_dict']
    print(f'sensor dict: {sensor_dict}')
    for idx, (key, value) in enumerate(sensor_dict.items()):
        print(f'id: {idx}, key: {key}, value: {value}')
    
    mujoco_fr3 = MujocoSim(config["mujoco"])
    # test sesnor image
    import cv2
    
    counter = 0
    while True:
        key = cv2.waitKey(1)
        if key == 'q':
            break
        
        # read img
        camera_name = 'ee_cam'
        img = mujoco_fr3.get_camera_img(camera_name)
        if img is None:
            print(f'did not get the image from the {camera_name}')
        cv2_img = img[:,:,::-1]
        cv2.imshow("example_img", cv2_img)
        
        # add traj data
        if counter %2 == 0:
            data = [0.5,0,1.0]
        else:
            data = [1,0,1.2]
        mujoco_fr3.update_trajectory_data(data)
        counter += 1
        