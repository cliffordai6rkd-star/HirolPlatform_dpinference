from factory.components.robot_factory import RobotFactory
from factory.components.motion_factory import MotionFactory
from hardware.base.utils import transform_pose, pose_diff
import glog as log
import abc, time
import gymnasium as gym
from dataset.utils import ActionType, Action_Type_Mapping_Dict, ObservationType
import numpy as np
from factory.components.motion_factory import Robot_Space
from scipy.spatial.transform import Rotation as R

class GymApi(gym.Env):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self._max_step_nums = config["max_step_nums"]
        
        self._action_type = config.get("action_type", "joint_position")
        self._action_type = Action_Type_Mapping_Dict[self._action_type]
        self._action_ori_type = config.get("action_orientation_type", "euler")
        self._delta_action_target = {}
        self._obs_type = config.get("observation_type", "jonit_position")
        self._obs_type = ObservationType(self._obs_type)
        self._contain_ft = config.get('contain_ft', False)
        self._last_ee = {}
        self._use_relative_pose = config.get("use_relative_pose", False)
        self._init_pose = {}
        robot_motion_cfg = config["motion_config"]
        self._tool_position_dof = config.get("tool_position_dof", 1)
        self._tool_state_max = config.get("tool_state_max", 1)
        
        self._reset_space = config.get("reset_space", "joint")
        self._reset_space = Robot_Space.JOINT_SPACE if self._reset_space== 'joint' else Robot_Space.CARTESIAN_SPACE
        self._reset_arm_command = config.get("reset_arm_command", None)
        self._reset_tool_command = config.get("reset_tool_command", [[1]])
        self._is_debug = config.get("is_debug", False)
        self._use_hardware = config.get("enable_hardware", True)
        
        self._robot_system = RobotFactory(robot_motion_cfg)
        self._robot_motion = MotionFactory(robot_motion_cfg, self._robot_system)
        self._robot_motion.create_motion_components()
        log.info(f"The robot motion component is successfully created in gym api! {self._obs_type} : {self._action_type}")
        
        # variable used for gym api
        self._step_counter = 0
        self.reset()
        
    def set_init_pose(self):
        time.sleep(1.0)
        ee_states = self.get_ee_state()
        for key, cur_ee_state in ee_states.items():
            self._delta_action_target[key] = cur_ee_state["pose"]
            self._last_ee[key] = cur_ee_state["pose"]
            log.info(f'Updated delta action, {self._delta_action_target[key]} for {key}!!!')
            if self._use_relative_pose:
                self._init_pose[key] = cur_ee_state["pose"]
                self._last_ee[key] = pose_diff(self._last_ee[key], self._init_pose[key])
                log.info(f'Updated init pose!!!')
                
    def set_action_type(self, action_type: ActionType):
        self._action_type = action_type
        
    def step(self, action):
        # action execution
        arm_action = action['arm']
        execute_arm_action = np.array([]); execute_tool_action = []
        # log.info(f'action type: {self._action_type} ori type: {self._action_ori_type}')
        # @TODO: hack
        gripper_position_dof = self._tool_position_dof
        if self._action_type in [ActionType.JOINT_POSITION, ActionType.JOINT_POSITION_DELTA]:
            dofs = self._robot_motion.get_model_dof_list()[1:]
            cur_joint_position = self.get_joint_state()
            action_index = 0
            for j, (key, jps) in enumerate(cur_joint_position.items()):
                jps = jps["position"]
                index_l = gripper_position_dof*j + action_index
                index_r = gripper_position_dof*j + dofs[j] + action_index
                action_index = index_r
                cur_arm_action = arm_action[index_l:index_r]
                if self._action_type == ActionType.JOINT_POSITION_DELTA:
                    cur_arm_action += jps
                execute_arm_action = np.hstack((execute_arm_action, cur_arm_action))
            self.set_joint_position(execute_arm_action)
        elif self._action_type in [ActionType.END_EFFECTOR_POSE, ActionType.END_EFFECTOR_POSE_DELTA]:
            cur_ee_pose = self.get_ee_state()
            action_index = 0
            for j, (key, pose) in enumerate(list(cur_ee_pose.items())):
                pose = pose["pose"]
                if self._action_ori_type == "euler":
                    index_l = 6 * j + action_index
                    index_r = 6 * (j+1) + action_index
                else: 
                    index_l = 7 * j + action_index
                    index_r = 7 * (j+1) + action_index
                action_index = index_r
                cur_arm_action = arm_action[index_l:index_r]
                if self._action_ori_type == "euler":
                    cur_arm_action = np.hstack((cur_arm_action, [0]))
                    cur_arm_action[3:] = R.from_euler("xyz", cur_arm_action[3:6]).as_quat()
                if self._action_type == ActionType.END_EFFECTOR_POSE_DELTA:
                    # try to use the fixed reset pose
                   cur_arm_action = transform_pose(self._delta_action_target[key], cur_arm_action, True)
                   self._delta_action_target[key] = cur_arm_action
                    # @TODO: how to deal with delta pose with umi
                        
                # for umi absolute relative pose
                elif self._use_relative_pose:
                    # for relative pose action representation
                    cur_arm_action = transform_pose(self._init_pose[key], cur_arm_action)
                execute_arm_action = np.hstack((execute_arm_action, cur_arm_action))
            self.set_ee_pose(execute_arm_action)
        else:
            raise ValueError(f"Unsupported action type: {self._action_type}")
        
        # tool execution
        tool_action = np.array(action['tool'])
        tool_type_dict = self._robot_system.get_tool_dict_state()
        tool_index = 0
        if tool_type_dict is not None:
            for j in range(len(tool_type_dict)):
                index_r = tool_index + gripper_position_dof
                if tool_action.ndim == 0:
                    execute_tool_action.append(np.array(tool_action))
                else:
                    execute_tool_action.append(np.array(tool_action[tool_index:index_r]))
                tool_index = index_r
            # log.info(f'tool action: {tool_action}')
            self._robot_motion.set_tool_command(np.array(tool_action))
        
        # obs
        start = time.perf_counter()
        observation = self.get_observation()
        # observation = {}
        obs_time = time.perf_counter() - start
        reward, done = self.compute_rewards()
        done = done or (self._step_counter >= self._max_step_nums)
        info = self.get_info()
        info['obs_time'] = obs_time
        
        return observation, reward, done, False, info
        
    def reset(self, *, seed = None, options = None):
        if self._use_hardware:
            self._robot_motion.update_execute_hardware(True)
            self._robot_system._enable_hardware = True
            time.sleep(0.3)
            log.info("The robot hardware all enabled!!!!!!")
        self._robot_motion.reset_robot_system(arm_command=self._reset_arm_command,
                                              space=self._reset_space,
                                              tool_command=self._reset_tool_command)
        self.set_init_pose()
    
    def get_joint_state(self):
        current_joint_state = self._robot_system.get_joint_states()
        end_effector_names = self._robot_motion.get_model_end_effector_link_list()
        ee_index = ['left', 'right'] if len(end_effector_names) > 1 else ['single']
        joint_states = {}
        for key in ee_index:
            joint_states[key] = {} 
            sliced_joint_states = self._robot_motion.get_type_joint_state(current_joint_state, key)
            joint_states[key]["position"] = sliced_joint_states._positions
            joint_states[key]["velocity"] = sliced_joint_states._velocities
            joint_states[key]["acceleration"] = sliced_joint_states._accelerations
        return joint_states
    
    def get_tool_state(self):
        tools_dict = self._robot_system.get_tool_dict_state()
        if tools_dict is None:
            return None
        
        tool_state_dict = {}
        for key, tool_state in tools_dict.items():
            tool_state_dict[key] = dict(position=tool_state._position)
        return tool_state_dict
    
    def get_ee_state(self):
        """
            @brief: return dict [key: 7D pose]
        """
        end_effector_names = self._robot_motion.get_model_end_effector_link_list()
        ee_index = ['left', 'right'] if len(end_effector_names) > 1 else ['single']
        model_types = self._robot_motion.get_model_types()
        poses = {}
        for i, ee_name in enumerate(end_effector_names):
            cur_model_type = model_types[i] if len(model_types) > 1 else model_types[0]
            frame_pose = self._robot_motion.get_frame_pose(ee_name, 
                                        cur_model_type, need_vel=True)
            key = ee_index[i]
            poses[key] = dict(pose=frame_pose[:7], twist=frame_pose[7:13])
        return poses
    
    def get_relative_ee_pose(self):
        ee_states = self.get_ee_state()
        relative_pose = {}
        for key, cur_ee_state in ee_states.items():
            pose = cur_ee_state["pose"]
            relative_pose[key] = dict(pose=pose_diff(pose, self._init_pose[key]))
        return relative_pose

    def get_camera_infos(self):
        cameras_data = self._robot_system.get_cameras_infos()
        if cameras_data is None: return None
        
        cur_colors = {}
        cur_depths = {}
        cur_imus = {}
        for cam_data in cameras_data:
            name = cam_data['name']
            if 'color' in name:
                cur_colors[name] = cam_data['img']
            if 'depth' in name:
                cur_depths[name] = cam_data['img']
            if 'imu' in name:
                cur_imus[name] = cam_data['imu']
        cameras_data = {"color": cur_colors, "depth": cur_depths, "imu": cur_imus}
        return cameras_data
    
    def get_ft_infos(self):
        ft_data = self._robot_system.get_ft_data()
        if not ft_data: return ft_data
        
        key = ["single"] if len(ft_data) == 1 else ["left", "right"]
        ft_dict = {}
        for i, cur_key in enumerate(key):
            if cur_key == "single":
                ft_dict[cur_key] = ft_data[i]["data"]
            else:
                for j, cur_ft_data in enumerate(ft_data):
                    if cur_key in cur_ft_data[i]["name"]:
                        ft_dict[cur_key] = cur_ft_data["data"]
                        ft_data.pop(j) # increase the efficency
                        break
        return ft_dict
    
    @abc.abstractmethod
    def get_observation(self):
        obs_state = {}
        joint_states = self.get_joint_state()
        joint_check = [ObservationType.JOINT_POSITION_ONLY, ObservationType.JOINT_POSITION_END_EFFECTOR]
        if self._obs_type in joint_check:
            if len(joint_states) == 0:
                raise ValueError(f'Cur {self._obs_type} do not get joint states!!!')
        ee_states = self.get_ee_state() if not self._use_relative_pose else self.get_relative_ee_pose()
        ee_check = [ObservationType.DELTA_END_EFFECTOR_POSE, ObservationType.END_EFFECTOR_POSE, 
                    ObservationType.JOINT_POSITION_END_EFFECTOR]
        if self._obs_type in ee_check:
            if len(ee_states) == 0:
                raise ValueError(f'Cur {self._obs_type} do not get ee states!!!')
        visual_ee_poses = {}
        for key, pose in ee_states.items():
            visual_ee_poses[key] = pose["pose"]
        self._robot_motion.sim_visualize_tcp(visual_ee_poses)
        tools_dict = self.get_tool_state()
        ft_dict = self.get_ft_infos()
        # log.info(f'ft dict: {ft_dict}, obs type: {self._obs_type}, contain ft: {self._contain_ft}')
        for key, joint_state in joint_states.items():
            obs_state[key] = np.array([])
            if self._obs_type == ObservationType.JOINT_POSITION_ONLY or self._obs_type == ObservationType.JOINT_POSITION_END_EFFECTOR:
                obs_state[key] = np.hstack((obs_state[key], joint_state["position"]))
            if self._obs_type == ObservationType.END_EFFECTOR_POSE or self._obs_type == ObservationType.JOINT_POSITION_END_EFFECTOR:
                obs_state[key] = np.hstack((obs_state[key], ee_states[key]["pose"]))
            if self._obs_type == ObservationType.DELTA_END_EFFECTOR_POSE:
                delta_ee = pose_diff(ee_states[key]["pose"], self._last_ee[key])
                obs_state[key] = np.hstack((obs_state[key], delta_ee))
                self._last_ee[key] = ee_states[key]["pose"]
            if self._obs_type == ObservationType.MASK:
                obs_state[key] = np.hstack((obs_state[key], np.zeros(7)))
            if self._contain_ft or self._obs_type == ObservationType.FT_ONLY:
                if ft_dict is None:
                    raise ValueError(f'The obs contain ft {self._contain_ft} with obs type {self._obs_type} need ft data from hardware!!!')
                obs_state[key] = np.hstack((obs_state[key], ft_dict[key]))
            # tool state
            # Training datasets (e.g., q2q) record the tool/gripper position as a real scalar
            # (in mm for Pika, range roughly [0, 90]). To match dataset stats at inference,
            # we must append the same physical quantity here (not a constant 0), otherwise
            # the last dimension's z-score will explode and harm predictions.
            if self._obs_type == ObservationType.MASK:
                # For MASK observations, keep a normalized mask value (0..1) if available
                tool_state = tools_dict[key]["position"] / self._tool_state_max
                obs_state[key] = np.hstack((obs_state[key], [0]))
            else:
                # Append the real tool position in the same units used by the dataset (e.g., mm)
                # tool_state = tools_dict[key]["position"] if tools_dict is not None else 0.0
                tool_state = tools_dict[key]["position"] / self._tool_state_max
                obs_state[key] = np.hstack((obs_state[key], [tool_state]))
                # log.info(f'obs tool state for {key} {tool_state}')
        camera_data = self.get_camera_infos()
        obs_dict = {'state': obs_state, 'colors': camera_data["color"], 
                    'depths': camera_data["depth"]}
        return obs_dict
    
    def _wait_key(self, right_key, desciption):
        while True:
            key = input(f'{desciption}')
            if key == right_key:
                break
    
    def set_joint_position(self, positions):
        # log.info(f'New command')
        if self._is_debug:
            self._robot_motion.update_execute_hardware(False)
            self._robot_motion.set_joint_positions(positions)
            while True:
                key = input(f'please press c to back position in sim!!!!')
                if key == 'c':
                    log.info(f'Back to original position')
                    current_position = self._robot_system.get_joint_states()._positions
                    self._robot_motion.set_joint_positions(current_position)
                    self._wait_key('c', 'please press c to execute hardware!!!!')
                    break             
        self._robot_motion.update_execute_hardware(self._use_hardware)
        self._robot_motion.set_joint_positions(positions, True)
        if self._is_debug:
            self._wait_key('c', 'please press c to proceed next command!!!!')
        
    def set_ee_pose(self, poses):
        if self._is_debug:
            self._robot_motion.update_execute_hardware(False)
            self._robot_motion.update_high_level_command(poses)
            while True:
                key = input(f'please press c to back to position in sim!!!!')
                if key == 'c':
                    cur_pose = None # @TODO: zyx
                    self._robot_motion.update_high_level_command(cur_pose)
                    self._wait_key('c', 'please press c to execute in hardware!!!!')
                    break
        self._robot_motion.update_execute_hardware(self._use_hardware)
        self._robot_motion.update_high_level_command(poses)
        # visual for targets
        visual_targets = {}
        key = {"single":[0,7]} if len(poses) <= 7 else {"left":[0,7], "right":[7,14]}
        for cur_key, indecies in key.items():
            visual_targets[cur_key] = poses[indecies[0]:indecies[1]]
        self._robot_motion.sim_visualize_targets(visual_targets)
        if self._is_debug:
            self._wait_key('c', 'please press c to proceed next command!!!!')
    
    def close(self):
        self._robot_motion.close()
    
    @abc.abstractmethod
    def compute_rewards(self):
        """
            Please implement your own reward logistics
        """
        return 0.0, False

    @abc.abstractmethod
    def get_info(self):
        """
            Please implement your own reward logistics
        """
        return {}
    
def main():
    from factory.utils import parse_args
    from hardware.base.utils import dynamic_load_yaml
    import os, random, cv2
    # testing gym api
    arguments = {"config": {"short_cut": "-c",
                            "symbol": "--config",
                            "type": str, 
                            "default": "factory/tasks/inferences_tasks/pi0/config/fr3_pi0_cfg.yaml",
                            "help": "Path to the config file"}}
    args = parse_args("pi0 inference", arguments)
    
    # Load configuration from the YAML file
    config = dynamic_load_yaml(args.config)
    
    fr3_gym = GymApi(config)
    i = 0
    # position_test = [0.013841196342567696, -0.5922128618960336, -0.023749771665655426, -2.4364204955424196, -0.016736788670502524, 1.921816076394133, 0.7890845040801358] # measure 
    position_test = [-0.17956917997172686, 0.31556980687824737, 
                     0.11859001558572345, -2.032536791881043, 
                     -0.06755871904469697, 2.4087075313408386, 
                     0.8794838978929802] # measure
    start_position = [-0.0016708960744338654, -0.7861180279768628, 
                      0.0011242960248884306, -2.3515749674770583, 
                      -0.0019059956649008665, 1.581831610375827, 
                      0.7904221443917989]
    pose_test = [0.62883634,  0.06791876,  0.18286911,  0.99813903, -0.03618238, -0.03220256, -0.03704464]
    start_pose = [3.09019823e-01,  1.99873044e-04,  4.84437265e-01,  9.99989858e-01,
                    -2.65401198e-03, -1.43284955e-03,  3.34460145e-03]
    # pose_test = [3.09019823e-01,  1.99873044e-04,  0.45,  9.99989858e-01,
    #                 -2.65401198e-03, -1.43284955e-03,  3.34460145e-03]
    pose_diff_test1 = [0,0,0.06, 0,0,0,1]
    pose_diff_test2 = [0,0,-0.06, 0,0,0,1]
    if config["action_type"] == "joint_position":
        start = start_position; test = position_test
    elif config["action_type"] == "joint_position_delta":
        start = [0,0,0,0,0,0,1.0]; test = [0,0,0,0,0,0,-1.0]
        log.info(f'satrt: {start}; test: {test}')
    elif config["action_type"] == "end_effector_pose":
        start = start_pose; test = pose_test
    elif config["action_type"] == "end_effector_pose_delta":
        start = pose_diff_test1; test = pose_diff_test2
    is_start = False
    while True:
        i += 1
        if i % 1000 == 0:
            position = test if is_start else start
            is_start = not is_start
            action = {"arm": position, "tool": np.array([random.choice([0, 1])])}
            log.info(f'action: {action}')
            res = fr3_gym.step(action)
            obs = res[0]
            for key, image in obs["colors"].items():
                # log.info(f'key: {key}, iamge: {image}')
                cv2.imshow(key, image)
                cv2.waitKey(1)
            # log.info(f'command: {position} for {config["action_type"]}')
        time.sleep(0.001)
        
        # reset pose test for a given pose
    cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()
    
