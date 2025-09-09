from factory.components.robot_factory import RobotFactory
from factory.components.motion_factory import MotionFactory
from hardware.base.utils import transform_pose
import glog as log
import abc, time
import gymnasium as gym
from dataset.utils import ActionType, dict_str2action_type
import numpy as np
from factory.components.motion_factory import Robot_Space
from scipy.spatial.transform import Rotation as R

class GymApi(gym.Env):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self._max_step_nums = config["max_step_nums"]
        
        self._action_type = config.get("action_type", "joint_position")
        self._action_type = dict_str2action_type[self._action_type]
        self._action_ori_type = config.get("action_orientation_type", "euler")
        robot_motion_cfg = config["motion_config"]
        
        self._reset_space = config.get("reset_space", "joint")
        self._reset_space = Robot_Space.JOINT_SPACE if self._reset_space== 'joint' else Robot_Space.CARTESIAN_SPACE
        self._reset_joint_position = config.get("reset_position", None)
        self._is_debug = config.get("is_debug", True)
        
        self._robot_system = RobotFactory(robot_motion_cfg)
        self._robot_motion = MotionFactory(robot_motion_cfg, self._robot_system)
        self._robot_motion.create_motion_components()
        self._robot_motion.update_execute_hardware(True)
        log.info("The robot motion component is successfully created in gym api!")
        
        # variable used for gym api
        self._step_counter = 0
        self.reset()
        
    def set_action_type(self, action_type: ActionType):
        self._action_type = action_type
        
    def step(self, action):
        # action execution
        arm_action = action['arm']
        execute_arm_action = np.array([]); execute_tool_action = []
              
        # @TODO: hack
        gripper_position_dof = 1
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
            for j, pose in enumerate(list(cur_ee_pose.values())):
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
                    cur_arm_action[3:] = R.from_euler(cur_arm_action[3:6]).as_quat()
                if self._action_type == ActionType.END_EFFECTOR_POSE_DELTA:
                   cur_arm_action = transform_pose(pose, cur_arm_action)
                execute_arm_action = np.hstack((execute_arm_action, cur_arm_action))
                self.set_ee_pose(execute_arm_action)
        else:
            raise ValueError(f"Unsupported action type: {self._action_type}")
        
        # tool execution
        tool_action = action['tool']
        tool_type_dict = self._robot_system.get_tool_dict_state()
        tool_index = 0
        if tool_type_dict is not None:
            for j in range(len(tool_type_dict)):
                index_r = tool_index + gripper_position_dof
                execute_tool_action.append(np.array(tool_action[tool_index:index_r]))
                tool_index = index_r
            self._robot_motion.set_tool_command(tool_action)
        
        # obs
        observation = self.get_observation()
        reward, done = self.compute_rewards()
        done = done or (self._step_counter >= self._max_step_nums)
        info = self.get_info()
        
        return observation, reward, done, False, info
        
    def reset(self, *, seed = None, options = None):
        self._robot_motion.reset_robot_system(arm_command=None,
                                              space=Robot_Space.JOINT_SPACE,
                                              tool_command=dict(single=np.array([1])))
    
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
        self._robot_system.get_joint_states()
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
                                                           cur_model_type)
            key = ee_index[i]
            poses[key] = dict(pose=frame_pose)
        return poses
    
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
    
    @abc.abstractmethod
    def get_observation(self):
        joint_states = self.get_joint_state()
        ee_states = self.get_ee_state()
        camera_data = self.get_camera_infos()
        tools_dict = self.get_tool_state()
        
        # other sensors: @TODO: zyx
        
        obs_dict = {'joint_states': joint_states, 'ee_states': ee_states, 
                    'tools': tools_dict, 'colors': camera_data["color"], 
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
        self._robot_motion.update_execute_hardware(True)
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
        self._robot_motion.update_execute_hardware(True)
        self._robot_motion.update_high_level_command(poses)
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
    