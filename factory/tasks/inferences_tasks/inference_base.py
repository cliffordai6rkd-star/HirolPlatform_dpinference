import abc
from factory.components.gym_interface import GymApi
from factory.tasks.inferences_tasks.utils import display_images, AnimationPlotter
from dataset.utils import ActionType, dict_str2action_type
import threading, time, cv2, os
from sshkeyboard import listen_keyboard, stop_listening
import glog as log

class InferenceBase(abc.ABC, metaclass=abc.ABCMeta):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self._gym_robot = GymApi(config)
        self._status_ok = True
        self._quit = False
        
        self._action_type = config["action_type"]
        self._action_type = dict_str2action_type[self._action_type]
        self._action_ori_type = config.get("action_orientation_type", "euler")
        self._obs_contain_ee = config.get("obs_contain_ee", False)
        self._num_episodes = config.get("num_episodes", 10)
        
        # display
        self._enable_display = config.get("enable_display", True)
        self._display_window_name = config["display_window_name"]
        
        # keyboard listening
        listen_keyboard_thread = threading.Thread(target=listen_keyboard, 
                kwargs={"on_press": self._keyboard_on_press, 
                        "until": None, "sequential": False,}, daemon=True)
        listen_keyboard_thread.start()
        
        # animation plotter
        self._joint_positions = None
        self._lock = threading.Lock()
        dof_list = self._gym_robot._robot_motion.get_model_dof_list()[1:]
        index = ["left", "right"] if len(dof_list) > 1 else ["single"]
        joint_state_names = []; action_names = []
        for i, dof in enumerate(dof_list):
            for j in range(dof):
                joint_state_names.append(f'{index[i]}_joint{j}')
                action_names.append(f'{index[i]}_action{j}')
        # hack for tools @TODO: zyx
        joint_state_names.append(f'gripper_state')
        action_names.append(f'gripper_action')
        self._plotter = AnimationPlotter(joint_state_names, action_names)
        self._plotter.start_animation()
        self._plotter.start_main_thread_updater()
    
    @abc.abstractmethod
    def convert_from_gym_obs(self):
        gym_obs = self._gym_robot.get_observation()
        return gym_obs
    
    @abc.abstractmethod
    def convert_to_gym_action(self, model_action):
        self._gym_robot.step(model_action)

    @abc.abstractmethod
    def start_inference(self):
        """Start the main inference loop.
        
        Continuously processes observations, runs inference, and executes actions
        until interrupted by keyboard input.
        """
        raise NotImplementedError
    
    def close(self):
        pass
    
    def image_display(self, gym_obs):
        if self._enable_display and gym_obs.get("colors"):
            display_images(gym_obs["colors"], self._display_window_name)    
    
    
    def _keyboard_on_press(self, key: str) -> None:
        # quit
        if key == 'q':
            log.info("Quit command received, shutting down...")
            print(f"{'='*15}Closing the inference thread!!!{'='*15}")
            self._gym_robot.close()
            stop_listening()
            self._quit = True
            self._status_ok = False
            self._plotter.clear_data()
            self._plotter.stop_animation()
            time.sleep(1.5)
            self.close()
            cv2.destroyAllWindows()
        # reset
        elif key == 'r':
            log.info("Reset command received")
            self._gym_robot.reset()
        elif key == 'd':
            self._status_ok = False
            log.info(f"Set done to True for current episode!!!")
