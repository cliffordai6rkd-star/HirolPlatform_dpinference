from factory.components.gym_interface import GymApi
import threading, time, cv2, os, random
import glog as log
import numpy as np
from factory.tasks.inferences_tasks.utils import display_images
from factory.tasks.inferences_tasks.inference_base import InferenceBase
import einops, copy
import matplotlib.pyplot as plt

# pi0 related
from openpi.policies import hirol_fr3_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config

# time statistics
from tools.performance_profiler import timer

class PI0_Inferencer(InferenceBase):
    def __init__(self, config):
        super().__init__(config)
        self._tasks = config["tasks"]
        self._execution_steps = config.get("execution_step", 5)
        self._execution_interruption = False
        
        # model loading
        model_cfg_name = config["model_cfg_name"]
        model_config = _config.get_config(model_cfg_name)
        model_dir = config["model_dir"]
        checkpoint_dir = download.maybe_download(model_dir)

        # Create a trained policy.
        self._pi0_policy = _policy_config.create_trained_policy(model_config, 
                                                    checkpoint_dir, default_prompt=self._tasks[0])
        
    def start_inference(self):
        execution_thread = None
        for episode_num in range(self._num_episodes):
            if self._quit: break
            
            self._gym_robot.reset()
            self._pi0_policy.reset()
            self._status_ok = True
            log.info(f'Starting the {episode_num} th episodes')
            while self._status_ok:
                with timer("gym_obs", "pi0_inferencer"):
                    pi0_obs = self.convert_from_gym_obs()
                    
                with timer("pi0_inference_time", "pi0_inferencer"):
                    result = self._pi0_policy.infer(pi0_obs)
                    
                    if execution_thread is not None and execution_thread.is_alive():
                        self._execution_interruption = True
                        execution_thread.join()
                    self._execution_interruption = False
                    execute_action = result["actions"]
                    # log.info(f'action shape: {execute_action[0].shape} for {episode_num}th episodes')
                    
                def multi_step_tasks():
                    with timer("gym_step", "pi0_inferencer"):
                        with self._lock:
                            joint_state = copy.deepcopy(self._joint_positions)
                        for i in range(self._execution_steps):
                            if self._execution_interruption or not self._status_ok:
                                break
                            # @TODO: hack for fr3, zyx
                            cur_action = execute_action[i]
                            self._plotter.update_signal(joint_state, cur_action)
                            # Process matplotlib events for plotter updates
                            plt.pause(0.001)  # Small pause to process GUI events
                            # log.info(f'Executing action for {i}th action: {cur_action}')
                            cur_gripper = 1.0 if cur_action[-1] > 0.055 else 0.0
                            action = {'arm': cur_action[:7], 'tool': np.array(cur_gripper)}
                            self._gym_robot.step(action)
                            time.sleep(0.001)
                execution_thread = threading.Thread(target=multi_step_tasks)
                execution_thread.start()
                log.info(f'result action chunk shape: {execute_action.shape}')
                
    def convert_from_gym_obs(self):
        gym_obs = super().convert_from_gym_obs()
        self._joint_positions = np.array([])
        self.image_display(gym_obs)
            
        pi0_obs = {}
        # @TODO: coupling solution for testing
        pi0_obs["state"] = np.array([])
        self._lock.acquire()
        self._joint_positions = np.array([])
        for key, joint_state in gym_obs['joint_states'].items():
            robot_state = joint_state["position"]
            self._joint_positions = np.hstack((self._joint_positions, robot_state, gym_obs["tools"][key]["position"]))
            if self._obs_contain_ee:
                robot_state = np.hstack((gym_obs[key]["ee_states"]))
            pi0_obs["state"] = np.hstack((pi0_obs["state"], robot_state, 
                                        gym_obs["tools"][key]["position"]))
        self._lock.release()
            
        for key, img in gym_obs["colors"].items():
            # log.info(f'{key} , shape: {img.shape}')
            pi0_obs[key] = img
            # pi0_obs[key] = cv2.resize(img, (224, 224))
            # if pi0_obs[key].shape[0] == 3:
            #     pi0_obs[key] = einops.rearrange(pi0_obs[key], "c h w -> h w c")
            # log.info(f'after shape: {pi0_obs[key].shape}')
            
        if isinstance(self._tasks, list):
            selected_index = random.randrange(len(self._tasks))
            pi0_obs["task"] = self._tasks[selected_index]
        else: pi0_obs["task"] = self._tasks
        log.info(f'pi0_obs: {pi0_obs["state"]}')
        return pi0_obs
        
    def convert_to_gym_action(self):
        pass
    
    def close(self):
        """Clean up resources and close display windows."""
        if self._enable_display:
            cv2.destroyWindow(self._display_window_name)
        self._gym_robot.close()
        del self._pi0_policy
          
def main():
    from factory.utils import parse_args
    from hardware.base.utils import dynamic_load_yaml
    # testing gym api
    arguments = {"config": {"short_cut": "-c",
                            "symbol": "--config",
                            "type": str, 
                            "default": "factory/tasks/inferences_tasks/pi0/config/fr3_pi0_cfg.yaml",
                            "help": "Path to the config file"}}
    args = parse_args("pi0 inference", arguments)
    
    # Load configuration from the YAML file
    config = dynamic_load_yaml(args.config)
    log.info(f'pi0 config: {config}')
    pi0_executor = PI0_Inferencer(config)
    pi0_executor.start_inference()
    
if __name__ == "__main__":
    main()
    