from reader import RerunEpisodeReader, ActionType
from dataset.lerobot.data_loader_base import DataLoaderBase
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import os
import numpy as np
import tqdm

class LerobotLoader(DataLoaderBase):
    def __init__(self, config, task_dir, json_file_name = "data.json", action_type = ActionType.JOINT_POSITION):
        super().__init__(config, task_dir, json_file_name, action_type)
        self._push_to_repo = self._config.get("push_to_repo", False)
        self._robot_name = config.get("robot_name", "fr3")
        self._repo_name = config.get("repo_name", "peg_in_hole")
        self._task_list = config.get("task_list", None)
        # 一种是只给一个task的文件夹
        if self._task_list is None:
            self._task_list = [self._task_dir]
        else: # 另一种是个所有task的文件夹并给每个task的名字在list里
            for i, task in enumerate(self._task_list):
                cur_task_dir = os.path.join(self._task_dir, task)
                if not os.path.exists(cur_task_dir):
                    raise ValueError(f'{cur_task_dir} did not exist, please check the value of task dir and task list')
                self._task_list[i] = cur_task_dir
        for cur_task_dir in self._task_list:
            self.load_all_episodes(cur_task_dir)
        
        # try to confirm the images and state length,
        # @TODO: Assuming all tasks having same observations and actions
        example_episode = self._all_episode_data[self._task_list[0]][0]
        example_step = example_episode[0]
        images = example_step.get("colors", {})
        depths = example_step.get("depths", {})
        self._image_keys = []; self._image_resolutions = []
        for key, image in images.items():
            self._image_keys.append(key)
            self._image_resolutions.append(tuple(image.shape))
        for key, image in depths.items():
            self._image_keys.append(key)
            self._image_resolutions.append(tuple(image.shape))
        # @TODO: Concataneate the tool state into the end
        self._joint_states_dim = 0; self._action_dim = 0
        joint_states = example_step.get("joint_states", {})
        actions = example_step.get("actions", {})
        for key, state in joint_states.items():
            self._joint_states_dim += len(state["position"])
        for key, action in actions.items():
            self._action_dim += len(action)
        
        self._tool_state_dim = 0
        tool_states = example_step.get("tools", {})
        for key, tool_state in tool_states.items():
            cur_tool_state_value = tool_state["position"]
            if isinstance(cur_tool_state_value, list):
                self._tool_state_dim += len(tool_state["position"])
            else: self._tool_state_dim += 1 # for floating number case
        self._total_obs_state_dim = self._joint_states_dim + self._tool_state_dim
        
    def convert_dataset(self):
        feature_dicts = {}
        for i, key in enumerate(self._image_keys):
            feature_dicts[key] = dict(dtype = "image",
                                shape = self._image_resolutions[i],
                    names = ["height", "width", "channel"],)
        feature_dicts["state"] = {"dtype": "float64",
                                    "shape": (self._total_obs_state_dim,),
                                    "names": ["state"],}
        feature_dicts["actions"] = {"dtype": "float64",
                                    "shape": (self._action_dim,),
                                    "names": ["actions"],}
        
        self._lerobot_dataset = LeRobotDataset.create(
            repo_id=self._repo_name,
            robot_type=self._robot_name,
            fps=30,
            features=feature_dicts,
            image_writer_threads=10,
            image_writer_processes=5,
        )
        
        # add episode data
        # for raw_dataset_name in RAW_DATASET_NAMES: # iterate about different dataset (task)
        #     raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
        #     for episode in raw_dataset:
        #         for step in episode["steps"].as_numpy_iterator():
        #             self._lerobot_dataset.add_frame(
        #                 {
        #                     "image": step["observation"]["image"],
        #                     "wrist_image": step["observation"]["wrist_image"],
        #                     "state": step["observation"]["state"],
        #                     "actions": step["action"],
        #                     "task": step["language_instruction"].decode(),
        #                 }
        #             )
        #         self._lerobot_dataset.save_episode()
        
        # 每一个task下的所有episodes
        for i, (task_dir, task_episode_data) in tqdm(enumerate(self._all_episode_data.items()), desc=f"processing f{task_dir} task", unit="task"):
            text_info = self._all_episode_text[i]
            # 同一个task下的每一个episode
            for episode_data in tqdm(task_episode_data, desc=f"processing episodes", unit="episode"):
                for step in tqdm(episode_data, desc=f"processing steps", unit="step"):
                    frame_feature = {}
                    for image_key in self._image_keys:
                        step_key = "colors" if "color" in image_key else "depths"
                        # print(f'step_key: {step_key}, image_key: {image_key} step: {step[step_key]}')
                        
                        frame_feature[image_key] = step[step_key][image_key]
                    frame_feature["state"] = np.array([])
                    tool_states = step["tools"]
                    # state: [arm state, tool state, another_arm_state, another_tool_state]
                    for key, value in step["joint_states"].items():
                        cur_feature_state = np.hstack((value["position"], tool_states[key]["position"]))
                        frame_feature["state"] = np.hstack((frame_feature["state"], cur_feature_state))
                    frame_feature["actions"] = np.array([])
                    for key, value in step["actions"].items():
                        frame_feature["actions"] = np.hstack((frame_feature["actions"], value))
                    frame_feature["task"] = text_info.decode()
                    
                    self._lerobot_dataset.add_frame(frame=frame_feature)
                self._lerobot_dataset.save_episode()
            
        # Optionally push to the Hugging Face Hub
        if self._push_to_repo:
            self._lerobot_dataset.push_to_hub(
                tags=[self._robot_name, self._repo_name],
                private=True,
                push_videos=True,
                license="apache-2.0",
            )
        return self._lerobot_dataset
    
    
if __name__ == '__main__':
    import yaml
    config_file = "config/lerobot_config.yaml"
    cur_path = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(cur_path, config_file)
    with open(cfg_file, 'r') as stream:
        config = yaml.safe_load(stream)
    print(f'config: {config}')
    task_dir = "data"
   
    lerobot_dataset = LerobotLoader(config, task_dir)
    dataset = lerobot_dataset.convert_dataset()
    print(f'len dataset: {len(dataset)}')
    print(f'dataset: {dataset}')
    # lerobot自动转换为tensor
    data_sample = next(iter(dataset))
    print(f'data: {data_sample}')
    