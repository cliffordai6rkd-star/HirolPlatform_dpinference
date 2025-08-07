import abc, os
import enum
from dataset.lerobot.reader import RerunEpisodeReader, ActionType

class DataLoaderBase(abc.ABC, metaclass=abc.ABCMeta):
    def __init__(self, config, task_dir:str, json_file_name:str = "data.json", action_type:ActionType = ActionType.JOINT_POSITION):
        self._config = config
        self._task_dir = task_dir
        self._json_file = json_file_name
        self._action_type = action_type
        self._all_episode_data = {}
        self._all_episode_text = []
    
    """
        parse for all episode data for one task and attach the action values
    """
    def load_all_episodes(self, task_dir):
        self._episode_reader = RerunEpisodeReader(task_dir=task_dir,
                                                  json_file=self._json_file,
                                                  action_type=self._action_type)
        episode_id = 0
        all_episode_data = []
        print(f'task_dir: {task_dir}')
        dirs = os.listdir(task_dir)
        for dir_name in dirs:
            print(f'dirname: {dir_name}')
            if 'episode' in dir_name:
                episode_number = int(dir_name.lstrip("episode_"))
                episode_id = episode_number
                print(f'Tring to load the {episode_number}th episode data in {task_dir}')
                episode_data = self._episode_reader.return_episode_data(episode_number)
                all_episode_data.append(episode_data)
        print(f'Finished loading for all episode data in {task_dir}') 
        self._all_episode_data[task_dir] = all_episode_data
        self._all_episode_text.append(self._episode_reader.get_episode_text_info(episode_id))
        
    @abc.abstractmethod
    def convert_dataset(self):
        raise NotImplementedError
    