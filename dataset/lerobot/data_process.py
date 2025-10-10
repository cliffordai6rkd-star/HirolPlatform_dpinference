"""
    Contributions to the unitree robotics xr_teleoperation projects
    with modification by ZYX
"""

import os
import cv2
import json
import datetime
import numpy as np
import time
from dataset.lerobot.rerun_visualizer import RerunLogger
from queue import Queue, Empty
import warnings
from threading import Thread
import glog as log

class EpisodeWriter():
    def __init__(self, task_dir, frequency=30, image_size=[640, 480], rerun_log = True, 
                 robot_info = None, version = "1.0.0", date = None, author = None, 
                 task_description = None, task_description_goal = None, task_description_steps = None):
        """
            image_size: [width, height]
            # Episode data format
                array data (saved to episode_dir/data.json):
                    item_id (trajectory id, nums of step in one episode),
                    joint_states, ee_states(end-effector pose), imus, tactiles(array)
                    tools: end-effector tools state (gripper, dexterous hand ...)
                image data (saved to episode_dir/keys):
                    colors (e.g.episode_dir/colors/<id_head>.jpg), 
                    depths (e.g.episode_dir/depths/<id_head>.jpg), 
                    tactiles(image, e.g.episode_dir/tactiles/id_<left_finger>.jpg)
                audio data (saved to epiode_dir/audios/keys):
                    audios (e.g.episode_dir/audios/audio_id_<left_mic>.npy)
            # save path:
                Each episode data is stored under the task dir/<episode_id>, e/g.
                the first episde is stored in task_dir/episode_0000
        """
        log.info("==> EpisodeWriter initializing...\n")
        self.task_dir = task_dir
        self.frequency = frequency
        self.image_size = image_size
        self.robot_info = robot_info

        self.rerun_log = rerun_log
        if self.rerun_log:
            log.info("==> RerunLogger initializing...\n")
            self.rerun_logger = RerunLogger(self.task_dir, prefix="online/", IdxRangeBoundary = 60, memory_limit = "300MB")
            log.info("==> RerunLogger initializing ok.\n")
        
        self.data = {}
        self.episode_data = []
        self.item_id = -1
        self.episode_id = -1
        if os.path.exists(self.task_dir):
            episode_dirs = [episode_dir for episode_dir in os.listdir(self.task_dir) if 'episode_' in episode_dir]
            episode_last = sorted(episode_dirs)[-1] if len(episode_dirs) > 0 else None
            self.episode_id = 0 if episode_last is None else int(episode_last.split('_')[-1])
            log.info(f"==> task_dir directory already exist, now self.episode_id is:{self.episode_id}\n")
        else:
            os.makedirs(self.task_dir)
            log.info(f"==> episode directory does not exist, now create one.\n")
        self.data_info(version=version, date=date, author=author)
        self.text = {}
        self.add_text_prompt(task_description_goal, task_description, task_description_steps)
        self.text_desc()

        self.is_available = True  # Indicates whether the class is available for new operations
        # Initialize the queue and worker thread
        self.item_data_queue = Queue(-1)
        self.stop_worker = False
        self.need_save = False  # Flag to indicate when save_episode is triggered
        self.worker_thread = Thread(target=self.process_queue)
        self.worker_thread.start()

        log.info("==> EpisodeWriter initialized successfully.\n")

    def data_info(self, version='1.0.0', date=None, author=None):
        self.info = {
                "version": "1.0.0" if version is None else version, 
                "date": datetime.date.today().strftime('%Y-%m-%d') if date is None else date,
                "author": "HIROL" if author is None else author,
                "image": {"width":self.image_size[0], "height":self.image_size[1], "fps":self.frequency},
                "depth": {"width":self.image_size[0], "height":self.image_size[1], "fps":self.frequency},
                "audio": {"sample_rate": 16000, "channels": 1, "format":"PCM", "bits":16},    # PCM_S16
                "joint_names": None if self.robot_info is None else self.robot_info["joint_names"],
                "tactile_names": None if self.robot_info is None else self.robot_info["tactile_names"], 
                "sim_state": ""
            }
    def text_desc(self):
        if self.text is None:
            self.text = {
                "goal": "Pick up the red cup on the table.",
                "desc": "Pick up the cup from the table and place it in another position. The operation should be smooth and the water in the cup should not spill out",
                "steps":"step1: searching for cups. step2: go to the target location. step3: pick up the cup",
            }
        
    def add_text_prompt(self, goal, desc, steps):
        if goal is None or desc is None or steps is None:
            warnings.warn(f'Task description is not fully defined!!!!')
            return 
        
        self.text["goal"] = goal
        self.text["desc"] = desc
        self.text["steps"] = steps
 
    def create_episode(self):
        """
        Create a new episode, called if you want to start to record a new episode data
        Returns:
            bool: True if the episode is successfully created, False otherwise.
        Note:
            Once successfully created, this function will only be available again after save_episode complete its save task.
        """
        if not self.is_available:
            log.info("==> The class is currently unavailable for new operations. Please wait until ongoing tasks are completed.")
            return False  # Return False if the class is unavailable

        # Reset episode-related data and create necessary directories
        self.item_id = -1
        self.episode_data = []
        self.episode_id = self.episode_id + 1
        
        self.episode_dir = os.path.join(self.task_dir, f"episode_{str(self.episode_id).zfill(4)}")
        self.color_dir = os.path.join(self.episode_dir, 'colors')
        self.depth_dir = os.path.join(self.episode_dir, 'depths')
        self.tactile_dir = os.path.join(self.episode_dir, 'tactiles')
        self.audio_dir = os.path.join(self.episode_dir, 'audios')
        self.json_path = os.path.join(self.episode_dir, 'data.json')
        os.makedirs(self.episode_dir, exist_ok=True)
        os.makedirs(self.color_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.tactile_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)
        if self.rerun_log:
            self.online_logger = RerunLogger(self.episode_dir, prefix="online/", IdxRangeBoundary = 60, memory_limit="300MB")

        self.is_available = False  # After the episode is created, the class is marked as unavailable until the episode is successfully saved
        log.info(f"==> New episode created: {self.episode_dir}")
        return True  # Return True if the episode is successfully created
        
    def add_item(self, colors, depths=None, joint_states=None, ee_states=None, 
                tools=None, tactiles=None, imus=None, audios=None, actions=None):
        """
            called from external to add new data for current episode,
            please make sure your data (except for image & audio data) are python list instead of np array 
        """
        # Increment the item ID
        self.item_id += 1
        # Create the item data dictionary, @TODO: whether to keep sim state
        item_data = {
            'idx': self.item_id,
            'colors': colors,
            'depths': depths,
            'joint_states': joint_states,
            'ee_states': ee_states,
            'tactiles': tactiles,
            'imus': imus,
            'audios': audios,
            'tools': tools,
            'actions': actions
        }
        # Enqueue the item data
        self.item_data_queue.put(item_data)

    def process_queue(self):
        while not self.stop_worker or not self.item_data_queue.empty():
            # Process items in the queue
            try:
                item_data = self.item_data_queue.get(timeout=1)
                # try:
                self._process_item_data(item_data)
                # except Exception as e:
                #     log.info(f"Error processing item_data (idx={item_data['idx']}): {e}")
                self.item_data_queue.task_done()
            except Empty:
                pass
        
            # Check if save_episode was triggered
            if self.need_save and self.item_data_queue.empty():
                self._save_episode()

    def _process_item_data(self, item_data):
        idx = item_data['idx']
        colors = item_data.get('colors', {})
        depths = item_data.get('depths', {})
        audios = item_data.get('audios', {})
        tactiles = item_data.get('tactiles', {})

        # Save images
        if colors:
            self._add_images_to_item_data(idx, colors, item_data, self.color_dir, 'colors', attribute_name='data')

        # Save depths
        if depths:
            self._add_images_to_item_data(idx, depths, item_data, self.depth_dir, 'depths', attribute_name='data')
                
        # save tactiles
        if tactiles:
            self._add_tactile_to_item_data(idx, tactiles, item_data)

        # Save audios
        if audios:
            for mic, audio in audios.items():
                audio_name = f'audio_{str(idx).zfill(6)}_{mic}.npy'
                np.save(os.path.join(self.audio_dir, audio_name), audio.astype(np.int16))
                item_data['audios'][mic] = os.path.join('audios', audio_name)

        # Update episode data
        self.episode_data.append(item_data)

        # Log data if necessary
        if self.rerun_log:
            curent_record_time = time.time()
            log.info(f"==> episode_id:{self.episode_id}  item_id:{idx}  current_time:{curent_record_time}")
            self.rerun_logger.log_item_data(item_data)

    def save_episode(self):
        """
        Trigger the save operation. This sets the save flag, and the process_queue thread will handle it.
        """
        self.need_save = True  # Set the save flag
        log.info(f"==> Episode saved start...")

    def _save_episode(self):
        """
        Save the episode data to a JSON file.
        """
        self.data['info'] = self.info
        self.data['text'] = self.text
        self.data['data'] = self.episode_data
        with open(self.json_path, 'w', encoding='utf-8') as jsonf:
            jsonf.write(json.dumps(self.data, indent=4, ensure_ascii=False))
        self.need_save = False     # Reset the save flag
        self.is_available = True   # Mark the class as available after saving
        log.info(f"==> Episode saved successfully to {self.json_path}.")

    def close(self):
        """
        Stop the worker thread and ensure all tasks are completed.
        """
        self.item_data_queue.join()
        if not self.is_available:  # If self.is_available is False, it means there is still data not saved.
            self.save_episode()
        while not self.is_available:
            time.sleep(0.01)
        self.stop_worker = True
        self.worker_thread.join()
        
    def _add_images_to_item_data(self, idx, image_data, item_data, save_dir, image_desc, attribute_name=None):
        for id, (image_key, image_value) in enumerate(image_data.items()):
            image_data = image_value[attribute_name] if attribute_name is not None else image_value
            extension = '.jpg' if '_color' in image_key else '.png'
            image_name = f'{str(idx).zfill(6)}_{image_key}{extension}'
            if not cv2.imwrite(os.path.join(save_dir, image_name), image_data):
                log.info(f"Failed to save {image_desc} image.")
            item_data[image_desc][image_key] = dict(
                path = os.path.join(image_desc, image_name),
                time_stamp = image_value["time_stamp"]
            )
    
    def _add_tactile_to_item_data(self, idx: int, tactiles: dict, item_data: dict):
        """Save tactile sensor data as numpy arrays"""
        for sensor_name, sensor_data in tactiles.items():
            # Extract tactile data array
            tactile_array = sensor_data['data']
            timestamp = sensor_data.get('timestamp', 0)
            
            # Save tactile data as .npy file
            tactile_filename = f'tactile_{str(idx).zfill(6)}_{sensor_name}.npy'
            tactile_path = os.path.join(self.tactile_dir, tactile_filename)
            np.save(tactile_path, tactile_array.astype(np.int32))
            
            # Store reference in episode data
            if 'tactiles' not in item_data:
                item_data['tactiles'] = {}
            item_data['tactiles'][sensor_name] = {
                'path': os.path.join('tactiles', tactile_filename),
                'timestamp': timestamp,
                'shape': tactile_array.shape
            }

    def _dict_contain_images(self, dict_data: dict):
        for key, value in dict_data.items():
            if value.dtype == np.uint8 or value.dtype == np.uint16:
                return True
            
        return False
    
    
class EpisodeLoader():
    def __init__(self, task_dir):
        pass    
    
    # def