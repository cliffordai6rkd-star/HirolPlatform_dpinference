import os
import json
import cv2
import numpy as np
import time, enum
from scipy.spatial.transform import Rotation as R
import rerun as rr
import rerun.blueprint as rrb
from datetime import datetime
os.environ["RUST_LOG"] = "error"

class ActionType(enum.Enum):
    JOINT_POSITION = 0
    JOINT_POSITION_DELTA = 1
    END_EFFECTOR_POSE = 2
    END_EFFECTOR_POSE_DELTA = 3
    JOINT_TORQUE = 4

class RerunEpisodeReader:
    def __init__(self, task_dir = ".", json_file="data.json", action_type: ActionType = ActionType.JOINT_POSITION):
        self.task_dir = task_dir
        self.json_file = json_file
        self.action_type = action_type

    def return_episode_data(self, episode_idx):
        # Load episode data on-demand
        episode_dir = os.path.join(self.task_dir, f"episode_{episode_idx:04d}")
        json_path = os.path.join(episode_dir, self.json_file)

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Episode {episode_idx} data.json not found.")

        with open(json_path, 'r', encoding='utf-8') as jsonf:
            json_file = json.load(jsonf)

        episode_data = []

        # Loop over the data entries and process each one
        last_state_data = None
        for item_data in json_file['data']:
            # Process images and other data
            colors = self._process_images(item_data, 'colors', episode_dir)
            depths = self._process_images(item_data, 'depths', episode_dir)
            audios = self._process_audio(item_data, 'audios', episode_dir)

            # Append the data in the item_data list
            cur_actions = {}
            if self.action_type == ActionType.JOINT_POSITION:
                joint_states = item_data.get("joint_states", {})
                cur_actions = self._get_absolute_action(joint_states,
                                            attribute_name="position")
            elif self.action_type == ActionType.END_EFFECTOR_POSE:
                cur_actions = self._get_absolute_action(item_data.get("ee_states", {}))
            elif self.action_type == ActionType.JOINT_POSITION_DELTA:
                joint_states = item_data.get("joint_states", {})
                cur_actions = self._get_delta_action(joint_states, last_state_data, "position")
                last_state_data = joint_states
            elif self.action_type == ActionType.END_EFFECTOR_POSE_DELTA:
                # @TODO: 欧拉角转换 for check
                ee_states = item_data.get("ee_states", {})
                for key, cur_ee_state in ee_states.items():
                    ee_states[key][3:] = R.from_quat(cur_ee_state[3:]).as_euler('xyz')
                cur_actions = self._get_delta_action(ee_states, last_state_data)
                for key, action in cur_actions.items():
                    cur_actions[key][3:] = R.from_euler(action[3:], "xyz").as_quat()
                last_state_data = ee_states
            else:
                raise ValueError(f'The action type {self.action_type} is not supported for reading episode data')
            
            # @TODO: add the gripper states as action for check
            tool_states = item_data.get("tools", {})
            for key, tool_state in tool_states.items():
                cur_actions[key] = np.hstack((cur_actions[key], tool_state["position"]))
            
            episode_data.append(
                {
                    'idx': item_data.get('idx', 0),
                    'colors': colors,
                    'depths': depths,
                    'joint_states': item_data.get('joint_states', {}),
                    'ee_states': item_data.get('ee_states', {}),
                    'tools': item_data.get('tools', {}),
                    'imus': item_data.get('imus', {}),
                    'tactiles': item_data.get('tactiles', {}),
                    'audios': audios,
                    'actions': cur_actions
                }
            )
            
        
        return episode_data
    
    def get_episode_text_info(self, episode_id):
        episode_dir = os.path.join(self.task_dir, f"episode_{episode_id:04d}")
        json_path = os.path.join(episode_dir, self.json_file)

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Episode {episode_id} data.json not found.")

        with open(json_path, 'r', encoding='utf-8') as jsonf:
            json_file = json.load(jsonf)

        text_info = json_file["text"]
        text_info = text_info["desc"] + text_info["steps"] + text_info["goal"]        
        return text_info
    
    def _get_absolute_action(self, states, attribute_name = None):
        cur_action = {}
        for key, state in states.items():
            if attribute_name is not None:
                cur_action[key] = state[attribute_name]
            else:
                cur_action[key] = state
        return cur_action
    
    def _get_delta_action(self, states, last_state_data, attribute_name = None):
        cur_action = {}
        last_state_value = {}
        for key, state in last_state_data.items():
            state_value = state if attribute_name is None else state[attribute_name]
            last_state_value[key] = state_value
        
        for key, state in states.items():
            state_value = state if attribute_name is None else state[attribute_name]
            if last_state_data is None:
                cur_action[key] = [0] * len(state_value)
            else:
                cur_action[key] = state_value - last_state_value[key]
        return cur_action

    def _process_images(self, item_data, data_type, dir_path):
        images = {}

        for key, file_name in item_data.get(data_type, {}).items():
            if file_name:
                file_path = os.path.join(dir_path, file_name)
                if os.path.exists(file_path):
                    image = cv2.imread(file_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images[key] = image
        return images

    def _process_audio(self, item_data, data_type, episode_dir):
        audio_item = item_data.get(data_type, {})
        if audio_item is None:
            return
        
        audio_data = {}
        dir_path = os.path.join(episode_dir, data_type)

        for key, file_name in audio_item.items():
            if file_name:
                file_path = os.path.join(dir_path, file_name)
                if os.path.exists(file_path):
                    pass  # Handle audio data if needed
        return audio_data

class RerunLogger:
    def __init__(self, prefix = "", IdxRangeBoundary = 30, memory_limit = None):
        self.prefix = prefix
        self.IdxRangeBoundary = IdxRangeBoundary
        rr.init(datetime.now().strftime("Runtime_%Y%m%d_%H%M%S"))
        if memory_limit:
            rr.spawn(memory_limit = memory_limit, hide_welcome_screen = True)
        else:
            rr.spawn(hide_welcome_screen = True)

        # Set up blueprint for live visualization
        if self.IdxRangeBoundary:
            self.setup_blueprint()

    def setup_blueprint(self):
        views = []

        data_plot_paths = [
                           f"{self.prefix}left_arm", 
                           f"{self.prefix}right_arm", 
                           f"{self.prefix}left_ee", 
                           f"{self.prefix}right_ee"
        ]
        for plot_path in data_plot_paths:
            view = rrb.TimeSeriesView(
                origin = plot_path,
                time_ranges=[
                    rrb.VisibleTimeRange(
                        "idx",
                        start = rrb.TimeRangeBoundary.cursor_relative(seq = -self.IdxRangeBoundary),
                        end = rrb.TimeRangeBoundary.cursor_relative(),
                    )
                ],
                plot_legend = rrb.PlotLegend(visible = True),
            )
            views.append(view)

        # image_plot_paths = [
        #                     f"{self.prefix}colors/color_0",
        #                     f"{self.prefix}colors/color_1",
        #                     f"{self.prefix}colors/color_2",
        #                     f"{self.prefix}colors/color_3"
        # ]
        # for plot_path in image_plot_paths:
        #     view = rrb.Spatial2DView(
        #         origin = plot_path,
        #         time_ranges=[
        #             rrb.VisibleTimeRange(
        #                 "idx",
        #                 start = rrb.TimeRangeBoundary.cursor_relative(seq = -self.IdxRangeBoundary),
        #                 end = rrb.TimeRangeBoundary.cursor_relative(),
        #             )
        #         ],
        #     )
        #     views.append(view)

        grid = rrb.Grid(contents = views,
                        grid_columns=2,               
                        column_shares=[1, 1],
                        row_shares=[1, 1], 
        )
        views.append(rr.blueprint.SelectionPanel(state=rrb.PanelState.Collapsed))
        views.append(rr.blueprint.TimePanel(state=rrb.PanelState.Collapsed))
        rr.send_blueprint(grid)


    def log_item_data(self, item_data: dict):
        rr.set_time_sequence("idx", item_data.get('idx', 0))

        # Log states
        states = item_data.get('states', {}) or {}
        for part, state_info in states.items():
            if part != "body" and state_info:
                values = state_info.get('qpos', [])
                for idx, val in enumerate(values):
                    rr.log(f"{self.prefix}{part}/states/qpos/{idx}", rr.Scalar(val))

        # Log actions
        actions = item_data.get('actions', {}) or {}
        for part, action_info in actions.items():
            if part != "body" and action_info:
                values = action_info.get('qpos', [])
                for idx, val in enumerate(values):
                    rr.log(f"{self.prefix}{part}/actions/qpos/{idx}", rr.Scalar(val))

        # # Log colors (images)
        # colors = item_data.get('colors', {}) or {}
        # for color_key, color_val in colors.items():
        #     if color_val is not None:
        #         rr.log(f"{self.prefix}colors/{color_key}", rr.Image(color_val))

        # # Log depths (images)
        # depths = item_data.get('depths', {}) or {}
        # for depth_key, depth_val in depths.items():
        #     if depth_val is not None:
        #         # rr.log(f"{self.prefix}depths/{depth_key}", rr.Image(depth_val))
        #         pass # Handle depth if needed

        # # Log tactile if needed
        # tactiles = item_data.get('tactiles', {}) or {}
        # for hand, tactile_vals in tactiles.items():
        #     if tactile_vals is not None:
        #         pass # Handle tactile if needed

        # # Log audios if needed
        # audios = item_data.get('audios', {}) or {}
        # for audio_key, audio_val in audios.items():
        #     if audio_val is not None:
        #         pass  # Handle audios if needed

    def log_episode_data(self, episode_data: list):
        for item_data in episode_data:
            self.log_item_data(item_data)



if __name__ == "__main__":
# episode_reader = RerunEpisodeReader(task_dir = unzip_file_output_dir)
# # TEST EXAMPLE 1 : OFFLINE DATA TEST
# episode_data6 = episode_reader.return_episode_data(6)
# logger_mp.info("Starting offline visualization...")
# offline_logger = RerunLogger(prefix="offline/")
# offline_logger.log_episode_data(episode_data6)
# logger_mp.info("Offline visualization completed.")
    
    data_folder = "./data/episode_0052"
    episode_reader = RerunEpisodeReader(task_dir='./data', action_type=ActionType.JOINT_POSITION_DELTA)
    data52 = episode_reader.return_episode_data(52)
    
    