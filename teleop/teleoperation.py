from factory.tasks.robot_teleoperation import TeleoperationFactory
from factory.components.motion_factory import MotionFactory
from factory.components.robot_factory import RobotFactory
from factory.utils import parse_args
from hardware.base.utils import dynamic_load_yaml
import os
import yaml
import glog as log

# 设置teleop模式环境变量，供硬件模块检测
os.environ['TELEOP_MODE'] = 'true'

def load_task_definitions():
    """加载任务定义"""
    task_file = "teleop/config/task_definitions.yaml"
    try:
        with open(task_file, 'r') as f:
            return yaml.safe_load(f)['tasks']
    except FileNotFoundError:
        log.info(f"Error: Task definitions file not found: {task_file}")
        return []
    except yaml.YAMLError as e:
        log.info(f"Error: Failed to parse task definitions: {e}")
        return []

def select_task(tasks):
    """显示任务列表并让用户选择"""
    log.info("\n" + "="*60)
    log.info("Available Tasks:")
    log.info("="*60)
    for task in tasks:
        log.info(f"  [{task['id']}] {task['name']}")
        log.info(f"      Description: {task['task_description']}")
        log.info(f"      Goal: {task['task_description_goal'][:50]}...")
        log.info("")
    log.info("="*60)

    while True:
        try:
            user_input = input(f"Select task (0-{len(tasks)-1}) or 'q' to quit: ").strip()
            if user_input.lower() == 'q':
                log.info("Exiting...")
                exit(0)

            task_id = int(user_input)
            if 0 <= task_id < len(tasks):
                return tasks[task_id]
            else:
                log.info(f"Please enter a number between 0 and {len(tasks)-1}")
        except ValueError:
            log.info("Please enter a valid number or 'q' to quit")

if __name__ == '__main__':
    arguments = {
        "config": {
            "short_cut": "-c",
            "symbol": "--config",
            "type": str,
            "default": "teleop/config/franka_3d_mouse.yaml",
            "help": "Path to the config file"
        },
        "task": {
            "short_cut": "-t",
            "symbol": "--task",
            "type": int,
            "default": -1,
            "help": "Task ID to use (skip interactive selection)"
        }
    }
    args = parse_args("Teleoperation for manipulation", arguments)

    # Load base configuration from the YAML file
    cur_path = os.path.dirname(os.path.abspath(__file__))
    config = dynamic_load_yaml(args.config)

    # Load and select task
    tasks = load_task_definitions()
    if not tasks:
        log.info("No tasks available. Please check task_definitions.yaml")
        exit(1)

    # Select task based on command line argument or interactive selection
    if args.task >= 0 and args.task < len(tasks):
        # Use command line argument if provided
        selected_task = tasks[args.task]
        log.info(f"\nUsing task from command line: [{selected_task['id']}] {selected_task['name']}")
    else:
        # Interactive selection
        selected_task = select_task(tasks)

    # Merge task configuration into main config
    config['save_path_prefix'] = selected_task['save_path_prefix']
    config['task_description'] = selected_task['task_description']
    config['task_description_goal'] = selected_task['task_description_goal']
    config['task_description_step'] = selected_task['task_description_step']

    log.info(f"\n✅ Selected Task: {selected_task['name']}")
    log.info(f"   Save Path: {selected_task['save_path_prefix']}")
    log.info(f"   Description: {selected_task['task_description']}")
    log.info(f"   Goal: {selected_task['task_description_goal']}")
    log.info(f"\nTask Steps:")
    log.info(selected_task['task_description_step'])
    log.info("\n" + "="*60)

    # create robot teleoperation system
    motion_config = config["motion_config"]
    robot_system = RobotFactory(motion_config)
    robot_motion_system = MotionFactory(motion_config, robot_system)
    robot_teleoperation = TeleoperationFactory(config, robot_motion_system)
    log.info(f'Started the teleoperation system')
    robot_teleoperation.create_robot_teleoperation_system()

    log.info(f'finished teleoperation process!!!')
    
