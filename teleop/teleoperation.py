from factory.tasks.robot_teleoperation import TeleoperationFactory
from factory.components.motion_factory import MotionFactory
from factory.components.robot_factory import RobotFactory
from hardware.base.utils import dynamic_load_yaml
import argparse, os

def parse_args():
    parser = argparse.ArgumentParser(description="Teleoperation for manipulation")
    parser.add_argument("-c", "--config", type=str, default="config/franka_3d_mouse.yaml", help="Path to the config file")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # Load configuration from the YAML file
    cur_path = os.path.dirname(os.path.abspath(__file__))
    config = dynamic_load_yaml(args.config)

    # create robot teleoperation system
    robot_system = RobotFactory(config)
    robot_motion_system = MotionFactory(config, robot_system)
    robot_teleoperation = TeleoperationFactory(config, robot_motion_system)
    print(f'Started the teleoperation system')
    robot_teleoperation.create_robot_teleoperation_system()
    
    print(f'finished teleoperation process!!!')
    