from simulation.mujoco.mujoco_sim import MujocoSim
from controller.impedance_controller import ImpedanceController
from motion.pin_model import RobotModel
from hardware.fr3.fr3_arm import Fr3Arm
import os 
import yaml, time
from cfg_handling import get_cfg
from factory.components.motion_factory import MotionFactory
from factory.components.robot_factory import RobotFactory
from hardware.base.utils import dynamic_load_yaml, convert_homo_2_7D_pose

def main():    
    motion_factory_cfg = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(motion_factory_cfg, '..', 'teleop/config/franka_3d_mouse_impedance.yaml')
    config = dynamic_load_yaml(config_path)
    
    config['use_trajectory_planner'] = False
    robot_system = RobotFactory(config)
    robot_motion_system = MotionFactory(config=config, robot=robot_system)
    
    test_times = 10
    target_site = "target_site"
    # target_mocap = target_site.split('_')[0]
    tcp_site = "TCP_site"
    tcp_mocap = tcp_site.split('_')[0]
    target_poses = []
    
    robot_system._robot.set_teaching_mode(True)
    for i in range(test_times):
        key = input(f'press enter for {i}th target pose')
        pose = robot_system._robot.get_ee_pose()
        print(f'{i}th target pose: {pose}')
        target_poses.append(convert_homo_2_7D_pose(pose))
        robot_system._simulation.set_target_mocap_pose(tcp_mocap, target_site)
        key = input("Press enter to continue...")

    robot_system._robot.set_teaching_mode(False)
    
    # execution
    
    for i in range(test_times):
        robot_motion_system.update_high_level_command(target_poses[i])
        time.sleep(0.1)
    
    # while True:
    #     # get mocap target
    #     target_value = mujoco.get_site_pose(target_site, "xyzw")
    #     target[model.ee_link] = target_value
    #     cur_tcp = mujoco.get_tcp_pose()
    #     joint_states = fr3.get_joint_states()
    #     mujoco.set_target_mocap_pose(tcp_mocap, cur_tcp)
    #     # success, joint_value, mode = impedance_controller.compute_controller(target, 
    #     #                                                                      joint_states)
    #     joint_value = model.id(joint_states._positions, joint_states._velocities, joint_states._accelerations)
    #     mode = "torque"
    #     mujoco.set_joint_command([mode] * len(joint_value), joint_value)
    #     # print(f'joint value: {joint_value}')
    #     fr3.set_joint_command(mode, joint_value)
    #     time.sleep(0.001)
        

if __name__ == "__main__":
    main()
    