from simulation.mujoco.mujoco_sim import MujocoSim
from controller.impedance_controller import ImpedanceController
from motion.pin_model import RobotModel
from hardware.fr3.fr3_arm import Fr3Arm
import os 
import yaml, time
from cfg_handling import get_cfg

def main():
    impedance_config = "controller/config/impedance_fr3_cfg.yaml"
    impedance_config = get_cfg(impedance_config)
    model_config = "motion/config/robot_model_fr3_cfg.yaml"
    model_config = get_cfg(model_config)["fr3_only"]
    print(f'model: {model_config}')
    controller_config = impedance_config["impedance"]
    print(f'controller: {controller_config}')
    mujoco_config = "simulation/config/mujoco_fr3_scene.yaml"
    mujoco_config = get_cfg(mujoco_config)["mujoco"]
    print(f'mujoco: {mujoco_config}')
    fr3_config = "hardware/fr3/config/fr3_cfg.yaml"
    fr3_config = get_cfg(fr3_config)["fr3"]
    
    model = RobotModel(model_config)
    impedance_controller = ImpedanceController(controller_config, model)
    mujoco = MujocoSim(mujoco_config)
    fr3 = Fr3Arm(fr3_config)
    
    target_site = "target_site"
    # target_mocap = target_site.split('_')[0]
    tcp_site = "TCP_site"
    tcp_mocap = tcp_site.split('_')[0]
    target = {}
    while True:
        # get mocap target
        target_value = mujoco.get_site_pose(target_site, "xyzw")
        target[model.ee_link] = target_value
        cur_tcp = mujoco.get_tcp_pose()
        joint_states = fr3.get_joint_states()
        mujoco.set_target_mocap_pose(tcp_mocap, cur_tcp)
        # success, joint_value, mode = impedance_controller.compute_controller(target, 
        #                                                                      joint_states)
        joint_value = model.id(joint_states._positions, joint_states._velocities, joint_states._accelerations)
        mode = "torque"
        mujoco.set_joint_command([mode] * len(joint_value), joint_value)
        # print(f'joint value: {joint_value}')
        fr3.set_joint_command(mode, joint_value)
        time.sleep(0.001)
        

if __name__ == "__main__":
    main()
    