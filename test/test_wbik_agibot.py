from controller.whole_body_ik import WholeBodyIk
from motion.pin_model import RobotModel
from simulation.mujoco.mujoco_sim import MujocoSim
import os, time
from cfg_handling import get_cfg
from hardware.base.utils import convert_homo_2_7D_pose

def main():
    model_config = "motion/config/robot_model_cfg_agibot_g1.yaml"
    model_config = get_cfg(model_config)
    print(f'model: {model_config}')
    controller_config = "controller/config/whole_body_ik_agibot_g1.yaml"
    controller_config = get_cfg(controller_config)["whole_body_ik"]
    print(f'controller: {controller_config}')
    mujoco_config = "simulation/config/mujoco_agibot_g1.yaml"
    mujoco_config = get_cfg(mujoco_config)["mujoco"]
    print(f'mujoco: {mujoco_config}')
    
    model = RobotModel(model_config)
    controller = WholeBodyIk(controller_config, model)
    mujoco = MujocoSim(mujoco_config)
    
    mocap_names = ["targetL", "targetR"]
    tcp_names = ["TCPL", "TCPR"]
    target = {}
    while True:
        for i, mocap in enumerate(mocap_names):
            name = controller.tracking_frames[i]["name"]
            pose_7d = mujoco.get_site_pose(mocap + "_site", quat_seq="xyzw")
            pose_7d[2] -= (1.0 - 0.6485) 
            target[name] = pose_7d 
        print(f'target: {target}')
        joint_states = mujoco.get_joint_states()
        model.update_kinematics(joint_states._positions)
        for i, tcp_mocap in enumerate(tcp_names):
            # cur pose
            name = controller.tracking_frames[i]["name"]
            pose_cur = model.get_frame_pose(name, joint_states._positions,
                                                    need_update=False)
            pose_cur = convert_homo_2_7D_pose(pose_cur)
            mujoco.set_target_mocap_pose(tcp_mocap, pose_cur)
            print(f'pose_cur for {name}: {pose_cur}')

        start = time.time()
        success, command, mode = controller.compute_controller(target, joint_states)
        print(f'used time for ik: {time.time() - start}')
        print(f'joint command: {command}, mode: {mode}')
        if success:
            mujoco.set_joint_command([mode]*model.nv, command)
        time.sleep(0.001)

if __name__ == '__main__':
    main()
    