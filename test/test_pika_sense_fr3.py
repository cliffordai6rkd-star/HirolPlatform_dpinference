from teleop.pika_tracker.pika_tracker import PikaTracker
from controller.controller_base import IKController
from simulation.mujoco.mujoco_sim import MujocoSim
from hardware.base.utils import negate_pose, transform_pose, convert_homo_2_7D_pose
from motion.pin_model import RobotModel
import glog as log
from cfg_handling import get_cfg
import numpy as np
import time

def get_sim_base_world_transform(sim):
    world2base_pose = [np.array([0, 0, 0, 0, 0, 0, 1])]
    base2world_pose = [negate_pose(world2base_pose[0])]
    if len(sim.base_body_name) != 0:
        world2base_pose = []
        base2world_pose = []
        for cur_base_body in sim.base_body_name:
            cur_world2base = sim.get_body_pose(cur_base_body)
            world2base_pose.append(cur_world2base)
            base2world_pose.append(negate_pose(cur_world2base))
    return world2base_pose, base2world_pose

def main():
    tracker_cfg = "teleop/pika_tracker/config/left_tracker_fr3_cfg.yaml"
    tracker_cfg = get_cfg(tracker_cfg)["pika_tracker"]
    tracker = PikaTracker(tracker_cfg)
    
    model_config = "motion/config/robot_model_fr3_cfg.yaml"
    model_cfg = get_cfg(model_config)["fr3_only"]
    model = RobotModel(model_cfg)
    controller_config = "controller/config/ik_fr3_cfg.yaml"
    controller_config = get_cfg(controller_config)["ik"]
    print(f'controller: {controller_config}')
    duo_ik = IKController(controller_config, model)
    
    # mujoco 
    mujoco_cfg = "simulation/config/mujoco_fr3_cfg.yaml"
    mujoco_cfg = get_cfg(mujoco_cfg)["mujoco"]
    mujoco = MujocoSim(mujoco_cfg)
    world2base, base2world = get_sim_base_world_transform(mujoco)
    log.info(f'world2bvase: {world2base}')
    target_site = ["targetL", "target"]
    tcp_mocaps = ["TCPL", "TCPR"]

    last_time = time.time()
    frames = model.get_model_end_links()
    print(f'frames: {frames}')
    total_dof = model.get_model_dof()
    total_dof = total_dof
    inteface_mode = "absolute_delta" # absolute_delta
    enabled = False; init_ee_pose = []
    while True:
        data = tracker.parse_data_2_robot_target(inteface_mode)
        print(f'suc: {data[0]}')
        if not data[0] and enabled: enabled = False
        
        if data[0]:
            # print(f'pose: {data[1]}')
            # print(f'tool: {data[2]}')
            arm_target = data[1]
            tool_target = data[2]
            
            joint_position = mujoco.get_joint_states()._positions
            targets = []
            
            # rotation transform and position translation
            value = arm_target["single"]
            if inteface_mode == "absolute_delta" and enabled:
                # log.info(f'Total init pose: {init_ee_pose}')
                cur_init_pose = init_ee_pose
                value = transform_pose(cur_init_pose, value, False)
                log.info(f'init pose: {cur_init_pose},\n diff pose: {value} \n target: {value}')
                
            target_mocap = target_site[1]
            target_mocap = target_mocap.split('_')[0]
            if inteface_mode == "absolute_delta":
                world2target = transform_pose(world2base[0], value)
            else: world2target = value
            mujoco.set_target_mocap_pose(target_mocap, world2target)
                
            # debug_mocp = tcp_mocaps[i]
            # debug_pose = transform_pose(world2base[i], data[3][i])
            # mujoco.set_target_mocap_pose(debug_mocp, debug_pose)
            
            # print(f'{i}th pose: {value}')
        
            if (enabled and inteface_mode == "absolute_delta") or inteface_mode == "absolute":
                frame_name = frames
                targets.append({frame_name: value})
            
            if tool_target["single"][-1] and not enabled and inteface_mode == "absolute_delta":
                cur_ee_pose = model.get_frame_pose(frames, joint_position, True, "single")
                cur_ee_pose = convert_homo_2_7D_pose(cur_ee_pose)
                init_ee_pose = cur_ee_pose
                enabled = True
                targets = None
            # check wheher need to do that
            
            # control 
            if targets:
                # print(f'target: {targets}')
                joint_states =  mujoco.get_joint_states()
                success, joint_target, mode = duo_ik.compute_controller(targets, joint_states)
            else: success = False
            
            if success:
                # print(f'success: {success}, joint target: {joint_target} mode: {[mode]*total_dof}')
                mujoco.set_joint_command([mode]*total_dof, joint_target)
        
        dt = time.time() - last_time
        if dt < (1.0 / 50):
            sleep_time = (1.0 / 50) - dt
            time.sleep(sleep_time)
        last_time = time.time()
        
if __name__ == "__main__":
    main()
    