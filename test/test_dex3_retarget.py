from hardware.unitreeG1.Dex3_Hand import Dex3Hand, minLimits_right, maxLimits_right
from multiprocessing import shared_memory
from teleop.XR.quest3.meta_quest3 import MetaQuest3
from simulation.mujoco.mujoco_sim import MujocoSim
from motion.pin_model import RobotModel
from controller.whole_body_ik import WholeBodyIk
from cfg_handling import get_cfg
from hardware.base.utils import dynamic_load_yaml
import numpy as np
from hardware.base.utils import negate_pose, transform_pose, convert_homo_2_7D_pose, convert_7D_2_homo, transform_quat, pose_diff
import time, copy
import glog as log

def get_sim_base_world_transform(sim: MujocoSim):
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

def main0():
    # mujoco 
    mujoco_cfg = "simulation/config/mujoco_right_dex3_hand.yaml"
    mujoco_cfg = get_cfg(mujoco_cfg)["mujoco"]
    mujoco = MujocoSim(mujoco_cfg)
    target_mocap = ["rt", "rm", "ri"]
    tcp_mocaps = ["TCPL", "TCPR"]
    world2base_pose, base2world_pose = get_sim_base_world_transform(mujoco)
    print(f'base2world: {base2world_pose}')
    
    # model and whole body controller
    model_cfg = "motion/config/robot_model_cfg_right_dex3.yaml"
    model_cfg = get_cfg(model_cfg)
    model = RobotModel(model_cfg)
    controller_cfg = "controller/config/whole_body_ik_right_dex3.yaml"
    controller_cfg = get_cfg(controller_cfg)["whole_body_ik"]
    wbik = WholeBodyIk(controller_cfg, model)
    
    while True:
        target = []
        for i, mocap in enumerate(target_mocap):
            name = wbik.tracking_frames[i]["name"]
            pose_7d = mujoco.get_site_pose(mocap + "_site", quat_seq="xyzw")
            pose_7d = transform_pose(base2world_pose[0], pose_7d)
            # pose_7d[2] -= (1.0 - 0.6485) 
            cur_target = {name:pose_7d}
            target.append(cur_target)
        # print(f'target: {target}')
        joint_states = mujoco.get_joint_states()
        # print(f'joint state: {joint_states._positions}')
        
        # check forward kinematics
        model.update_kinematics(joint_states._positions)
        for i, tcp_mocap in enumerate(target_mocap):
            # cur pose
            name = wbik.tracking_frames[i]["name"]
            # print(f'{i}th name: {name}')
            pose_cur = model.get_frame_pose(name, joint_states._positions,
                                                    need_update=False)
            pose_cur = convert_homo_2_7D_pose(pose_cur)
            pose_cur[2] += 0.3
            # pose_cur = transform_pose(world2base_pose[0], pose_cur)
            mujoco.set_target_mocap_pose(tcp_mocap, pose_cur)
            # print(f'pose_cur for {name}: {pose_cur}')

        # start = time.time()
        # success, command, mode = wbik.compute_controller(target, joint_states)
        # print(f'used time for ik: {time.time() - start}')
        # print(f'joint command: {command}, mode: {mode}')
        # if success:
        #     mujoco.set_joint_command([mode]*model.nv, command)
        # else:
        #     raise ValueError
        time.sleep(0.01)

def hardware_testing():
    # mujoco 
    mujoco_cfg = "simulation/config/mujoco_right_dex3_hand.yaml"
    mujoco_cfg = get_cfg(mujoco_cfg)["mujoco"]
    mujoco = MujocoSim(mujoco_cfg)
    target_mocap = ["rh", "rm", "ri"]
    tcp_mocaps = ["TCPL", "TCPR"]
    world2base_pose, base2world_pose = get_sim_base_world_transform(mujoco)
    print(f'base2world: {base2world_pose}')
    
    # model and whole body controller
    # model_cfg = "motion/config/robot_model_cfg_right_dex3.yaml"
    # model_cfg = get_cfg(model_cfg)
    # model = RobotModel(model_cfg)
    # controller_cfg = "controller/config/whole_body_ik_right_dex3.yaml"
    # controller_cfg = get_cfg(controller_cfg)["whole_body_ik"]
    # wbik = WholeBodyIk(controller_cfg, model)
    
    # dex3 
    dex3_config = "hardware/unitreeG1/config/right_dex3_hand.yaml"
    dex3_config = dynamic_load_yaml(dex3_config)["dex3_hand"]
    dex3 = Dex3Hand(dex3_config)
    dex3.set_hardware_command(minLimits_right)
    time.sleep(2)
    
    test_dof = 0
    if test_dof < 0: test_dof = 0
    if test_dof > dex3._MOTOR_MAX: test_dof = dex3._MOTOR_MAX - 1
    print(f'test dof: {test_dof}')
    posi = copy.deepcopy(minLimits_right)
    # posi[1:2] = [maxLimits_right[1], maxLimits_right[2]]
    counter = 0
    while True:
        if counter % 1000 == 0:
            if np.isclose(posi[test_dof], minLimits_right[test_dof]):
                posi[test_dof] = maxLimits_right[test_dof]
            else: 
                posi[test_dof] = minLimits_right[test_dof]
            time.sleep(0.005)
            print(f'posi: {posi[test_dof]}, min: {minLimits_right[test_dof]}')
            # dex3.set_hardware_command(posi)
        log.info(f'posi: {posi[test_dof]}, min: {minLimits_right[test_dof]}')
        # mujoco sync
        hand_state = dex3.get_tool_state()._position
        nv = len(hand_state)
        mujoco.set_joint_command(['position']*nv, hand_state)
        
        counter += 1
        time.sleep(0.001)

# check hardware and test retargeting
def hardware_retargeting():
    # mujoco 
    mujoco_cfg = "simulation/config/mujoco_right_dex3_hand.yaml"
    mujoco_cfg = get_cfg(mujoco_cfg)["mujoco"]
    mujoco = MujocoSim(mujoco_cfg)
    target_mocap = ["rh", "rm", "ri"]
    tcp_mocaps = ["TCPL", "TCPR"]
    world2base_pose, base2world_pose = get_sim_base_world_transform(mujoco)
    print(f'base2world: {base2world_pose}')
    
    # model and whole body controller
    model_cfg = "motion/config/robot_model_cfg_right_dex3.yaml"
    model_cfg = get_cfg(model_cfg)
    model = RobotModel(model_cfg)
    controller_cfg = "controller/config/whole_body_ik_right_dex3.yaml"
    controller_cfg = get_cfg(controller_cfg)["whole_body_ik"]
    wbik = WholeBodyIk(controller_cfg, model)
    
    while True:
        target = []
        for i, mocap in enumerate(target_mocap):
            name = wbik.tracking_frames[i]["name"]
            pose_7d = mujoco.get_site_pose(mocap + "_site", quat_seq="xyzw")
            pose_7d = transform_pose(base2world_pose[0], pose_7d)
            # pose_7d[2] -= (1.0 - 0.6485) 
            cur_target = {name:pose_7d}
            target.append(cur_target)
        print(f'target: {target}')
        joint_states = mujoco.get_joint_states()
        print(f'joint state: {joint_states._positions}')
        # model.update_kinematics(joint_states._positions)
        # for i, tcp_mocap in enumerate(target_mocap):
        #     # cur pose
        #     name = wbik.tracking_frames[i]["name"]
        #     print(f'{i}th name: {name}')
        #     pose_cur = model.get_frame_pose(name, joint_states._positions,
        #                                             need_update=False)
        #     pose_cur = convert_homo_2_7D_pose(pose_cur)
        #     pose_cur = transform_pose(world2base_pose[0], pose_cur)
        #     mujoco.set_target_mocap_pose(tcp_mocap, pose_cur)
        #     print(f'pose_cur for {name}: {pose_cur}')

        start = time.time()
        success, command, mode = wbik.compute_controller(target, joint_states)
        print(f'used time for ik: {time.time() - start}')
        print(f'joint command: {command}, mode: {mode}')
        # if success:
        #     mujoco.set_joint_command([mode]*model.nv, command)
        # else:
        #     raise ValueError
        time.sleep(0.01)

def main():
    dex3_config = "hardware/unitreeG1/config/right_dex3_hand.yaml"
    dex3_config = get_cfg(dex3_config)["dex3_hand"]
    dex3 = Dex3Hand(dex3_config)
    
    xr_config = "teleop/XR/quest3/config/meta_quest3_duo_fr3.yaml"
    xr_config = get_cfg(xr_config)["meta_quest3"]
    
    # cfg 补充
    ASPECT_RATIO_THRESHOLD = 2.0 # If the aspect ratio exceeds this value, it is considered binocular
    if (xr_config['image_shape'][1] / xr_config['image_shape'][0] > ASPECT_RATIO_THRESHOLD):
        BINOCULAR = True
    else:
        BINOCULAR = False
    xr_config['binocular'] = BINOCULAR
        
    if BINOCULAR and not (xr_config['image_shape'][1] / xr_config['image_shape'][0] > ASPECT_RATIO_THRESHOLD):
        tv_img_shape = (xr_config['image_shape'][0], xr_config['image_shape'][1] * 2, 3)
    else:
        tv_img_shape = (xr_config['image_shape'][0], xr_config['image_shape'][1], 3)
    xr_config['image_shape'] = tv_img_shape
    
    img_shm = shared_memory.SharedMemory(create=True, 
                    size = np.prod(xr_config['image_shape']) * np.uint8().itemsize)
    xr_config["img_shm_name"] = img_shm.name
    img_array = np.ndarray(tv_img_shape, dtype=np.uint8, buffer=img_shm.buf)
    quest3 = MetaQuest3(xr_config)
    
    # mujoco 
    mujoco_cfg = "simulation/config/mujoco_duo_fr3.yaml"
    mujoco_cfg = get_cfg(mujoco_cfg)["mujoco"]
    mujoco = MujocoSim(mujoco_cfg)
    target_site = ["rt", "rm", "ri"]
    tcp_mocaps = ["TCPL", "TCPR"]
    
    while True:
        data = quest3.parse_data_2_robot_target("absolute")
        
        if data[0]:
            hand_target = data[2]
            
    
    
if __name__ == "__main__":
    # main0()
    hardware_testing()
    
