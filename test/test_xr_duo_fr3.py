from teleop.XR.quest3.meta_quest3 import MetaQuest3
from hardware.sensors.cameras.realsense_camera import RealsenseCamera
from multiprocessing import shared_memory
from cfg_handling import get_cfg
import numpy as np
import time, cv2
from controller.duo_controller import DuoController
from simulation.mujoco.mujoco_sim import MujocoSim
from motion.pin_model import RobotModel
from hardware.base.utils import negate_pose, transform_pose, convert_homo_2_7D_pose, convert_7D_2_homo, transform_quat, pose_diff
from hardware.base.img_utils import combine_image
from motion.duo_model import DuoRobotModel
import glog as log


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
    # frame transformation consts
    # -0.17710499  0.96118575 -0.29617468

    model_config = "motion/config/robot_model_cfg_duo_fr3.yaml"
    model_cfg = get_cfg(model_config)["duo_fr3_only"]
    model = DuoRobotModel(model_cfg)
    controller_config = "controller/config/duo_ik_cfg.yaml"
    controller_config = get_cfg(controller_config)["duo_controller"]
    print(f'controller: {controller_config}')
    duo_ik = DuoController(controller_config, model)
    
    # mujoco 
    mujoco_cfg = "simulation/config/mujoco_duo_fr3.yaml"
    mujoco_cfg = get_cfg(mujoco_cfg)["mujoco"]
    mujoco = MujocoSim(mujoco_cfg)
    world2base, base2world = get_sim_base_world_transform(mujoco)
    target_site = ["targetL", "targetR"]
    tcp_mocaps = ["TCPL", "TCPR"]

    last_time = time.time()
    frames = model.get_model_end_links()
    total_dof = model.get_model_dof()
    total_dof = total_dof[0] + total_dof[1]
    inteface_mode = "absolute_delta" # absolute_delta
    enabled = False; init_ee_pose = []
    while True:
        data = quest3.parse_data_2_robot_target(inteface_mode)
        print(f'suc: {data[0]}')
        if not data[0] and enabled: enabled = False
        
        index = ['left', 'right']
        if data[0]:
            # print(f'pose: {data[1]}')
            # print(f'tool: {data[2]}')
            arm_target = data[1]
            tool_target = data[2]
            
            joint_position = mujoco.get_joint_states()._positions
            targets = []
            for i, (key, value) in enumerate(arm_target.items()):
                # rotation transform and position translation
                if inteface_mode == "absolute_delta" and enabled:
                    # log.info(f'Total init pose: {init_ee_pose}')
                    cur_init_pose = init_ee_pose[i]
                    value = transform_pose(cur_init_pose, value, False)
                    log.info(f'init pose: {cur_init_pose},\n diff pose: {value} \n target: {value}')
                    
                target_mocap = target_site[i]
                target_mocap = target_mocap.split('_')[0]
                if inteface_mode == "absolute_delta":
                    world2target = transform_pose(world2base[i], value)
                else: world2target = value
                mujoco.set_target_mocap_pose(target_mocap, world2target)
                # debug_mocp = tcp_mocaps[i]
                # debug_pose = transform_pose(world2base[i], data[3][i])
                # mujoco.set_target_mocap_pose(debug_mocp, debug_pose)
                
                # print(f'{i}th pose: {value}')
            
                if (enabled and inteface_mode == "absolute_delta") or inteface_mode == "absolute":
                    frame_name = frames[i]
                    targets.append({frame_name: value})
                
                if tool_target[key][-1] and not enabled and inteface_mode == "absolute_delta":
                    if i == 0: init_ee_pose = []
                    cur_ee_pose = model.get_frame_pose(frames[i], joint_position, True, key)
                    cur_ee_pose = convert_homo_2_7D_pose(cur_ee_pose)
                    init_ee_pose.append(cur_ee_pose) 
                    if i == 1:
                        log.info(f'Curr ee pose is set to init pose. total init pose: {init_ee_pose}')
                        enabled = True
                    targets = None
                # check wheher need to do that
            
            # control 
            if not targets is None:
                # print(f'target: {targets}')
                joint_states =  mujoco.get_joint_states()
                success, joint_target, mode = duo_ik.compute_controller(targets, joint_states)
            else: success = False
            
            if success:
                mode = mode[0]
                # print(f'success: {success}, joint target: {joint_target} mode: {[mode]*total_dof}')
                mujoco.set_joint_command([mode]*total_dof, joint_target)
        
        images = mujoco.get_all_camera_images()
        cur_img = images[0]['img']
        if len(images) > 1:
            for img in images[1:]:
                cur_img = combine_image(cur_img, img['img'])
        cur_img = cv2.resize(cur_img, (tv_img_shape[1], tv_img_shape[0]))
        np.copyto(img_array, np.array(cur_img))
        
        cv2.imshow("color", cur_img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        
        dt = time.time() - last_time
        if dt < (1.0 / 50):
            sleep_time = (1.0 / 50) - dt
            time.sleep(sleep_time)
        last_time = time.time()
        
if __name__ == "__main__":
    main()
    