from teleop.xr.meta_quest3 import MetaQuest3
from hardware.sensors.cameras.realsense_camera import RealsenseCamera
from multiprocessing import shared_memory
from cfg_handling import get_cfg
import numpy as np
import time, cv2
from simulation.mujoco.mujoco_sim import MujocoSim
from motion.pin_model import RobotModel
from controller.whole_body_ik import WholeBodyIk

def main():
    xr_config = "teleop/xr/meta_quest3_cfg.yaml"
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
    
    camera_config = "hardware/sensors/cameras/config/d435i_cfg.yaml"
    d435_cfg = get_cfg(camera_config)["D435i"]
    d435 = RealsenseCamera(d435_cfg) 
    
    model_config = "motion/config/robot_model_cfg_agibot_g1.yaml"
    model_config = get_cfg(model_config)
    print(f'model: {model_config}')
    model = RobotModel(model_config)
    # controller_config = "controller/config/whole_body_ik_agibot_g1.yaml"
    # controller_config = get_cfg(controller_config)["whole_body_ik"]
    # print(f'controller: {controller_config}')
    # wbik = WholeBodyIk(controller_config, model)
    
    # mujoco 
    mujoco_cfg = "simulation/config/mujoco_agibot_g1.yaml"
    mujoco_cfg = get_cfg(mujoco_cfg)["mujoco"]
    mujoco = MujocoSim(mujoco_cfg)
    target_site = ["targetL", "targetR"]
    tcp_mocaps = ["TCPL", "TCPR"]

    
    last_time = time.time()
    while True:
        data = quest3.parse_data_2_robot_target("absolute")
        # print(f'suc: {data[0]}, pose: {data[1]}')
        
        arm_target = data[1]
        for i, (key, value) in enumerate(arm_target.items()):
            target_mocap = tcp_mocaps[i]
            target_mocap = target_mocap.split('_')[0]
            mujoco.set_target_mocap_pose(target_mocap, value)
        
        cur_img = d435.capture_all_data()['image']
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
    