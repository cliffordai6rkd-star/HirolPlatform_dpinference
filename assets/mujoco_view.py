import mujoco
import mujoco.viewer
import numpy as np
import time
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
print(f'cur dir: {cur_dir}')
# franka emika panda hand is the only hand xml
# franka fr3.xml is the only fr3 robot
# only hand xml: franka_hand.xml
model_path = os.path.join(cur_dir, 'franka_fr3', 'fr3.urdf')
print(f'model path: {model_path}')
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
model.opt.timestep = 0.001
    
with mujoco.viewer.launch_passive(model, data, show_right_ui=True) as viewer:
    while True:
        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync() 
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        