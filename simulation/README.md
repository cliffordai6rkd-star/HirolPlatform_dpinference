```
conda create -n mujoco python=3.10
conda activate mujoco
python simulation/mujoco_env_creator.py
python simulation/mujoco_env_creator.py --config simulation/scene_config/unitree_g1_demo.yaml
python simulation/mujoco_env_creator.py --config simulation/scene_config/fr3_demo.yaml
```
or in mac M(x) env
```
micromamba create -n mujoco python=3.10
micromamba activate mujoco
mjpython simulation/mujoco_env_creator.py
mjpython simulation/mujoco_env_creator.py --config simulation/scene_config/unitree_g1_demo.yaml
mjpython simulation/mujoco_env_creator.py --config simulation/scene_config/fr3_demo.yaml
```

