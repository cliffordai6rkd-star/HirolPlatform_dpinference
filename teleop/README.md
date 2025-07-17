# 机器人遥操
## 完成config文件的构成, 请参考[teleoperation_cfg][default]
- `use_trajectory_planner`: 是否启用planner
- `use_simulation_target`: 是否使用mujoco的方块作为目标设置, 如果为false则根据遥操设备来进行遥操


## 使用示例
- 可以参考[main_test]()文件中的main entry 
- 脚本运行方法
    1. `cd teleop`切换到teleop的文件夹
    2. `python teleoperation.py`这样会运行default的[config文件][default], 如果想指定自定义的config文件可以运行`python teleoperation.py -c <文件相对路径(相对工程路径)>`
- 更多使用案例
    - FR3 3D mouse impedance controller： `python teleoperation.py -c teleop/config/franka_3d_mouse_impedance.yaml`
    - Dual Fr3 with dual 3d mouse ik controller: ``

## 可能存在的bug (待使用者的反馈)
- 

[default]:(./config/franka_3d_mouse.yaml)