# Simulation
## 接口说明
1. `__init__`: config 文件传入
2. `sim_therad`: 仿真实时线程更新机器人状态以及执行关节端命令
3. `get_joint_states`: 获取仿真中机器人系统所有的关节状态
4. `get_tcp_pose`: 获取机器人的末端7维位姿
5. `set_joint_command`: 设置机器人系统的关节目标
6. `parse_config`: 解析config文件
7. `render`: 仿真额外的debug渲染
8. `update_trajectory_data`: 在仿真环境中可视化轨迹数据

## 参数说明
- Mujoco 参数， 请参考[mujoco_cfg](./config/mujoco_fr3_cfg.yaml)
  1. `base_xml_file`: entire scene xml file
  2. `robot_xml`: robot xml file
  3. `quat_sequence`: 四元数的顺序
  4. `dt`: 仿真线程频率
  5. `joint_names`: 需控制的主动关节名字
  6. `actuator_names`": 需要控制仿真的关节的驱动器名字
  7. `actuator_mode`: 每个驱动器控制的模式
  8. `ee_site_name`: 末端的名字
  9. `extra_render`: 可视化以及debug的渲染
  10. `max_traj_len`: 可视化轨迹的圆球的数量
  11. `use_custom_key_frame`: 机器人所有关节初始关节角度 （如果没有在xml进行自定义设置的话将该项射程None）
  12. `sensor_dict`: 各种传感器的信息 (名字和数据维度信息)


## 使用示例
- 可以参考[main_test]()文件中的main entry 
- 

## 可能存在的bug (待使用者的反馈)
- 需支持传感器的读入
- 需支持多机器人加载以及架构
