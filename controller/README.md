# Controller development (末端到关节端的控制器)

## 接口说明
1. `__init__`创建子类的实现： 需要cfg文件以及pinocchio kinematics model的实例
2. `compute_controller` 通过机器人的joint state信息以及末端位姿目标来计算关节目标，会返回是否成功，关节目标，以及关节控制模式

## 具体实现子类的参数说明
- Controller 参数
  1. IK controller: 请参考[ik](./ik_readme.md)
  2. Impedance controller: 请参考 [impedance](./impedance_readme.md)
  3. whole body ik: 待集成

- Pinocchio kinematics 参数: (创建实例的[cfg_file](./config/ik_fr3_cfg.yaml)中的model下)
  1. `urdf_path`: urdf的相对路径
  2. `base_link`: base link的名字
  3. `end_effector_link`: end effector link的名字

## 使用示例
- 可以参考[teleop](../teleop/teleoperation.py)文件中的main entry以及robot system实例创建的过程 

## 可能存在的bug (待使用者的反馈)
- 需要更改新版pinocchio model的兼容
- 接入WBIK的方法
