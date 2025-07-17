# Trajectory generation

## 接口说明
1. `__init__`实例创建： `config`: trajectory的config文件可以参考[cartesian_trajectory _cfg][cfg_file]; `buffer`: 用来储存生成的点的Buffer（该buffer的创建也可在[cartesian_trajectory][cfg_file]查看; `lock`: 通过`threading.lock()`方式进行创建并当作参数传入
2. `plan_trajectory`: 具体生成轨迹的函数实现， 请注意这个函数是个长时间blocking的，但是在生成每个点并加进buffer的时候会有sleep的时间按。 所以建议使用的时候给这个函数创建一个独立线程来使用， 需要取点的时候可以从buffer里面把生成的点给pop出来, 具体实现为位置的五次项插值，旋转上的slerp插值。

## 参数说明
- Trajectory 参数
  1. interpolation_type:7维位姿平移向量的插值方式 (**现只支持“quintic”的五次项方式**)
  2. dt： 每个生成的点之间的步长，以及blocking的plan函数睡眠时间
  3. max_velocity: 用来限制两个位姿之间的位置
- Buffer 参数
    1. size： buffer size（如果之前生成的轨迹已经把buffer占满，新的轨迹加入buffer前会把队头的旧轨迹点扔掉）
    2. 

## 使用示例
- 可以参考[main_test](./cartesian_trajectory.py)文件中的main entry (不包含线程使用示例)
- 线程使用示例请参考[teleop](../teleop/teleoperation.py)的`trajectory_task`线程函数

## 可能存在的bug (待使用者的反馈)
- 




[cfg_file]:config/cartesian_polynomial_traj_cfg.yaml