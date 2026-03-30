# DP / FR3 推理链路梳理

这份文档的目标不是解释算法，而是回答两个工程问题：

1. 如果想搞明白 `dp_inference.py` 的推理链路最终是怎么把命令发给 FR3 的，需要先看清楚哪些底层脚本、哪些函数。
2. 如果想自己写一个新的 inference 代码，应该从哪些脚本作为 base。


## 一句话先说清楚

当前 `factory/tasks/inferences_tasks/dp/config/fr3_dp_ddim_inference_cfg.yaml` 默认配置里：

- `action_type: "joint_position"`
- 所以 **policy 输出默认是 joint command**
- `ee_state` 在这条默认链路里主要是 **观测的一部分**
- 最终发给 FR3 的底层命令仍然是 `Fr3Arm.set_joint_command(...)`

也就是说，默认 DP 这条链路不是“policy 直接输出 ee pose 发给 FR3”，而是“读观测 -> policy 预测 joint action -> 下发 joint command”。


## 一、想搞明白命令如何发给 FR3，优先看这 7 层

建议按下面顺序看，不要跳着看。

### 1. 任务入口层：`dp_inference.py`

文件：

- `factory/tasks/inferences_tasks/dp/dp_inference.py`

先看这些函数：

- `DP_Inferencer.__init__`
- `DP_Inferencer._load_dp_model`
- `DP_Inferencer.start_inference`
- `DP_Inferencer.convert_from_gym_obs`
- `DP_Inferencer._convert_gym_obs_to_dp_format`

你要看明白的点：

- ckpt 是如何被 `torch.load(...)` 读出来的
- workspace / policy 是怎么从 payload 恢复的
- policy 的 `predict_action(...)` 在哪里被调用
- policy 输出的 `result["action"]` 是怎么变成 `action_np`
- `action_np` 是怎么交给父类执行的

关键事实：

- 真正往下执行动作的是 `start_inference()` 里这一句：
  - `self.convert_to_gym_action(action_np)`
- 这个函数不在 `dp_inference.py`，而在它的父类 `InferenceBase`


### 2. 推理公共执行层：`inference_base.py`

文件：

- `factory/tasks/inferences_tasks/inference_base.py`

重点函数：

- `InferenceBase.convert_from_gym_obs`
- `InferenceBase.convert_to_gym_action`
- `InferenceBase.convert_to_gym_action_single_step`

你要看明白的点：

- policy 输出的 action chunk 如何按时间步逐个执行
- 一条 action 如何被拆成：
  - `action["arm"]`
  - `action["tool"]`
- `action_type` 不同，切片逻辑为什么不同
- 为什么最终统一走到：
  - `self._gym_robot.step(action)`

这一层的作用可以理解成：

- 上面接 policy 输出
- 下面接 GymApi
- 它负责把“模型张量/数组动作”翻译成“机器人执行动作字典”


### 3. Gym 适配层：`gym_interface.py`

文件：

- `factory/components/gym_interface.py`

这是最关键的一层，必须看清楚。

重点函数：

- `GymApi.step`
- `GymApi.get_observation`
- `GymApi.get_joint_state`
- `GymApi.get_ee_state`
- `GymApi.set_joint_position`
- `GymApi.set_ee_pose`

你要看明白的点：

- 观测是如何从机器人系统里读出来的
- `ee_state` 是如何通过 `get_ee_state()` 得到的
- `joint_position` 动作和 `end_effector_pose` 动作在这里如何分叉

最重要的两条分支：

1. 如果 `action_type` 是 `joint_position`
   - `GymApi.step()` -> `set_joint_position(...)`
2. 如果 `action_type` 是 `end_effector_pose`
   - `GymApi.step()` -> `set_ee_pose(...)`

所以这一层决定了：

- 你的 policy 输出究竟被当成 joint 命令
- 还是被当成 ee pose target


### 4. 运动系统层：`motion_factory.py`

文件：

- `factory/components/motion_factory.py`

重点函数：

- `MotionFactory.create_motion_components`
- `MotionFactory._controller_task`
- `MotionFactory._traj_task`
- `MotionFactory.set_joint_positions`
- `MotionFactory.update_high_level_command`
- `MotionFactory._get_controller_target`
- `MotionFactory.get_frame_pose`

这是理解“ee pose 怎么变 joint command”的核心层。

你要分两条路径看：

#### 路径 A：joint action 路径

- `GymApi.set_joint_position(...)`
- -> `MotionFactory.set_joint_positions(...)`
- -> `RobotFactory.set_joint_commands(...)`

这条路径没有经过 controller 做 IK。

#### 路径 B：ee pose 路径

- `GymApi.set_ee_pose(...)`
- -> `MotionFactory.update_high_level_command(...)`
- -> `_controller_task()` 后台线程取出 `_high_level_command`
- -> `self._controller.compute_controller(...)`
- -> 得到 `joint_target`
- -> `RobotFactory.set_joint_commands(...)`

也就是说：

- `update_high_level_command(...)` 只是“更新高层末端目标”
- 真正把 ee pose target 消费掉的是 `_controller_task()`


### 5. 控制器层：`controller_base.py` 及具体 controller

最少先看：

- `controller/controller_base.py`

重点函数：

- `ControllerBase.compute_controller`
- `IKController.compute_controller`

如果 motion config 用的是 IK controller，这里就是 ee target -> joint target 的那一步。

当前 DP 默认 include 的 motion config 是：

- `factory/components/motion_configs/left_fr3_with_pika_ati_ik.yaml`

里面关键配置是：

- `controller_type: "ik"`

所以默认 ee pose 分支下，真正工作的 controller 是：

- `IKController.compute_controller(...)`

它做的事是：

- 把 7D pose 转成齐次变换
- 调 IK 求解器
- 返回 `joint_target` 和 `mode`

如果你以后换成阻抗控制器，还要额外看：

- `controller/cartesian_impedance_controller.py`


### 6. 机器人系统分发层：`robot_factory.py`

文件：

- `factory/components/robot_factory.py`

重点函数：

- `RobotFactory.create_robot_system`
- `RobotFactory.get_joint_states`
- `RobotFactory.set_joint_commands`
- `RobotFactory.set_robot_joint_command`
- `RobotFactory.set_tool_command`
- `RobotFactory.enable_async_control`
- `RobotFactory._async_command_loop`

这一层非常关键，因为它是真正的“总分发器”。

你要看明白的点：

- 什么时候命令是同步直接发
- 什么时候先喂给 smoother，再由异步线程高频发
- arm 命令和 gripper 命令在这里是怎么分开的

尤其注意：

- 如果开了 `use_smoother: true`
- 并且开了 `auto_enable_async_control: true`

那么 `set_joint_commands(...)` 很可能不是当场把命令直接发给机械臂，而是：

1. 先更新 smoother target
2. 后台 `_async_command_loop()` 再不断调用
   - `set_robot_joint_command(...)`

所以从工程时序上说，很多情况下“真正发命令”的线程不在 inference 主线程，而在 `RobotFactory` 的异步线程里。


### 7. FR3 硬件层：`fr3_arm.py`

文件：

- `hardware/fr3/fr3_arm.py`

这是最终必须读到的文件。

重点函数：

- `Fr3Arm.initialize`
- `Fr3Arm.update_robot_state_thread`
- `Fr3Arm.set_joint_command`
- `Fr3Arm.set_joint_position`
- `Fr3Arm.set_joint_velocity`
- `Fr3Arm.set_joint_torque`
- `Fr3Arm.recover`

你要看明白的点：

- `RobotFactory.set_robot_joint_command(...)` 最后是怎么落到这个类上的
- FR3 用的控制模式是：
  - `position`
  - `velocity`
  - `torque`
- 真正给 panda / libfranka / panda_py 控制器喂命令的地方在哪里

最终最底层的动作下发点就是：

- `Fr3Arm.set_joint_position(...)`
- `Fr3Arm.set_joint_velocity(...)`
- `Fr3Arm.set_joint_torque(...)`

这几个函数内部最终都会调用：

- `self._panda_py_controller.set_control(...)`

如果你只想找“命令真正离开本项目、进入 FR3 驱动”的最后一个调用点，看这里就够了。


## 二、如果目标是看懂 `ee_state`，你要额外盯住哪些函数

很多人会把“ee_state”和“ee action”混在一起，这里单独拆开。

### 1. `ee_state` 作为观测是怎么来的

先看：

- `GymApi.get_observation`
- `GymApi.get_ee_state`
- `MotionFactory.get_frame_pose`
- `RobotFactory.get_joint_states`

这条链路本质是：

- 先读当前 joint states
- 再通过 robot model 做 FK
- 得到 ee pose
- 再把 ee pose 拼进 observation

注意：

- 默认 FR3 DP 配置是 `action_type: joint_position`
- 所以 `ee_state` 默认主要用于“输入给 policy”
- 不是“直接作为动作输出发给 FR3”


### 2. `ee pose` 作为动作目标时怎么走

先看：

- `GymApi.step`
- `GymApi.set_ee_pose`
- `MotionFactory.update_high_level_command`
- `MotionFactory._controller_task`
- `ControllerBase.compute_controller`
- `IKController.compute_controller`
- `RobotFactory.set_joint_commands`
- `Fr3Arm.set_joint_command`

这一条才是“ee target -> controller -> joint target -> FR3”的链路。


## 三、如果你想自己写一个 inference，建议从哪几个脚本作为 base

这里分成“强烈推荐”和“按需求参考”两组。

### 强烈推荐作为 base 的脚本

#### 1. `InferenceBase`

文件：

- `factory/tasks/inferences_tasks/inference_base.py`

这是最推荐的 base。

原因：

- 已经封装好了观测读取入口
- 已经封装好了 action chunk 执行逻辑
- 已经封装好了 joint / ee / tool 的动作翻译
- 已经接好了 `GymApi`

如果你自己写新的 inference 类，最稳的做法通常是：

- 新建一个类继承 `InferenceBase`
- 重点实现你自己的：
  - `start_inference`
  - `convert_from_gym_obs`
  - `policy_reset`
  - `policy_prediction`
  - `close`


#### 2. `DP_Inferencer`

文件：

- `factory/tasks/inferences_tasks/dp/dp_inference.py`

如果你写的还是 diffusion policy 风格，最直接的 base 就是它。

适合的场景：

- 你的模型输入也是多帧图像 + state
- 你的模型输出也是 action chunk
- 你只需要替换 ckpt 加载、obs 映射、action 后处理

这时通常不需要重写下面几层：

- `GymApi`
- `MotionFactory`
- `RobotFactory`
- `Fr3Arm`


### 按需求参考的脚本

#### 3. `eval_real_robot_example.py`

文件：

- `factory/tasks/inferences_tasks/dp/eval_real_robot_example.py`

这个脚本不是当前项目主链路的 base，但很适合参考。

适合的用途：

- 想看一个“更裸”的 real robot 推理循环
- 想对比当前项目封装前后的动作调度逻辑
- 想看 policy control loop / timing / action horizon 是怎么写的


#### 4. `GymApi`

文件：

- `factory/components/gym_interface.py`

如果你准备写的 inference 不想继承 `InferenceBase`，而是自己直接管 obs/action，那么至少要把 `GymApi` 当作 base API。

最常用接口：

- `get_observation()`
- `step(action)`
- `reset()`
- `set_action_type(...)`


#### 5. `MotionFactory`

文件：

- `factory/components/motion_factory.py`

如果你要自己设计“高层 ee target -> controller -> joint command”的执行层，就必须把它当 base 看。

但一般不建议一开始就直接绕过 `InferenceBase` 和 `GymApi` 去操作这一层，除非你非常明确要自己控制：

- 后台 controller thread
- 轨迹规划 thread
- 高层指令缓存
- async smoother / hardware streaming


## 四、自己写 inference 时，推荐的落地方式

### 最推荐的方式

新建一个和 `DP_Inferencer` 同级的新类：

- 继承 `InferenceBase`
- 保留 `GymApi` / `MotionFactory` / `RobotFactory` / `Fr3Arm` 整条下层链路不动
- 你只改：
  - 模型加载
  - 观测格式转换
  - policy 输出格式转换

这样最稳，因为：

- 下层真机控制已经接好了
- 你不容易把 FR3 的控制线程、smoother、恢复逻辑写坏


### 如果你的模型输出是 joint action

你应该优先参考：

- `factory/tasks/inferences_tasks/dp/dp_inference.py`
- `factory/tasks/inferences_tasks/inference_base.py`
- `factory/components/gym_interface.py`

重点走通：

- `predict_action(...)`
- `convert_to_gym_action(...)`
- `GymApi.step(...)` 的 `joint_position` 分支


### 如果你的模型输出是 ee pose action

你应该优先参考：

- `factory/tasks/inferences_tasks/inference_base.py`
- `factory/components/gym_interface.py`
- `factory/components/motion_factory.py`
- `controller/controller_base.py`
- 当前使用的具体 controller 文件

重点走通：

- `GymApi.step(...)` 的 `END_EFFECTOR_POSE` 分支
- `set_ee_pose(...)`
- `update_high_level_command(...)`
- `_controller_task()`
- `compute_controller(...)`


## 五、最小必读清单

如果你时间不多，只看下面这些文件就够你把 FR3 下发链路串起来：

- `factory/tasks/inferences_tasks/dp/dp_inference.py`
- `factory/tasks/inferences_tasks/inference_base.py`
- `factory/components/gym_interface.py`
- `factory/components/motion_factory.py`
- `controller/controller_base.py`
- `factory/components/robot_factory.py`
- `hardware/fr3/fr3_arm.py`

如果还要看夹爪：

- `hardware/fr3/franka_hand.py`
- 或当前 gripper 对应文件


## 六、最小调用链速记

### 默认 DP 配置下

`dp_inference.py`
-> `InferenceBase.convert_to_gym_action`
-> `GymApi.step`
-> `GymApi.set_joint_position`
-> `MotionFactory.set_joint_positions`
-> `RobotFactory.set_joint_commands`
-> `RobotFactory._async_command_loop`（如果开 smoother + async）
-> `RobotFactory.set_robot_joint_command`
-> `Fr3Arm.set_joint_command`
-> `Fr3Arm.set_joint_position`
-> `panda_py_controller.set_control(...)`


### 如果你改成 ee pose action

`dp_inference.py`
-> `InferenceBase.convert_to_gym_action`
-> `GymApi.step`
-> `GymApi.set_ee_pose`
-> `MotionFactory.update_high_level_command`
-> `MotionFactory._controller_task`
-> `IKController.compute_controller` 或其他 controller
-> `RobotFactory.set_joint_commands`
-> `Fr3Arm.set_joint_command`
-> `panda_py_controller.set_control(...)`


## 七、结论

如果你的目标是“搞明白 FR3 是怎么收到命令的”，最关键的不是只盯着 `dp_inference.py`，而是要把下面三层连起来看：

- `InferenceBase` 负责把模型输出变成机器人动作
- `GymApi / MotionFactory` 负责把高层动作翻译成 joint target
- `RobotFactory / Fr3Arm` 负责把 joint target 真正发到硬件

如果你的目标是“自己写 inference”，最推荐的 base 是：

- 第一选择：`InferenceBase`
- 第二选择：参考 `DP_Inferencer`

除非你要重写整套硬件执行框架，否则不要从 `MotionFactory` 或 `Fr3Arm` 直接起步。


## 八、现有 base 和 DP 推理链路里我认为存在的问题

这一节不是说系统不能用，而是从“可读性、可维护性、可扩展性、时序可解释性”的角度，指出当前设计里比较明显的问题和风险点。

### 1. 配置语义和实际执行路径容易让人误判

当前 DP 默认配置里：

- `action_type` 是 `joint_position`
- 但底层 motion config 用的是 `ik` controller 和 cartesian plan 相关配置
- 同时观测里又可能包含 `ee_state`

这会让第一次读代码的人很容易误以为：

- policy 输出的是 ee pose
- 或者 `ee_state` 会被直接发给 FR3

但默认实际不是这样。

问题本质：

- 配置层把“观测空间”“动作空间”“底层控制器类型”混在了一起
- 这些概念虽然相关，但不是一回事

结果就是：

- 你必须跨很多层代码才能确认当前到底走 joint path 还是 ee path
- 很不利于快速调试和新人接手


### 2. `InferenceBase` 和下层实现耦合过深

`InferenceBase` 理论上应该只是推理公共层，但现在它直接访问了不少下层内部对象，例如：

- `self._gym_robot._robot_motion.get_model_dof_list()`
- `self._joint_positions`
- 一些 `_lock`、`_last_gripper_open`、`_tool_position_dof` 之类的内部状态

这带来的问题是：

- base class 不是面向稳定接口，而是直接依赖下层内部结构
- 一旦 `GymApi` 或 `MotionFactory` 里某些内部字段改名，inference 层就会一起坏
- 你想单独复用 `InferenceBase` 到另一种机器人/另一种 action packing 方式，会很痛苦

换句话说，当前的 `InferenceBase` 更像“项目专用执行模板”，不像一个边界清晰的抽象基类。


### 3. action packing 方式过于隐式，容易切错维度

当前 action 的拆分依赖很多隐含前提：

- dof 顺序
- arm / tool 的拼接顺序
- 单臂 / 双臂差异
- euler / quaternion 差异
- tool dof 的插入位置

这些逻辑主要散落在：

- `InferenceBase.convert_to_gym_action`
- `InferenceBase.convert_to_gym_action_single_step`
- `GymApi.step`

问题是：

- 没有一个统一的 action schema 对象
- 很多切片逻辑是手写 index 算出来的
- 一旦数据集动作维度或 tool 排列变化，很容易 silent mismatch

这种问题最麻烦的地方在于：

- 代码不一定报错
- 但机器人动作会“看起来能跑，实际不对”


### 4. `dp_inference.py` 里的观测转换函数可读性很差，而且看起来有异常代码片段

`DP_Inferencer._convert_gym_obs_to_dp_format()` 这一段现在能看到明显的重复/截断痕迹，里面混入了看起来不该出现的残片，比如重复的 `obs_dict_np` 拼接片段。

这至少说明两件事：

- 这个函数当前可读性很差
- 它很像经历过不完整合并或手工改动

即使它在当前环境下还能跑，这也是明显的维护风险。

因为这个函数正好处在：

- gym obs
- policy input

之间的关键位置。

一旦这里有隐藏 bug，结果通常不是直接 crash，而是：

- 输入错位
- state 维度错义
- 模型输出异常但难定位


### 5. `convert_from_gym_obs()` 用递归补 observation queue，不是一个好的控制流设计

在 `DP_Inferencer.convert_from_gym_obs()` 里，为了凑够 `n_obs_steps`，现在是：

- 不够就 `time.sleep(0.01)`
- 然后递归再次调用 `self.convert_from_gym_obs()`

这在逻辑上能工作，但设计上不好，原因是：

- 递归不适合做这种轮询式采样
- 调用链变深后不容易调试
- 出问题时 stack trace 会更难看
- 这件事本来更适合显式 while loop

对于实时推理代码来说，这种递归控制流会降低可读性。


### 6. `ee_state` 不是从硬件直接读 TCP，而是经由 joint state + model FK 计算出来

当前 `GymApi.get_ee_state()` 的路径是：

- `RobotFactory.get_joint_states()`
- `MotionFactory.get_frame_pose()`
- robot model FK
- 得到 ee pose

这意味着当前 `ee_state` 更准确地说是：

- 基于当前 joint states 计算出来的 model-side end-effector pose

这不一定等于：

- 控制器内部真实 TCP 状态
- 硬件直接返回的 flange / tool 实测 pose

潜在问题是：

- 如果 URDF / tool 外参 / 标定 / TCP 定义和真实硬件不完全一致
- 观测里的 ee pose 和真实末端就会有系统偏差

对于“把 ee pose 当观测强依赖”的 policy，这会影响表现。


### 7. 真实下发线程被 async smoother 隐藏了，时序不直观

当前链路里，主线程里你看到的是：

- `GymApi.step(...)`
- `RobotFactory.set_joint_commands(...)`

但如果开了：

- `use_smoother: true`
- `auto_enable_async_control: true`

那真正高频下发是：

- `RobotFactory._async_command_loop()`

这会带来几个问题：

- 主线程“动作已提交”不等于“硬件已执行”
- 推理频率和实际硬件发送频率被分离了
- 时序 bug 变得更难定位
- 你很难凭直觉判断某一步到底卡在 inference、smoother 还是 hardware send

这个设计本身不是错，但它需要更明确的文档和更清晰的状态可视化；否则调试非常费劲。


### 8. `MotionFactory.set_joint_positions(...)` 的 `_blocking_motion` 语义可疑

`MotionFactory.set_joint_positions(joint_commands, is_continous_joint_command=True)` 里：

- 一进函数先把 `_blocking_motion = True`
- 只有在 `is_continous_joint_command == False` 时才会把它改回去

而当前 `GymApi.set_joint_position()` 调它时传的是：

- `set_joint_positions(positions, True)`

这意味着 `_blocking_motion` 很可能会一直保持为 `True`。

这未必会立刻炸，因为 joint path 本身不是靠 controller thread 在发，但这是一个明显的语义风险：

- 如果后续又混用到 high-level ee command path
- `_controller_task()` 里有 `not self._blocking_motion` 条件
- 那 ee 路径可能被莫名其妙地抑制

这类状态变量如果不重新梳理，后面扩展多模式推理时很容易埋坑。


### 9. 观测里的 tool state 注释和实际实现有语义漂移风险

`GymApi.get_observation()` 里，注释强调要和训练数据统计一致，甚至提到数据集可能记录的是物理量，比如 mm。

但实际代码又对 tool state 做了：

- `/ self._tool_state_max`

也就是归一化。

这不一定错，但会留下一个很大的维护问题：

- 文档说的是“尽量匹配原始物理量”
- 实现却可能在做归一化

如果后续有人根据注释改数据集，或者根据代码改推理，两边很容易不一致。

对 imitation / diffusion policy 来说，这种 feature 统计不一致非常致命。


### 10. controller、action type、observation type 三者缺少统一的真值来源

现在你想回答“这一条推理到底怎么执行”，通常要同时看：

- inference config 里的 `action_type`
- motion config 里的 `controller_type`
- observation config 里的 `observation_type`
- 有时还得看 `use_relative_pose`
- 以及 smoother / async 是否打开

问题是这些信息分散在不同 yaml 和不同类里，没有一个统一的 runtime summary。

结果就是：

- 运行前很难一眼确认系统语义
- 调试时很难快速回答“当前到底是不是 ee 控制模式”
- 文档和实际执行路径容易长期漂移

这一点对于后续继续加新的 inference backend 会越来越痛。


## 九、如果以后要重构，我认为最优先的方向

这里只给方向，不展开具体实现。

### 1. 把 observation schema 和 action schema 显式化

至少应该把下面这些信息做成显式对象，而不是散在切片逻辑里：

- arm 维度
- tool 维度
- 单臂 / 双臂布局
- euler / quaternion 布局
- state 中各字段的顺序和单位


### 2. 把 `InferenceBase` 和 `GymApi` 之间收敛到稳定接口

理想情况是 inference 层不要直接碰：

- `_robot_motion`
- `_joint_positions`
- 其他私有字段

而是只通过几个稳定接口拿：

- action spec
- observation spec
- current state
- execute action


### 3. 给 runtime 打一份清晰的 execution summary

程序启动时最好明确打印：

- 当前 observation type
- 当前 action type
- 当前 controller type
- 是否启用 smoother
- 是否启用 async control
- 最终硬件发送线程是谁

这样能大幅降低误判。


### 4. 把实时线程和高层推理逻辑的边界写清楚

现在最大的问题之一不是不能用，而是：

- 主线程在干什么
- controller thread 在干什么
- async smoother thread 在干什么

边界不够显式。

这会让任何时序相关 bug 都变得很难解释。


## 十、对你自己写 inference 的实际建议

如果你只是想快速写一个新的 inference，而不是重做整套架构，我的建议非常明确：

- 继续复用 `InferenceBase`
- 继续复用 `GymApi`
- 不要一上来重写 `MotionFactory` 和 `RobotFactory`
- 先把你自己的 obs 映射和 action 映射写清楚
- 在第一版里优先走 joint action path，少碰 ee high-level command path

原因不是 ee path 不能用，而是当前代码里：

- joint path 更直接
- 调试更短
- 真机链路更容易确认
- 出问题时更容易追到底层 `Fr3Arm.set_joint_command(...)`

如果以后你确认自己确实需要“policy 直接输出 ee pose”，再单独梳理 ee path 会更稳。
