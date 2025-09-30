import sys
from a2d_sdk.robot import RobotDds as Robot
import time
robot = Robot()
time.sleep(0.5)  # 等待初始化
# print(robot.whole_body_status())

# robot.move_head([0, 0])
# time.sleep(2)
# robot.move_head([0.3, 0.5])
# time.sleep(2)

# 查询手部关节位置（12自由度）
# hand_pos, hand_timestamp = robot.hand_joint_states()
# if hand_pos[0] is not None:
#     print(f"当前手位置: {hand_pos}")
#     print(f"手部时间戳: {hand_timestamp}")
# else:
#     print("无法获取手部位置（可能末端不是手部类型）")

# # 使用夹爪模式控制灵巧手
# print("使用夹爪模式控制：两手半开")
robot.move_hand_as_gripper([0.0, 1.0])
# time.sleep(2)

# 也可以使用完整关节控制（12自由度）
# positions = [0.1] * 12  # 12个关节位置（弧度）
# robot.move_hand(positions)
# time.sleep(2)

# arm_pos, ts = robot.arm_joint_states()
# print(f"当前臂位置: {arm_pos}, 时间戳: {ts}")
# arm_pos[6]+=0.2
# arm_pos[13]+=0.2
# robot.move_arm(arm_pos)
# time.sleep(3)
# arm_pos, ts = robot.arm_joint_states()
# print(f"当前臂位置: {arm_pos}, 时间戳: {ts}")

# robot.reset()
# robot.shutdown()

# sys.exit(0)