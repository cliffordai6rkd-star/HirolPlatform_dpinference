import time
from a2d_sdk.robot import RobotController
robor_controller = RobotController()
while True:
    motion_status = robor_controller.get_motion_status()
    print(motion_status)
    time.sleep(1)