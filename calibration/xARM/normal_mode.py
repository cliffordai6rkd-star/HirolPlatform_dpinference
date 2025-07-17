import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from xarm.wrapper import XArmAPI



# left 192.168.11.11
# right 192.168.11.12
ip = '192.168.11.12'
arm = XArmAPI(ip)
arm.motion_enable(enable=True)
# # 示校模式2
# arm.set_mode(2)
# arm.set_state(state=0)


# 正常模式
arm.set_mode(0)
arm.set_state(state=0)
