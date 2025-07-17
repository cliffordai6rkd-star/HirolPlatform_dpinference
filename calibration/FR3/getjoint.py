import panda_py
from panda_py import libfranka
import logging
from panda_py import constants
import numpy as np
from matplotlib import pyplot as plt

# Panda hostname/IP and Desk login information of your robot
hostname = '192.168.1.100'

# panda-py is chatty, activate information log level


# panda-py is chatty, activate information log level
import logging
logging.basicConfig(level=logging.INFO)
# 初始化
panda = panda_py.Panda(hostname)
gripper = libfranka.Gripper(hostname)

# pose = panda.get_pose()
pose = panda.q
print(pose)