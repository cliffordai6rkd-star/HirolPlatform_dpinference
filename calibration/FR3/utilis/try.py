import panda_py
# import torch
import numpy as np
from panda_py import libfranka,Panda
import logging
import time
import math

hostname = "192.168.1.102"
panda = Panda(hostname)
gripper = libfranka.Gripper(hostname)
panda.move_to_start(speed_factor=0.1)
# a = panda.get_pose()
# q = panda_py.ik(a)
# # panda.move_to_joint_position(q)
# print(q)