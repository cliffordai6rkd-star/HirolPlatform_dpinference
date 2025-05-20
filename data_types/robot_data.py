import dataclasses
from matplotlib import pyplot as plt
import numpy as np
from typing import Sequence

from data_types import se3


@dataclasses.dataclass
class FtData:
  timestamp: float
  wrench: np.ndarray


class RobotState(object):
  def __init__(self,
               timestamp: float,
               pose: se3.Transform = None,
               twist: np.ndarray = None,
               jpos: np.ndarray = None,
               jvel: np.ndarray = None,
               ft_data: FtData = None):
    self.pose = pose
    self.twist = twist
    self.jpos = jpos
    self.jvel = jvel
    self.timestamp = timestamp
    self.ft_data = ft_data


class RobotCommand(object):
  def __init__(self,
               pose: se3.Transform = None,
               twist: np.ndarray = None,
               wrench: np.ndarray = None):
    self.pose = pose
    self.twist = twist
    self.wrench = wrench


def plot_wrench(wrench_log: Sequence[FtData]):
    ts = [x.timestamp for x in wrench_log]
    fts = np.array([x.wrench for x in wrench_log])
    plt.subplot(211)
    plt.plot(ts, fts[:, 0], 'r')
    plt.plot(ts, fts[:, 1], 'g')
    plt.plot(ts, fts[:, 2], 'b')
    plt.subplot(212)
    plt.plot(ts, fts[:, 3], 'r')
    plt.plot(ts, fts[:, 4], 'g')
    plt.plot(ts, fts[:, 5], 'b')
    plt.show()
