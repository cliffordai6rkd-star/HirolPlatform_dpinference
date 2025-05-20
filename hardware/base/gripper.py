"""Gripper interface, model, type, and config.

Copyright 2021 Wenzhao Lian. All rights reserved.
"""

import abc
import numpy as np


class GripperBase(metaclass=abc.ABCMeta):
    
  @abc.abstractmethod
  def print_state(self) -> None:
    pass

  # @abc.abstractmethod
  # def close(self) -> bool:
  #   pass

  # @abc.abstractmethod
  # def is_gripping(self) -> bool:
  #   pass
