import abc
import numpy as np
from typing import Sequence, Union

class ArmBase(metaclass=abc.ABCMeta):
  
  @abc.abstractmethod
  def get_model(self):
      pass
  
#   @abc.abstractmethod
#   def get_ee_orientation(self):
#       pass
  
  @abc.abstractmethod
  def get_tcp_pose(self):
      pass

#   @abc.abstractmethod
#   def get_ee_position(self):
#       pass

  @abc.abstractmethod
  def get_state(self):
      pass

 
