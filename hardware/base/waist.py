import abc
import numpy as np
from typing import Sequence, Union

class WaistBase(metaclass=abc.ABCMeta):
  
  @abc.abstractmethod
  def print_state(self):
      pass
 
