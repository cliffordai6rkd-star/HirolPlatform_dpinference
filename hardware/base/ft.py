import abc, threading, copy, time
import glob as log
import numpy as np

class FTBase(abc.ABC, metaclass=abc.ABCMeta):
    def __init__(self, config):
        self._config = config
        self._thread_lock = threading.Lock()
        self._zero_offset = np.zeros(6)
        self._ft_data = np.zeros(6)
        self._time_stamp = time.perf_counter()
        self._update_frequency = config.get("frequency", 300)
        self._is_initialized = False
        
    @abc.abstractmethod
    def initialize(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def close(self):
        raise NotImplementedError
    
    def get_ft_data(self):
        self._thread_lock.acquire()
        ft_data = copy.copy(self._ft_data)
        time_stamp = copy.copy(self._time_stamp)
        self._thread_lock.release()
        return ft_data, time_stamp
        