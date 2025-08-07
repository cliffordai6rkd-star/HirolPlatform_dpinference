import abc, threading, warnings, copy

class CameraBase(abc.ABC, metaclass=abc.ABCMeta):
    def __init__(self, config):
        self._config = config
        # [height width]
        self._img_shape = config['image_shape']
        self._contain_depth = config.get('contain_depth', False)
        self._contain_imu = config.get('contain_imu', False)
        self._fps = config.get('fps', None)
        self._lock = threading.Lock()
        # data 
        self._image_data = None
        self._depth_map_data = None
        self._imu_data = None
        self._is_initialized = False
        self._is_initialized = self.initialize()

    def capture_all_data(self):
        if not self._is_initialized:
            raise RuntimeError("Camera is not initialized, cannot capture data.")
        
        _, image = self.read_image()
        if self._contain_depth:
            _, depth_map = self.read_depth_map() 
        else:
            depth_map = None
        if self._contain_imu:
            _, imu_data = self.read_imu() if self._contain_imu else None
        else:
            imu_data = None
        return {
            'image': image,
            'depth_map': depth_map,
            'imu': imu_data
        }
    
    @abc.abstractmethod
    def initialize(self):
        raise NotImplementedError

    def read_image(self):
        if not self._is_initialized or self._image_data is None:
            warnings.warn(f"The camera is not initialized {self._is_initialized} or "
                          f"still not retrieve the image{self._image_data is None}")
            return False, None
        self._lock.acquire()
        image = copy.deepcopy(self._image_data)
        self._lock.release()
        return True, image
    
    def read_depth_map(self):
        if not self._contain_depth:
            warnings.warn("This camera does not support depth map")
            return False, None
        
        if not self._is_initialized or self._depth_map_data is None:
            warnings.warn("The camera is not initialized or "
                          "still not retrieve the depth map")
            return False, None
        self._lock.acquire()
        depth_map = copy.deepcopy(self._depth_map_data)
        self._lock.release()
        return True, depth_map

    def read_imu(self):
        if not self._contain_depth:
            warnings.warn("This camera does not support imu map")
            return False, None
        
        if not self._is_initialized or self._imu_data is None:
            warnings.warn("The camera is not initialized or "
                          "still not retrieve the imu")
            return False, None
        self._lock.acquire()
        imu = copy.deepcopy(self._imu_data)
        self._lock.release()
        return True, imu
    
    @abc.abstractmethod
    def close(self):
        raise NotImplementedError
    
    def get_resolution(self):
        return self._img_shape
    