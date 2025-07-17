# 条件导入 a2d_sdk
try:
    from a2d_sdk.robot import CosineCamera as Camera
    A2D_SDK_AVAILABLE = True
except ImportError:
    A2D_SDK_AVAILABLE = False
    print("Warning: a2d_sdk not available, using mock camera implementation")
    
    import numpy as np
    
    class MockCamera:
        def __init__(self, camera_group):
            self._camera_group = camera_group
            print(f"[Mock] Camera initialized for group: {camera_group}")
        
        def get_fps(self, camera_name):
            return 30.0
        
        def get_latest_image(self, name):
            if 'depth' in name:
                return np.zeros((480, 640), dtype=np.uint16)
            else:
                return np.zeros((480, 640, 3), dtype=np.uint8)
        
        def close(self):
            print("[Mock] Camera closed")
    
    Camera = MockCamera

from hardware.base.camera import CameraBase
import threading, time, warnings

class AgibotCamera(CameraBase):
    def __init__(self, config):
        """
        Initialize the Agibot camera with the given configuration.
        :param config: Configuration dictionary containing camera parameters.
        """
        self._camera_name = config.get('camera_name', 'head')
        if self._camera_name == 'head':
            self._camera_group = [self._camera_name, self._camera_name + '_depth']
        self._camera = Camera(self._camera_group)
        self._thread_running = True
        
        super().__init__(config)

    def initialize(self):
        if self._is_initialized:
            return True
        
        self._fps = 0.0
        self._contain_depth = False
        for camera_name in self._camera_group:
            curr_fps = self._camera.get_fps(camera_name)
            if self._fps < curr_fps:
                self._fps = curr_fps
            if 'depth' in camera_name:
                self._contain_depth = True
        
        self._thread = threading.Thread(target=self.update_tasks)
        self._thread.start()
        return True
        
    def update_tasks(self):
        print(f'Agibot camera {self._camera_group} started update thread!!!')
        
        last_read_time = time.time()
        while self._thread_running:
            image = self.get_image(self._camera_group[0])
            if self._contain_depth:
                depth = self.get_image(self._camera_group[1])
            self._lock.acquire()
            self._image_data = image
            self._depth_map_data = depth if self._contain_depth else None
            self._lock.release()
            
            dt = time.time() - last_read_time
            if dt < (1.0 / self._fps):
                sleep_time = (1.0 / self._fps) - dt
                time.sleep(sleep_time)
            elif dt > 1.2 * (1.0 / self._fps):
                warnings.warn(f'Read frequency is slow: expected: {1.0 / self._fps}, '
                              f'actual: {1.0 / dt}')
            last_read_time = time.time()
        
        
    def get_image(self, name):
        return self._camera.get_latest_image(name)
    
    def close(self):
        self._camera.close()     
