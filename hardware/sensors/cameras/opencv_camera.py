import cv2
from hardware.base.camera import CameraBase
import warnings, threading, time, copy
import numpy as np

class OpencvCamera(CameraBase):
    def __init__(self, config):
        """
            image_shape: [height, width]
            device_id: /dev/video*
        """
        self._device_id = config['device_id']
        
        self._thread_running = True
        super().__init__(config)
        
    def initialize(self):
        if self._is_initialized:
            return True
        
        # this camera do not support imu & depth
        if self._contain_depth or self._contain_imu:
            raise ValueError("Opencv camera do not support imu and depth")
        
        self._cap = cv2.VideoCapture(self.id, cv2.CAP_V4L2)
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_shape[0])
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.img_shape[1])
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Test if the camera can read frames
        success, _ = self._cap.read()
        if not success:
            self.close()
            raise ValueError(f'could not read frame from the open cv camera {self._device_id}')

        # thread
        self._thread_handler = threading.Thread(target=self.update_camera_thread)
        self._thread_handler.start()
       
        return True
        
    def update_camera_thread(self):
        print(f'opencv camera thread started!!!')
        
        last_read_time = time.time()
        while self._thread_running:
            # frame reading
            ret, color_image = self.cap.read()
            if ret:
                self._lock.acquire()
                self._image_data = copy.deepcopy(color_image)
                self._lock.release()

            dt = time.time() - last_read_time
            if dt < (1.0 / self._fps):
                sleep_time = (1.0 / self._fps) - dt
                time.sleep(0.85 * sleep_time)
            else:
                warnings.warn(f'Camera could not reach the {self._fps}hz, '
                              f'actual freq: {1.0 / dt}hz')
            last_read_time = time.time()
        print(f'Opencv camera {self._device_id} thread is successfully stopped!')
            
    def close(self):
        self._thread_running = False
        self._thread_handler.join()
        self._cap.release()
        print(f'Opencv camera {self._device_id} successfully closed!!')
    
if __name__ == "__main__":
    config = ""
    