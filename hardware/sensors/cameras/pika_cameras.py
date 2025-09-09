from hardware.base.camera import CameraBase
from pika.camera import FisheyeCamera, RealSenseCamera
import glog as log
import time
import threading, copy

class PikaCameras(CameraBase):
    def __init__(self, config):
        self._fisheye_index = config.get("fisheye_index", 0)
        self._rs_serial = config["realsense_serial"]
        self._thread_running = True
        self._thread = None
        self._fish_eye_image_data = None
        super().__init__(config)
       
    def initialize(self):
        if self._is_initialized:
            log.warn(f'The pika cameras are already initialized!!!')
            return

        self._realsense_camera = RealSenseCamera(camera_width=self._img_shape[1], 
                                                 camera_height=self._img_shape[0],
                                                 serial_number=self._rs_serial)
        if not self._realsense_camera.connect():
            log.error(f'Connect pika realsense camera failed!!')
            return False
        
        self._fish_eye_camera = FisheyeCamera(camera_width=self._img_shape[1], 
                                              camera_height=self._img_shape[0],
                                              device_id=self._fish_eye_camera())
        if not self._fish_eye_camera.connect():
            log.error(f'Connect pika fish eye camera failed!!')
            return False
        self._thread = threading.Thread(target=self.update_thread)
        self._thread.start()
        return True
        
    def update_thread(self):
        while self._thread_running:
            success, rgb, depth = self._realsense_camera.get_frames()
            fishe_eye_success, fish_eye_image = self._fish_eye_camera.get_frame()
            self._lock.acquire()
            if success:
                self._image_data = rgb
                self._depth_map_data = depth
                self._time_stamp = time.perf_counter()
            if fishe_eye_success:
                self._fish_eye_image_data = fish_eye_image
            self._lock.release()
    
    def capture_all_data(self):
        all_data = super().capture_all_data()
        self._lock.acquire()
        all_data["fish_eye_image"] = copy.copy(self._fish_eye_image_data)
        self._lock.release()
        
    def close(self):
        self._thread_running = False
        if not self._thread is None:
            self._thread.join()
        self._fish_eye_camera.disconnect()
        self._realsense_camera.disconnect()
    