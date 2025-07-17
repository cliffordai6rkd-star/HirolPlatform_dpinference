import pyrealsense2 as rs
from hardware.base.camera import CameraBase
import warnings, threading, time
import numpy as np

class RealsenseCamera(CameraBase):
    def __init__(self, config):
        """
            image_shape: [height, width]
            serial_number: the serial port your camera connected
        """
        self._img_shape = config['image_shape']
        self._fps = config['fps']
        self._serial_number = config['serial_number']
        
        align_to = rs.stream.color
        self._align = rs.align(align_to)
        self._thread_running = True
        super().__init__(config)
        
        
    def initialize(self):
        if self._is_initialized:
            return True
        
        self._pipeline = rs.pipeline()
        rs_config = rs.config()
        if self._serial_number is not None:
            rs_config.enable_device(self._serial_number)

        rs_config.enable_stream(rs.stream.color, self._img_shape[1], self._img_shape[0], rs.format.bgr8, self._fps)

        if self._contain_depth:
            rs_config.enable_stream(rs.stream.depth, self._img_shape[1], self._img_shape[0], rs.format.z16, self._fps)

        if self._contain_imu:
            rs_config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)  # acclerometer
            rs_config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)   # gyro
        
        profile = self._pipeline.start(rs_config)
        self._device = profile.get_device()
        if self._device is None:
            raise ValueError(f"Could not construct the realsense camera {self._serial_number}")
        
        if self._contain_depth:
            assert self._device is not None
            depth_sensor = self._device.first_depth_sensor()
            self.g_depth_scale = depth_sensor.get_depth_scale()

        self._intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        
        # start thread
        self._thread_handler = threading.Thread(target=self.update_camera_thread)
        self._thread_handler.start()
        while self._image_data is None:
            pass
        
        print(f'The realsesne camera is successfully initialized!!!')
        return True
        
    def update_camera_thread(self):
        print(f'Realsense camera {self._serial_number} thread started!!!')

        last_read_time = time.time()
        while self._thread_running:
            # frame reading
            frames = self._pipeline.wait_for_frames() # blocking
            aligned_frames = self._align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            if self._contain_depth:
                depth_frame = aligned_frames.get_depth_frame()
            if self._contain_imu:
                accel_frame = frames.first_or_default(rs.stream.accel)
                gyro_frame = frames.first_or_default(rs.stream.gyro)
                if accel_frame and gyro_frame:
                    # imu data obtain 
                    accel_data = accel_frame.as_motion_frame().get_motion_data()
                    gyro_data = gyro_frame.as_motion_frame().get_motion_data()
                    # timestamp = accel_frame.get_timestamp()

            if not color_frame:
                time.sleep(0.01)
                continue

            self._lock.acquire()
            self._image_data = np.asanyarray(color_frame.get_data())
            # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            self._depth_map_data = np.asanyarray(depth_frame.get_data()) if self._contain_depth else None
            if self._contain_imu:
                self._imu_data = np.array([accel_data.x, accel_data.y, accel_data.z,
                                           gyro_data.x, gyro_data.y, gyro_data.z])
            self._lock.release()
            
            dt = time.time() - last_read_time
            if dt < (1.0 / self._fps):
                sleep_time = (1.0 / self._fps) - dt
                time.sleep(0.8 * sleep_time)
            elif dt > 1.2 / self._fps:
                warnings.warn(f'Camera could not reach the {self._fps}hz, '
                              f'actual freq: {1.0 / dt}hz')
            last_read_time = time.time()
        print(f'Realsense {self._serial_number} thread is suceessfully stopped!!')
    
    def close(self):
        self._thread_running = False
        self._thread_handler.join()
        self._pipeline.stop()
        print(f'Realsense {self._serial_number} is successfully closed!!!')
    
if __name__ == "__main__":
    import os, yaml

    def get_cfg(file):
        cur_path = os.path.dirname(os.path.abspath(__file__))
        cfg_file = os.path.join(cur_path, file)
        with open(cfg_file, 'r') as stream:
            config = yaml.safe_load(stream)
        return config

    config = "hardware/sensors/cameras/config/d435i_cfg.yaml"
    import cv2
    d435_cfg = get_cfg(config)["D435i"]
    
    d435 = RealsenseCamera(d435_cfg) 
    while True:
        all_data = d435.capture_all_data()
        print(f"data: {all_data}")
        cv2.namedWindow('RealSense Color', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense Color', all_data['image'])
        
        cv2.namedWindow('RealSense Depth', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense Depth', all_data['depth_map'])
        
        print(f"imu data: {all_data['imu']}")
        
        # 按'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.01)
        