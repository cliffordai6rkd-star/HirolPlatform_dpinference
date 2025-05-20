from typing import Text, Mapping, Any
from hardware.base.camera import CameraBase
import pyrealsense2 as rs
import numpy as np
import cv2
import glog as log

class Camera(CameraBase):
    def __init__(self, config: Mapping[Text, Any]):
        super().__init__()

        self.pipe = rs.pipeline()
        profile = self.pipe.start()
        log.debug(f"pyrealsense pipe profile {profile}")

    def capture(self):
        rgb,d=None,None
        frames = self.pipe.wait_for_frames()
        for f in frames:
            log.debug(f.profile)

        rgb_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if rgb_frame and depth_frame:
        # Convert frames to numpy arrays
            rgb = np.asanyarray(rgb_frame.get_data())
            d = np.asanyarray(depth_frame.get_data())

            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            log.info(f"rgb:{rgb.shape}, depth:{d.shape}")
            # Save the frames as images
            # cv2.imwrite('rgb.png', rgb)
            # cv2.imwrite('depth.png', d)

        return rgb, d