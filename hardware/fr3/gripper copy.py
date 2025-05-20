from panda_py import controllers, libfranka, Panda
from hardware.base.gripper import GripperBase
import glog as log

class Gripper(GripperBase):
    def __init__(self, ip: str):
        super().__init__()
        self.instance = libfranka.Gripper(ip)
        log.info(f"Gripper instance created with IP: {ip}")
        log.info(f"Server version: {self.instance.server_version()}")

    def open(self) -> bool:
        self.instance.move(0.08, 0.2)
        pass

    def close(self) -> bool:
        self.instance.grasp(0, 0.2, 10, 0.04, 0.04)

    def stop(self) -> bool:
        return self.instance.stop()
    
    def is_gripping(self) -> bool:
        return self.instance.read_once().is_grasped

    def homing(self):
        self.instance.homing()
        log.info("Gripper homing completed.")

    def get_state(self):
        state = self.instance.read_once()
        log.info(
            f"is_grasped: {state.is_grasped}, \n"
            f"max_width: {state.max_width * 1e3}mm, \n"
            f"temperature: {state.temperature}, \n"
            f"time: {state.time.to_msec()}ms, \n"
            f"width: {state.width * 1e3}mm"
        )

    def grasp(self, width: float, speed: float, force: float, epsilon_inner: float = 0.005, epsilon_outer: float = 0.005) -> bool:
        return self.instance.grasp(width, speed, force, epsilon_inner, epsilon_outer)
