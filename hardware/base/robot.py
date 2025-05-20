import sys
sys.path.append('.')

import logging, abc

# from hardware import camera as camera_lib
# from hardware import ft_sensor as ft_sensor_lib

class Robot(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    def print_state(self):
        pass

    # def get_latency(self) -> float:
    #     """
    #     Returns the latency of the robot's communication system in seconds.
    #     """
    #     raise NotImplementedError("This method should be implemented by subclasses.")

    # def get_transmission_rate(self) -> float:
    #     """
    #     Returns the transmission rate of the robot's communication system in Mbps.
    #     """
    #     raise NotImplementedError("This method should be implemented by subclasses.")

    # def get_frame_loss_rate(self) -> float:
    #     """
    #     Returns the frame loss rate of the robot's communication system as a percentage.
    #     """
    #     raise NotImplementedError("This method should be implemented by subclasses.")

    # def get_repeatability(self) -> float:
    #     """
    #     Returns the repeatability of the robot in millimeters.
    #     """
    #     raise NotImplementedError("This method should be implemented by subclasses.")

    # def get_tracking_accuracy(self) -> float:
    #     """
    #     Returns the tracking accuracy of the robot in millimeters.
    #     """
    #     raise NotImplementedError("This method should be implemented by subclasses.")

    # def get_hand_eye_calibration_accuracy(self) -> float:
    #     """
    #     Returns the hand-eye calibration accuracy of the robot in millimeters.
    #     """
    #     raise NotImplementedError("This method should be implemented by subclasses.")

    def move_to_start(self) -> None:
        raise NotImplementedError("This method should be implemented by subclasses.")
