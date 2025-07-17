from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="hand_eye_calib",
            executable="hand_eye_calib_node",
            output="screen",
        ),
    ])
