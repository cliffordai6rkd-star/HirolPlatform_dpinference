from teleop.base.teleoperation_base import TeleoperationDeviceBase
from pika.sense import Sense
from pika.tracker.vive_tracker import PoseData
import glog as log
import numpy as np
import time, threading
from sshkeyboard import listen_keyboard, stop_listening
from typing import Dict, Tuple, Optional
from scipy.spatial.transform import Rotation

class PikaTracker(TeleoperationDeviceBase):
    T_ROBOT_TRACKER = np.array([[ 0, 0,-1, 0],
                           [-1, 0, 0, 0],
                           [ 0, 1, 0, 0],
                           [ 0, 0, 0, 1]])
    T_TRACKER_ROBOT = np.array([[ 0,-1, 0, 0],
                           [ 0, 0, 1, 0],
                           [-1, 0, 0, 0],
                           [ 0, 0, 0, 1]])
    
    # init pose for absolute delta pose calculation
    INIT_TARGET_POSE = [[0,0,0,0,0,0,1], [0,0,0,0,0,0,1]]
    
    def __init__(self, config):
        super().__init__(config)
        self._serial_port: dict = config.get("serial_ports")
        self._tracker = {}
        for key, port in self._serial_port.items():
            tracker[key] = Sense(port=port)

        # Vive Tracker configuration
        self._vive_config_path = config.get("vive_config_path")
        self._vive_lh_config = config.get("vive_lh_config")
        self._vive_args = config.get("vive_args")
        
        # Device configuration
        self._device_id: dict[str, str] = config.get("target_device")  # Default tracker ID
        self._output_left = config.get("output_left", True)
        self._output_right = config.get("output_right", True)
        
        # Initialize offset pose for relative positioning
        self._init_pose = config.get('init_pose', None)
        self._last_quat = [np.array([0,0,0,1]), np.array([0,0,0,1])]
        if not self._init_pose is None:
            self._init_pose_rot = [self._init_pose["initial_pose_left"][3:], 
                                    self._init_pose["initial_pose_right"][3:]]
            self._last_quat = self._init_pose_rot
            self._init_pose_trans = [self._init_pose["initial_pose_left"][:3], 
                                    self._init_pose["initial_pose_right"][:3]]
        self._device_enabled = False
        
    def initialize(self) -> bool:
        """Initialize the Pika Sense device and Vive Tracker."""
        if self._is_initialized:
            return True
        
        try:
            # Connect to Pika Sense device
            if not self._tracker.connect():
                log.error("Failed to connect to Pika Sense device")
                return False
            
            # Set up Vive Tracker configuration if provided
            if self._vive_config_path or self._vive_lh_config or self._vive_args:
                self._tracker.set_vive_tracker_config(
                    config_path=self._vive_config_path,
                    lh_config=self._vive_lh_config,
                    args=self._vive_args
                )
            
            # Initialize Vive Tracker
            vive_tracker = self._tracker.get_vive_tracker()
            if vive_tracker is None:
                log.error("Failed to initialize Vive Tracker")
                return False
            
            # keyboard listening for update init pose
            listen_keyboard_thread = threading.Thread(target=listen_keyboard, 
                                        kwargs={"on_press": self._keyboard_on_press, 
                                                "until": None, "sequential": False,}, 
                                        daemon=True)
            listen_keyboard_thread.start()
            
            # Wait for tracker data to stabilize
            time.sleep(1.0)
            
            log.info(f"PikaTracker initialized successfully with device: {self._target_device}")
            return True
            
        except Exception as e:
            log.error(f"Failed to initialize PikaTracker: {e}")
            return False
        
    def _keyboard_on_press(self, key):
        if key == "i":
            pose, _ = self.read_data()
            if pose is None:
                log.warning("Failed to read pose data for update init pose!!!")
                return
            self.INIT_TARGET_POSE[0] = pose["left"]
            self.INIT_TARGET_POSE[1] = pose["right"]   
            self._device_enabled = True 
            log.info("init pose is updated!!!")

    def read_data(self):
        pose = self._tracker.get_pose(None)
        if pose is None: return None, None
        
        pose_quat = {}
        for device_name, cur_pose in pose.items():
            if self._output_left and self._device_id["left"] == device_name:
                cur_pose = PoseData()
                pose_quat["left"] = np.zeros(7)
                pose_quat["left"][:3] = cur_pose.position
                pose_quat["left"][3:] = [cur_pose.rotation[1:], cur_pose.rotation[0]]  # [qx, qy, qz, qw]
            if self._output_right and self._device_id["right"] == device_name:
                cur_pose = PoseData()
                pose_quat["right"] = np.zeros(7)
                pose_quat["right"][:3] = cur_pose.position
                pose_quat["right"][3:] = [cur_pose.rotation[1:], cur_pose.rotation[0]]  # [qx, qy, qz, qw]
        
        tool_data = {}
        for key, tracker in self._tracker.items():
            encoder = tracker.get_encoder_data()["rad"]
            tool_data[key] = encoder
        return pose_quat, tool_data

    def parse_data_2_robot_target(self, mode: str) -> Tuple[bool, Optional[Dict], Optional[Dict]]:
        """Parse Vive Tracker pose data to robot target format."""
        if 'absolute' not in mode:
            log.warn('The pika tracker only supports absolute pose related teleoperation')
            return False, None, None
        
        if not self._is_initialized:
            log.warning("Device not initialized")
            return False, None, None
        
        # Get pose data from Vive Tracker
        pose_quat, tool_encoder = self.read_data()
        if pose_quat is None:
            log.warn(f"No pose data available for device: {self._target_device}")
            return False, None, None
        
        # @TODO: Initialize reference pose on first valid reading
        if self._init_pose is None and command_state == 1:
            self._init_pose = pose_7d.copy()
            self._device_enabled = True
            log.info("Reference pose initialized, teleoperation enabled")
            return False, None, None
        
        # Prepare robot target dict
        pose_target = {}
        tool_target = {}
        
        # Configure output based on settings
        if self._output_left and self._output_right:
            pose_target['left'] = pose_7d
            pose_target['right'] = pose_7d  # Same pose for both arms
            tool_target['left'] = np.array([gripper_distance, command_state])
            tool_target['right'] = np.array([gripper_distance, command_state])
        elif self._output_left:
            pose_target['single'] = pose_7d
            tool_target['single'] = np.array([gripper_distance, command_state])
        else:  # output_right or default
            pose_target['single'] = pose_7d
            tool_target['single'] = np.array([gripper_distance, command_state])
        
        if mode == "absolute" or (mode == "absolute_delta" and self._device_enabled):
            return True, pose_target, tool_target
        else: return False, None, None
            
    def print_data(self):
        data = self.parse_data_2_robot_target("absolute")
        log.info(f'pose: {data[1]}')
    
    def close(self):
        """Clean up resources and disconnect devices."""
        try:
            self._device_enabled = False
            
            if self._tracker:
                self._tracker.disconnect()
                log.info("PikaTracker disconnected successfully")
                
        except Exception as e:
            log.error(f"Error closing PikaTracker: {e}")
            
        return True


if __name__ == "__main__":
    """Test script for PikaTracker."""
    
    # Example configuration
    test_config = {
        "serial_port": "/dev/ttyUSB0",
        "target_device": "tracker_LHR_CB1CD34E",  # Replace with your actual tracker ID
        "output_left": True,
        "output_right": False,
        "vive_config_path": None,  # Set if you have a specific config file
        "vive_lh_config": None,    # Set lighthouse config if needed  
        "vive_args": None,         # Additional pysurvive arguments
        "transform_matrix": [       # Identity matrix - modify as needed for coordinate transformation
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
    }
    
    print("🎯 Testing PikaTracker Implementation")
    print("=" * 50)
    
    try:
        # Initialize PikaTracker
        print("1️⃣  Initializing PikaTracker...")
        tracker = PikaTracker(test_config)
        
        if not tracker._is_initialized:
            print("❌ Failed to initialize PikaTracker")
            exit(1)
        
        print("✅ PikaTracker initialized successfully!")
        
        # Start data reading
        print("\n2️⃣  Starting data reading...")
        tracker.read_data()
        
        # Print available devices
        print("\n3️⃣  Available Vive Tracker devices:")
        devices = tracker._tracker.get_tracker_devices()
        for i, device in enumerate(devices):
            print(f"   {i+1}. {device}")
        
        if not devices:
            print("   ⚠️  No Vive Tracker devices found!")
            print("   Make sure your tracker is connected and SteamVR is running")
        
        # Test data parsing
        print("\n4️⃣  Testing data parsing (press Command button to enable)...")
        print("   Press Ctrl+C to stop")
        
        enabled_shown = False
        for i in range(100):  # Test for ~10 seconds
            success, pose_target, tool_target = tracker.parse_data_2_robot_target("absolute")
            
            if success:
                if not enabled_shown:
                    print("   ✅ Teleoperation enabled!")
                    enabled_shown = True
                
                if i % 10 == 0:  # Print every 10 iterations
                    print(f"\n   📊 Data at iteration {i}:")
                    if pose_target:
                        for key, pose in pose_target.items():
                            pos = pose[:3]
                            quat = pose[3:]
                            print(f"     {key} pose: pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], "
                                  f"quat=[{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]")
                    
                    if tool_target:
                        for key, tool in tool_target.items():
                            print(f"     {key} tool: gripper={tool[0]:.2f}mm, command={tool[1]}")
            
            elif i % 20 == 0:  # Show waiting message less frequently
                print("   ⏳ Waiting for valid data... (press Command button to enable)")
            
            time.sleep(0.1)
        
        print("\n5️⃣  Final device status:")
        tracker.print_data()
        
    except KeyboardInterrupt:
        print("\n⛔ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n6️⃣  Cleaning up...")
        if 'tracker' in locals():
            tracker.close()
        print("✅ Test completed!")
    
    print("\n" + "=" * 50)
    print("💡 Usage Notes:")
    print("• Make sure SteamVR is running and tracker is connected")
    print("• Update 'target_device' in config with your actual tracker ID")
    print("• Press the Command button on Pika Sense to enable teleoperation")
    print("• Modify 'transform_matrix' for coordinate system alignment")
    print("=" * 50)