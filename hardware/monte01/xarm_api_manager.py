"""
XArm API Manager - Singleton pattern to share XArmAPI instance
"""
try:
    from xarm.wrapper import XArmAPI
except ImportError:
    XArmAPI = None

import threading
import glog as log
from hardware.monte01.xarm_defs import *


class XArmAPIManager:
    """
    Singleton manager for XArmAPI instances
    
    Since XArmAPI can only have one active connection per robot IP,
    this manager ensures that arm and gripper components share the same instance.
    """
    
    _instances = {}  # Dictionary to store instances per IP
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls, ip: str, **kwargs) -> XArmAPI:
        """
        Get or create XArmAPI instance for given IP
        
        Args:
            ip: Robot IP address
            **kwargs: Additional arguments for XArmAPI initialization
            
        Returns:
            XArmAPI instance or None if initialization failed
        """
        with cls._lock:
            if ip not in cls._instances:
                try:
                    # Set default parameters
                    default_kwargs = {
                        'default_gripper_baud': 921600,
                        'is_radian': True
                    }
                    default_kwargs.update(kwargs)
                    
                    log.info(f"Creating new XArmAPI instance for {ip}")
                    instance = XArmAPI(ip, **default_kwargs)
                    
                    # Basic initialization
                    instance.clean_warn()
                    instance.clean_error()
                    instance.motion_enable(enable=True)
                    instance.set_mode(XARM_MODE_SERVO)
                    instance.set_state(state=XARM_STATE_SPORT)
                    
                    cls._instances[ip] = instance
                    log.info(f"XArmAPI instance created successfully for {ip}")
                    
                except Exception as e:
                    log.error(f"Failed to create XArmAPI instance for {ip}: {e}")
                    cls._instances[ip] = None
                    
            return cls._instances.get(ip)
    
    @classmethod
    def close_instance(cls, ip: str):
        """
        Close XArmAPI instance for given IP
        
        Args:
            ip: Robot IP address
        """
        with cls._lock:
            if ip in cls._instances and cls._instances[ip] is not None:
                try:
                    cls._instances[ip].disconnect()
                    log.info(f"XArmAPI instance closed for {ip}")
                except Exception as e:
                    log.error(f"Error closing XArmAPI instance for {ip}: {e}")
                finally:
                    del cls._instances[ip]
    
    @classmethod
    def close_all(cls):
        """Close all XArmAPI instances"""
        with cls._lock:
            for ip in list(cls._instances.keys()):
                cls.close_instance(ip)
    
    @classmethod
    def get_active_instances(cls) -> list:
        """Get list of active IP addresses"""
        with cls._lock:
            return [ip for ip, instance in cls._instances.items() if instance is not None]