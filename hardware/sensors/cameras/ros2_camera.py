"""
ROS2 camera implementation.

This module is designed to degrade gracefully on machines without a working
ROS2/rclpy installation (e.g. missing `librcl_action.so` or no ROS setup
sourced). In that case we fall back to a lightweight mock camera so that the
rest of the inference stack can still run.
"""

# Try to import rclpy and related classes. If this fails (e.g. missing
# native ROS2 libs), keep going with a mock implementation.
try:  # pragma: no cover - environment-dependent import
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
    from rclpy.executors import SingleThreadedExecutor
    RCLPY_AVAILABLE = True
except Exception as e:  # ImportError or runtime load errors
    RCLPY_AVAILABLE = False
    import warnings as _warnings

    _warnings.warn(
        f"rclpy failed to import or initialize ({e!s}). "
        "Ros2Camera will fall back to a mock implementation. "
        "If you need real ROS2 camera input, please fix your ROS2 setup "
        "(source the correct setup.bash, ensure shared libraries such as "
        "librcl_action.so are on LD_LIBRARY_PATH, etc.)."
    )
    # Define lightweight placeholders so type hints and attribute access
    # inside dead code paths do not crash when rclpy is absent.
    rclpy = None  # type: ignore[assignment]
    Node = object  # type: ignore[assignment]
    QoSProfile = object  # type: ignore[assignment]
    QoSReliabilityPolicy = object  # type: ignore[assignment]
    QoSHistoryPolicy = object  # type: ignore[assignment]
    SingleThreadedExecutor = object  # type: ignore[assignment]

try:  # pragma: no cover - depends on ROS Python installation
    from sensor_msgs.msg import Image, Imu
except Exception:
    # Minimal placeholders to satisfy type annotations if sensor_msgs
    # is unavailable; these are never used when running in mock mode.
    class Image:  # type: ignore[no-redef]
        pass

    class Imu:  # type: ignore[no-redef]
        pass

# 条件导入 cv_bridge
try:
    from cv_bridge import CvBridge, CvBridgeError
    CV_BRIDGE_AVAILABLE = True
except ImportError:
    CV_BRIDGE_AVAILABLE = False
    print("Warning: cv_bridge not available, using mock implementation")
    
    import numpy as np
    
    class CvBridgeError(Exception):
        pass
    
    class MockCvBridge:
        def __init__(self):
            pass
            
        def imgmsg_to_cv2(self, msg, desired_encoding="bgr8"):
            """Mock implementation of cv_bridge imgmsg_to_cv2"""
            try:
                if desired_encoding in ["bgr8", "rgb8"]:
                    # 8-bit color image
                    image_array = np.frombuffer(msg.data, dtype=np.uint8)
                    image = image_array.reshape((msg.height, msg.width, 3))
                elif desired_encoding == "16UC1":
                    # 16-bit depth image
                    image_array = np.frombuffer(msg.data, dtype=np.uint16)
                    image = image_array.reshape((msg.height, msg.width))
                elif desired_encoding == "32FC1":
                    # 32-bit float depth image
                    image_array = np.frombuffer(msg.data, dtype=np.float32)
                    image = image_array.reshape((msg.height, msg.width))
                else:
                    raise CvBridgeError(f"Unsupported encoding: {desired_encoding}")
                return image
            except Exception as e:
                raise CvBridgeError(f"Mock cv_bridge conversion failed: {e}")
    
    CvBridge = MockCvBridge

from hardware.base.camera import CameraBase
import threading
import time
import copy
import warnings
import numpy as np
import cv2
import glog as log

class Ros2Camera(CameraBase):
    def __init__(self, config):
        """
        ROS2 Camera implementation that subscribes to ROS2 image topics
        
        Config parameters:
            image_topic: str - ROS2 topic for color images (default: '/camera_head_front/color/image_raw')
            depth_topic: str - ROS2 topic for depth images (optional)
            imu_topic: str - ROS2 topic for IMU data (optional)
            fps: int - Expected frame rate for data validation (default: 30)
            image_shape: [height, width] - Expected image dimensions
            contain_depth: bool - Whether to subscribe to depth data
            contain_imu: bool - Whether to subscribe to IMU data
        """
        # Initialize ROS2 first (if not already initialized). When rclpy is not
        # available we simply skip this and later run in mock mode.
        if RCLPY_AVAILABLE and rclpy is not None:
            try:
                if not rclpy.ok():
                    rclpy.init(args=None)
            except Exception as e:
                log.warning(f"Failed to initialize ROS2 (using rclpy): {e}")
        else:
            log.info(
                "rclpy is not available; Ros2Camera will operate in mock mode "
                "with synthetic images instead of real ROS2 topics."
            )
        
        # Extract configuration
        self._image_topic = config.get('image_topic', '/camera_head_front/color/image_raw')
        self._depth_topic = config.get('depth_topic', '/camera_head_front/depth/image_rect_raw')
        self._imu_topic = config.get('imu_topic', '/camera_head_front/imu')
        self._fps = config.get('fps', 30)
        self._img_shape = config.get('image_shape', [480, 640])
        
        # Thread management
        self._thread_running = True
        self._ros_node = None
        self._executor = None
        self._ros_thread = None
        self._bridge = CvBridge()
        self._mock_thread = None

        # Data timestamps for FPS monitoring
        self._last_image_time = 0
        self._last_depth_time = 0
        self._last_imu_time = 0

        # Flag indicating whether we should use a mock implementation instead
        # of real ROS2 subscription (e.g. rclpy missing or failed to load).
        self._use_mock = not RCLPY_AVAILABLE

        # Initialize base class
        super().__init__(config)
    
    def initialize(self):
        """Initialize ROS2 node and create subscriptions"""
        if self._is_initialized:
            return True

        # In environments without a working rclpy/ROS2 setup, fall back to a
        # simple mock publisher that produces black images. This keeps the rest
        # of the stack running (e.g. inference from datasets) without requiring
        # full ROS2.
        if self._use_mock:
            log.warning(
                "ROS2 (rclpy) is not available; Ros2Camera is running in "
                "mock mode with zero-valued images."
            )
            self._start_mock_stream()
            return True

        log.info(f"Initializing ROS2 Camera - image_topic: {self._image_topic}")
        
        try:
            # Check if ROS2 environment is available with timeout
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("ROS2 initialization timeout")
            
            # Set a timeout for ROS2 initialization (5 seconds)
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(5)
            
            try:
                # Create ROS2 node with timeout protection
                node_name = f'ros2_camera_{id(self)}'
                self._ros_node = Node(node_name)
                log.info(f"ROS2 Camera node created")
                signal.alarm(0)  # Cancel the alarm
            except TimeoutError:
                signal.alarm(0)  # Cancel the alarm
                log.warning("ROS2 Camera initialization timed out - ROS2 environment may not be available")
                return False
            
            # Set up QoS profile for sensor data
            # Use BEST_EFFORT for real-time data, allowing some frame drops
            qos_profile = QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1  # Keep only the latest message
            )
            
            # Create image subscription
            self._image_subscription = self._ros_node.create_subscription(
                Image,
                self._image_topic,
                self._image_callback,
                qos_profile
            )
            log.info(f"Subscribed to image topic: {self._image_topic}")
            
            # Create depth subscription if needed
            if self._contain_depth:
                self._depth_subscription = self._ros_node.create_subscription(
                    Image,
                    self._depth_topic, 
                    self._depth_callback,
                    qos_profile
                )
                log.info(f"Subscribed to depth topic: {self._depth_topic}")
            
            # Create IMU subscription if needed
            if self._contain_imu:
                self._imu_subscription = self._ros_node.create_subscription(
                    Imu,
                    self._imu_topic,
                    self._imu_callback,
                    qos_profile
                )
                log.info(f"Subscribed to IMU topic: {self._imu_topic}")
            
            # Start ROS2 spinning in separate thread
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._ros_node)
            
            # Add callback counters for debugging
            self._image_callback_count = 0
            self._depth_callback_count = 0
            
            self._ros_thread = threading.Thread(target=self._ros_spin_thread, daemon=True)
            self._ros_thread.start()
            
            # Wait a bit to ensure subscriptions are established (with timeout)
            time.sleep(0.5)  # Increased wait time to allow subscription
            
            log.info(f"ROS2 Camera initialized successfully")
            log.info(f"  - Image topic: {self._image_topic}")
            if self._contain_depth:
                log.info(f"  - Depth topic: {self._depth_topic}")
            if self._contain_imu:
                log.info(f"  - IMU topic: {self._imu_topic}")
            return True
            
        except Exception as e:
            log.warning(f"ROS2 Camera initialization failed (this is OK if ROS2 is not running): {e}")
            # Cleanup any partial initialization
            self._cleanup_on_failure()
            # Return False to indicate camera not available, but don't crash the system
            return False

    def _start_mock_stream(self):
        """Start background thread that publishes dummy images in mock mode."""

        def _mock_loop():
            log.info("Ros2Camera mock stream thread started (no ROS2 backend).")
            fps = float(self._fps or 30)
            dt = 1.0 / max(fps, 1.0)
            h, w = self._img_shape

            while self._thread_running:
                # Generate black image and optional depth map
                color = np.zeros((h, w, 3), dtype=np.uint8)
                depth = (
                    np.zeros((h, w), dtype=np.uint16) if self._contain_depth else None
                )
                with self._lock:
                    self._image_data = color
                    self._depth_map_data = depth
                    self._time_stamp = time.perf_counter()
                time.sleep(dt)

        self._mock_thread = threading.Thread(target=_mock_loop, daemon=True)
        self._mock_thread.start()
    
    def _cleanup_on_failure(self):
        """Clean up resources when initialization fails"""
        try:
            self._thread_running = False
            if hasattr(self, '_executor') and self._executor:
                self._executor.shutdown()
            if hasattr(self, '_ros_node') and self._ros_node:
                self._ros_node.destroy_node()
        except Exception as cleanup_error:
            log.info(f"Cleanup error (ignored): {cleanup_error}")
    
    def _ros_spin_thread(self):
        """Run ROS2 executor in separate thread"""
        log.info("ROS2 Camera spin thread started")
        spin_count = 0
        try:
            # Guard with RCLPY_AVAILABLE so this never runs when rclpy is
            # missing; in that case we rely on the mock stream instead.
            while self._thread_running and RCLPY_AVAILABLE and rclpy.ok():
                self._executor.spin_once(timeout_sec=0.1)
                spin_count += 1
                
                # Log spin activity every 100 cycles (roughly every 10 seconds)
                if spin_count % 100 == 0:
                    log.debug(f"ROS2 spin thread active: cycle #{spin_count}, callbacks received: {getattr(self, '_image_callback_count', 0)}")
                    
        except Exception as e:
            log.error(f"ROS2 spin thread error at cycle {spin_count}: {e}")
            import traceback
            log.error(f"Traceback: {traceback.format_exc()}")
        finally:
            log.info(f"ROS2 Camera spin thread stopped after {spin_count} cycles")
    
    def _image_callback(self, msg: Image):
        """Callback for color image messages"""
        try:
            self._image_callback_count += 1
            
            if self._image_callback_count == 1:
                log.info(f"First image received: {msg.width}x{msg.height}, encoding: {msg.encoding}")
            elif self._image_callback_count <= 5:  # Log first 5 frames for debugging
                log.debug(f"Image callback #{self._image_callback_count}: {msg.width}x{msg.height}")
            elif self._image_callback_count % 30 == 0:  # Then log every 30 frames
                log.debug(f"Image callback #{self._image_callback_count}: {msg.width}x{msg.height}")
            
            # Convert ROS Image to OpenCV format
            # Handle different image encodings
            if msg.encoding == "rgb8":
                cv_image = self._bridge.imgmsg_to_cv2(msg, "rgb8")
                # Convert RGB to BGR for OpenCV
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            elif msg.encoding == "bgr8":
                cv_image = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            else:
                log.warning(f"Unsupported image encoding: {msg.encoding}, attempting bgr8 conversion")
                cv_image = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Validate image dimensions
            if cv_image.shape[:2] != tuple(self._img_shape):
                log.info(f"Image size mismatch: expected {self._img_shape}, got {cv_image.shape[:2]} - updating expected size")
                self._img_shape = list(cv_image.shape[:2])
            
            # Thread-safe update of image data
            # Use direct assignment instead of deepcopy for better performance
            with self._lock:
                self._image_data = cv_image.copy()  # OpenCV copy is faster than deepcopy
                self._time_stamp = time.perf_counter()
            
            # Monitor frame rate
            current_time = time.perf_counter()
            if self._last_image_time > 0:
                fps = 1.0 / (current_time - self._last_image_time)
                if self._image_callback_count % 30 == 0:  # Log FPS every 30 frames
                    log.debug(f"Image FPS: {fps:.1f}")
            self._last_image_time = current_time
            
        except CvBridgeError as e:
            log.error(f"Image conversion error: {e}")
            import traceback
            log.error(f"CvBridge traceback: {traceback.format_exc()}")
        except Exception as e:
            log.error(f"Image callback error: {e}")
            import traceback
            log.error(f"Callback traceback: {traceback.format_exc()}")
    
    def _depth_callback(self, msg: Image):
        """Callback for depth image messages"""
        if not self._contain_depth:
            return
            
        try:
            self._depth_callback_count += 1
            if self._depth_callback_count == 1:
                log.info(f"First depth image received: {msg.width}x{msg.height}, encoding: {msg.encoding}")
            elif self._depth_callback_count % 30 == 0:
                log.info(f"Depth callback #{self._depth_callback_count}")
                
            # Convert depth image (usually 16UC1 or 32FC1)
            if msg.encoding == "16UC1":
                cv_depth = self._bridge.imgmsg_to_cv2(msg, "16UC1")
            elif msg.encoding == "32FC1": 
                cv_depth = self._bridge.imgmsg_to_cv2(msg, "32FC1")
            else:
                log.warning(f"Unsupported depth encoding: {msg.encoding}")
                return
            
            # Thread-safe update of depth data
            with self._lock:
                self._depth_map_data = cv_depth.copy()  # Faster than deepcopy
            
            # Monitor frame rate
            current_time = time.perf_counter()
            if self._last_depth_time > 0:
                fps = 1.0 / (current_time - self._last_depth_time)
                if fps < self._fps * 0.8:
                    log.info(f"Depth FPS low: {fps:.1f} (expected: {self._fps})")
            self._last_depth_time = current_time
            
        except CvBridgeError as e:
            log.error(f"Depth conversion error: {e}")
        except Exception as e:
            log.error(f"Depth callback error: {e}")
            import traceback
            log.error(f"Depth callback traceback: {traceback.format_exc()}")
    
    def _imu_callback(self, msg: Imu):
        """Callback for IMU messages"""
        if not self._contain_imu:
            return
            
        try:
            # Extract IMU data
            imu_data = {
                'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
                'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z],
                'orientation': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
                'timestamp': time.time()
            }
            
            # Thread-safe update of IMU data
            with self._lock:
                self._imu_data = copy.deepcopy(imu_data)
            
            # Monitor data rate
            current_time = time.perf_counter()
            if self._last_imu_time > 0:
                rate = 1.0 / (current_time - self._last_imu_time)
                if rate < 50:  # IMU typically runs at higher rates
                    log.info(f"IMU rate low: {rate:.1f} Hz")
            self._last_imu_time = current_time
            
        except Exception as e:
            log.error(f"IMU callback error: {e}")
    
    def close(self):
        """Clean up resources"""
        log.info("Closing ROS2 Camera...")
        
        # Stop the spinning thread
        self._thread_running = False
        
        if self._ros_thread and self._ros_thread.is_alive():
            self._ros_thread.join(timeout=2.0)
        
        # Clean up ROS2 resources
        if self._executor:
            if self._ros_node:
                self._executor.remove_node(self._ros_node)
            self._executor.shutdown()
        
        if self._ros_node:
            self._ros_node.destroy_node()
        
        log.info("ROS2 Camera closed successfully")

    def get_resolution(self):
        """Get camera resolution"""
        return self._img_shape
    
    def get_status(self):
        """Get camera status for debugging"""
        status = {
            'initialized': self._is_initialized,
            'ros_node_active': self._ros_node is not None,
            'image_topic': self._image_topic,
            'has_image_data': self._image_data is not None,
            'last_image_time': self._last_image_time,
            'thread_running': self._thread_running,
            'image_callback_count': getattr(self, '_image_callback_count', 0),
            'depth_callback_count': getattr(self, '_depth_callback_count', 0)
        }
        if self._image_data is not None:
            status['image_shape'] = self._image_data.shape
        return status
    
    # Properties for backward compatibility
    @property
    def img_shape(self):
        return self._img_shape
    
    @property
    def fps(self):
        return self._fps
    
    @property
    def id(self):
        return self._image_topic
