import time
import threading
import struct
from typing import Dict, Any, Optional
import numpy as np
import zmq
import glog as log
from hardware.base.tactile_base import TactileBase


def unpack_paxini_message(message_data: bytes) -> Optional[Dict[str, Any]]:
    """
    Unpack ZMQ message containing Paxini tactile data.
    
    Message format:
    - sensor_count (4 bytes)
    - For each sensor:
      - sensor_id_length (4 bytes)
      - sensor_id (variable bytes)
      - timestamp (8 bytes)
      - data_shape_dims (4 bytes) - number of dimensions
      - data_shape (dims * 4 bytes) - shape of each dimension
      - data_length (4 bytes) - total data bytes
      - tactile_data (variable bytes) - flattened numpy array
    
    Args:
        message_data: Raw ZMQ message bytes
        
    Returns:
        Dict with sensor data or None if parsing fails
    """
    try:
        sensors = {}
        offset = 0
        
        # Parse sensor count
        sensor_count = struct.unpack('<i', message_data[offset:offset+4])[0]
        offset += 4
        
        # Parse each sensor's data
        for _ in range(sensor_count):
            # Sensor ID length and content
            sensor_id_length = struct.unpack('<i', message_data[offset:offset+4])[0]
            offset += 4
            sensor_id = message_data[offset:offset+sensor_id_length].decode('utf-8')
            offset += sensor_id_length
            
            # Timestamp
            timestamp = struct.unpack('<q', message_data[offset:offset+8])[0]
            offset += 8
            
            # Data shape
            shape_dims = struct.unpack('<i', message_data[offset:offset+4])[0]
            offset += 4
            shape = []
            for _ in range(shape_dims):
                dim = struct.unpack('<i', message_data[offset:offset+4])[0]
                shape.append(dim)
                offset += 4
            
            # Tactile data
            data_length = struct.unpack('<i', message_data[offset:offset+4])[0]
            offset += 4
            data_bytes = message_data[offset:offset+data_length]
            offset += data_length
            
            # Reconstruct numpy array
            tactile_data = np.frombuffer(data_bytes, dtype=np.int32).reshape(shape)
            
            sensors[sensor_id] = {
                "tactile_data": tactile_data,
                "timestamp": timestamp,
                "shape": shape
            }
            
        return sensors
        
    except Exception as e:
        log.error(f"Error unpacking Paxini message: {e}")
        return None


class PaxiniNetworkSensor(TactileBase):
    """
    Paxini tactile sensor implementation using ZMQ network communication.
    
    Handles exactly ONE control box receiving data over network.
    For multiple control boxes, use multiple PaxiniNetworkSensor instances.
    
    Configuration:
        communication_type: network
        ip: "192.168.1.100"        # Server IP
        port: 5556                 # ZMQ port  
        topic: "PAXINI_TACTILE_STREAM"
        control_box:               # Single control box
          sensors:
            - model: "GEN2-IP-L5325"
              connect_ids: [5, 6]
    
    Remote setup:
        Computer A: Paxini hardware → Server → ZMQ Publisher  
        Computer B: PaxiniNetworkSensor ← ZMQ Subscriber ← Network
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Network-specific configuration
        self._ip = config.get("ip", "localhost")
        self._port = config.get("port", 5556)
        self._topic = config.get("topic", "PAXINI_TACTILE_STREAM")
        self._timeout = config.get("timeout", 5.0)
        self._reconnect_interval = config.get("reconnect_interval", 1.0)
        
        # Parse control box configuration (single box only)
        self._init_control_box(config)
        
        # ZMQ and threading components
        self._context = None
        self._subscriber = None
        self._thread = None
        self._running = False
        
        log.info(f"Initializing PaxiniNetworkSensor")
        log.info(f"  Network: {self._ip}:{self._port}")
        log.info(f"  Topic: {self._topic}")
        log.info(f"  Expected sensor ID: {self._sensor_id}")
        log.info(f"  Expected modules: {self._expected_total_modules}")
        
        # Initialize base class
        super().__init__(config)
        
        log.info(f"PaxiniNetworkSensor initialized successfully")

    def _init_control_box(self, config: Dict[str, Any]):
        """Initialize single control box configuration"""
        self._control_box = config.get("control_box", {})
        
        if not self._control_box:
            raise ValueError("Network configuration must include 'control_box' configuration")
        
        # Parse control box and calculate total modules
        sensors = self._control_box.get("sensors", [])
        if not sensors:
            raise ValueError("Control box requires 'sensors' list")
        
        # Count modules for this box
        total_modules = 0
        for sensor_config in sensors:
            connect_ids = sensor_config.get("connect_ids", [])
            total_modules += len(connect_ids)
        
        self._expected_total_modules = total_modules
        
        # Generate sensor_id
        self._sensor_id = config.get("sensor_id", f"PAXINI_{total_modules:02d}_MODULES")
        
        log.info(f"Parsed configuration: 1 control box, {total_modules} total modules")

    def initialize(self) -> bool:
        """Initialize ZMQ connection and start background thread"""
        # Check if already initialized (avoid double initialization)
        if hasattr(self, '_is_initialized') and self._is_initialized:
            log.info("Paxini network sensor already initialized, skipping...")
            return True
            
        try:
            # Setup ZMQ context and subscriber
            self._context = zmq.Context()
            self._subscriber = self._context.socket(zmq.SUB)
            connect_str = f"tcp://{self._ip}:{self._port}"
            self._subscriber.connect(connect_str)
            self._subscriber.setsockopt_string(zmq.SUBSCRIBE, self._topic)
            
            # Set socket timeout
            self._subscriber.setsockopt(zmq.RCVTIMEO, int(self._timeout * 1000))
            
            log.info(f"ZMQ subscriber connected to {connect_str}")
            log.info(f"Subscribed to topic: {self._topic}")
            log.info(f"Listening for sensor ID: {self._sensor_id}")
            
            # Start background receiving thread
            self._running = True
            self._thread = threading.Thread(target=self._receive_loop, daemon=True)
            self._thread.start()
            
            # Wait for first data to confirm connection
            timeout_start = time.time()
            while self._tactile_data is None and self._running:
                if time.time() - timeout_start > self._timeout:
                    log.warning(f"No data received within {self._timeout}s timeout")
                    self.close()
                    return False
                time.sleep(0.1)
            
            if self._tactile_data is not None:
                log.info(f"PaxiniNetworkSensor initialized successfully")
                log.info(f"Received data shape: {self._tactile_data.shape}")
                return True
            else:
                log.error(f"Failed to receive initial data")
                self.close()
                return False
                
        except Exception as e:
            log.error(f"Error initializing PaxiniNetworkSensor: {e}")
            self.close()
            return False

    def _receive_loop(self):
        """Main receiving loop running in background thread"""
        poller = zmq.Poller()
        poller.register(self._subscriber, zmq.POLLIN)
        
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        log.info("Starting Paxini network receive loop")
        
        while self._running:
            try:
                # Poll for messages with timeout
                socks = dict(poller.poll(timeout=1000))  # 1 second timeout
                
                if self._subscriber in socks and socks[self._subscriber] == zmq.POLLIN:
                    # Receive message
                    topic, message_data = self._subscriber.recv_multipart()
                    
                    # Parse bundled message
                    sensors_data = unpack_paxini_message(message_data)
                    
                    if sensors_data is None:
                        log.warning("Failed to parse received message")
                        continue
                    
                    # Check if our sensor ID is in the message
                    if self._sensor_id not in sensors_data:
                        # This message doesn't contain our sensor data
                        continue
                    
                    # Extract our sensor's data
                    sensor_data = sensors_data[self._sensor_id]
                    tactile_data = sensor_data["tactile_data"]
                    timestamp = sensor_data["timestamp"] / 1000.0  # Convert to seconds
                    
                    # Update data in thread-safe manner
                    self._lock.acquire()
                    self._tactile_data = tactile_data.copy()
                    self._time_stamp = timestamp
                    self._lock.release()
                    
                    # Reset error counter on successful receive
                    consecutive_errors = 0
                
            except zmq.ZMQError as e:
                if e.errno == zmq.ETERM:
                    # Context was terminated, exit loop
                    break
                elif e.errno == zmq.EAGAIN:
                    # Timeout, continue polling
                    continue
                else:
                    consecutive_errors += 1
                    log.error(f"ZMQ error in receive loop: {e}")
                    
            except Exception as e:
                consecutive_errors += 1
                log.error(f"Error in receive loop: {e}")
            
            # Handle consecutive errors
            if consecutive_errors >= max_consecutive_errors:
                log.error(f"Too many consecutive errors ({consecutive_errors}), attempting reconnect...")
                self._attempt_reconnect()
                consecutive_errors = 0
                time.sleep(self._reconnect_interval)
        
        log.info("Paxini network receive loop stopped")

    def _attempt_reconnect(self):
        """Attempt to reconnect to ZMQ server"""
        try:
            log.info("Attempting to reconnect...")
            
            # Close existing subscriber
            if self._subscriber:
                self._subscriber.close()
            
            # Create new subscriber
            self._subscriber = self._context.socket(zmq.SUB)
            connect_str = f"tcp://{self._ip}:{self._port}"
            self._subscriber.connect(connect_str)
            self._subscriber.setsockopt_string(zmq.SUBSCRIBE, self._topic)
            self._subscriber.setsockopt(zmq.RCVTIMEO, int(self._timeout * 1000))
            
            log.info("Reconnection successful")
            
        except Exception as e:
            log.error(f"Reconnection failed: {e}")

    def close(self) -> bool:
        """Close ZMQ connection and stop background thread"""
        log.info("Closing PaxiniNetworkSensor...")
        
        # Stop background thread
        if self._running:
            self._running = False
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=2.0)
        
        # Close ZMQ components
        try:
            if self._subscriber:
                self._subscriber.close()
                self._subscriber = None
            
            if self._context:
                self._context.term()
                self._context = None
                
        except Exception as e:
            log.error(f"Error closing ZMQ components: {e}")
        
        log.info("PaxiniNetworkSensor closed")
        return True

    def get_sensor_info(self) -> Dict[str, Any]:
        """Get network-specific sensor information"""
        info = super().get_sensor_info()
        
        # Add network-specific information
        info.update({
            "communication_type": "network",
            "ip": self._ip,
            "port": self._port,
            "topic": self._topic,
            "sensor_id": self._sensor_id,
            "timeout": self._timeout,
            "is_receiving": self._running and self._thread and self._thread.is_alive()
        })
        
        # Add control box information
        sensors = self._control_box.get("sensors", [])
        total_modules = sum(len(s.get("connect_ids", [])) for s in sensors)
        
        box_summary = {
            "network_source": f"{self._ip}:{self._port}",  # Network equivalent of port
            "sensor_models": [s.get("model", "Unknown") for s in sensors],
            "total_modules": total_modules
        }
        
        info.update({
            "total_modules": total_modules,
            "box_summary": box_summary
        })
        
        return info

    def print_state(self) -> None:
        """Print network sensor state"""
        super().print_state()
        
        if self._is_initialized:
            log.info(f"  Network: {self._ip}:{self._port}")
            log.info(f"  Topic: {self._topic}")
            log.info(f"  Sensor ID: {self._sensor_id}")
            log.info(f"  Receiving: {self._running and self._thread and self._thread.is_alive()}")


def create_paxini_zmq_server(serial_sensor, ip="*", port=5556, topic="PAXINI_TACTILE_STREAM"):
    """
    Create ZMQ server to broadcast Paxini tactile data over network.
    
    This function should be run on the remote computer with physical Paxini sensors.
    
    Args:
        serial_sensor: PaxiniSerialSensor instance
        ip: Bind IP address ("*" for all interfaces)
        port: ZMQ port
        topic: ZMQ topic name
    """
    import threading
    
    def server_loop():
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.bind(f"tcp://{ip}:{port}")
        
        log.info(f"Paxini ZMQ server started on {ip}:{port}")
        log.info(f"Broadcasting topic: {topic}")
        
        try:
            while True:
                # Get data from serial sensor
                success, tactile_data, timestamp = serial_sensor.read_tactile_data()
                
                if success and tactile_data is not None:
                    # Pack message
                    sensor_info = serial_sensor.get_sensor_info()
                    sensor_id = f"PAXINI_{sensor_info['total_modules']:02d}_MODULES"
                    
                    # Create message
                    message = bytearray()
                    
                    # Sensor count (1 for single sensor)
                    message.extend(struct.pack('<i', 1))
                    
                    # Sensor ID
                    sensor_id_bytes = sensor_id.encode('utf-8')
                    message.extend(struct.pack('<i', len(sensor_id_bytes)))
                    message.extend(sensor_id_bytes)
                    
                    # Timestamp (convert to milliseconds)
                    timestamp_ms = int(timestamp * 1000)
                    message.extend(struct.pack('<q', timestamp_ms))
                    
                    # Data shape
                    shape = tactile_data.shape
                    message.extend(struct.pack('<i', len(shape)))
                    for dim in shape:
                        message.extend(struct.pack('<i', dim))
                    
                    # Tactile data
                    data_bytes = tactile_data.astype(np.int32).tobytes()
                    message.extend(struct.pack('<i', len(data_bytes)))
                    message.extend(data_bytes)
                    
                    # Send message
                    socket.send_multipart([topic.encode('utf-8'), bytes(message)])
                
                time.sleep(0.01)  # 100Hz broadcast
                
        except KeyboardInterrupt:
            log.info("ZMQ server stopped by user")
        except Exception as e:
            log.error(f"ZMQ server error: {e}")
        finally:
            socket.close()
            context.term()
    
    # Start server in background thread
    server_thread = threading.Thread(target=server_loop, daemon=True)
    server_thread.start()
    log.info("ZMQ server thread started")
    return server_thread