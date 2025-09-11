import os
import time
import threading
from typing import List, Dict, Any, Optional
import numpy as np
import serial
import yaml
import glog as log
from hardware.base.tactile_base import TactileBase


def extract_low_high_byte(d: int) -> List[int]:
    """
    Extract low and high byte from decimal number
    
    Args:
        d: decimal_number
    """
    high_byte = (d >> 8) & 0xFF
    low_byte = d & 0xFF
    return [low_byte, high_byte]


def decimal_to_hex(d: int) -> int:
    """Convert decimal to hex (8-bit mask)"""
    h = d & 0xFF
    return h


class PaxiniSerialSensor(TactileBase):
    """
    Paxini tactile sensor implementation using serial communication.
    
    Supports:
    - Single control box with multiple sensor types
    - Multiple control boxes
    - All Paxini sensor models (GEN1/GEN2 series)
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Paxini-specific configuration
        self._communication_type = config.get("communication_type", "serial")
        
        # Model-specific configuration mapping (from protocol documentation)
        self._model_config_map = {
            "GEN1-IP-S2516": {"port_offset": 0, "control_mode": 2},
            "GEN1-DP-S2716": {"port_offset": 1, "control_mode": 2},
            "GEN2-IP-L5325": {"port_offset": 0, "control_mode": 5},
            "GEN2-IP-M3025": {"port_offset": 0, "control_mode": 5},
            "GEN2-MP-M2324": {"port_offset": 1, "control_mode": 5},
            "GEN2-DP-L3530": {"port_offset": 2, "control_mode": 5},
            "GEN2-DP-M2826": {"port_offset": 2, "control_mode": 5},
            "GEN2-DP-S2716": {"port_offset": 0, "control_mode": 1},
        }
        
        # Initialize control box configurations
        self._init_control_boxes(config)
        
        # Initialize base class (this calls initialize())
        super().__init__(config)
        
        # Start data acquisition thread AFTER initialization completes
        self._running = False
        self._data_thread = None
        if self._is_initialized:
            self._start_data_acquisition()

    def _init_control_boxes(self, config: Dict[str, Any]):
        """Initialize control box configurations with validation"""
        self._control_boxes = config.get("control_boxes", [])
        
        if not self._control_boxes:
            raise ValueError("Serial communication requires 'control_boxes' list")
        
        # Validate each control box configuration
        for i, box_config in enumerate(self._control_boxes):
            if "port" not in box_config:
                raise ValueError(f"Control box {i} missing 'port' configuration")
            
            # Set defaults
            box_config.setdefault("baudrate", 460800)
            box_config.setdefault("timeout", 1.0)
            box_config.setdefault("control_mode", 5)
            
            # Validate sensors
            sensors = box_config.get("sensors", [])
            if not sensors:
                raise ValueError(f"Control box {i} requires 'sensors' list")
            
            # Validate each sensor
            control_mode = box_config["control_mode"]
            for j, sensor_config in enumerate(sensors):
                sensor_model = sensor_config.get("model")
                if not sensor_model:
                    raise ValueError(f"Control box {i}, sensor {j} missing 'model'")
                
                # Check compatibility
                if sensor_model not in self._model_config_map:
                    raise ValueError(f"Unsupported sensor model: {sensor_model}")
                
                expected_mode = self._model_config_map[sensor_model]["control_mode"]
                if expected_mode != control_mode:
                    raise ValueError(
                        f"Control box {i}: Sensor {sensor_model} requires mode {expected_mode}, "
                        f"but box is configured for mode {control_mode}"
                    )
                
                # Auto-set port offset
                expected_port_offset = self._model_config_map[sensor_model]["port_offset"]
                sensor_config["port_offset"] = expected_port_offset
                sensor_config.setdefault("connect_ids", [1])

    def initialize(self) -> bool:
        """Initialize all control boxes and sensors"""
        # Check if already initialized (avoid double initialization)
        if hasattr(self, '_is_initialized') and self._is_initialized:
            log.info("Paxini serial sensors already initialized, skipping...")
            return True
            
        log.info("Initializing Paxini serial sensors...")
        
        self._serial_connections = []
        
        try:
            # Initialize each control box
            for i, box_config in enumerate(self._control_boxes):
                success = self._init_control_box(i, box_config)
                if not success:
                    log.error(f"Failed to initialize control box {i}")
                    self._cleanup_connections()
                    return False
            
            log.info(f"Successfully initialized {len(self._control_boxes)} control boxes")
            
            # Pre-fetch initial data to ensure connection works
            log.info("Pre-fetching initial data...")
            initial_data = self._collect_all_sensor_data()
            if initial_data is not None:
                self._tactile_data = initial_data
                self._time_stamp = time.perf_counter()
                log.info(f"Initial data acquired, shape: {initial_data.shape}")
            else:
                log.warning("Could not acquire initial data, but initialization succeeded")
            
            return True
            
        except Exception as e:
            log.error(f"Error initializing Paxini sensors: {e}")
            self._cleanup_connections()
            return False

    def _init_control_box(self, box_id: int, box_config: Dict[str, Any]) -> bool:
        """Initialize a single control box"""
        port = box_config["port"]
        baudrate = box_config["baudrate"]
        timeout = box_config["timeout"]
        control_mode = box_config["control_mode"]
        
        try:
            # Open serial connection
            ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)
            self._serial_connections.append(ser)
            
            log.info(f"Opened serial connection to {port}")
            
            # Initialize control box
            version = self._get_control_box_version(ser)
            log.info(f"Control box {box_id} version: {version}")
            
            # Set control box mode
            success = self._set_control_box_mode(ser, control_mode)
            if not success:
                log.error(f"Failed to set control box {box_id} mode to {control_mode}")
                return False
            
            # Verify mode setting
            actual_mode = self._get_control_box_mode(ser)
            log.info(f"Control box {box_id} mode set to: {actual_mode}")
            
            # Recalibrate all sensors on this box
            sensors = box_config.get("sensors", [])
            for sensor_config in sensors:
                connect_ids = sensor_config.get("connect_ids", [])
                for connect_id in connect_ids:
                    success = self._recalibrate_sensor_module(ser, connect_id)
                    log.info(f"Calibrated module {connect_id}: {'✅' if success else '❌'}")
            
            log.info(f"Control box {box_id} initialized successfully")
            return True
            
        except serial.SerialException as e:
            log.error(f"Error opening serial port {port}: {e}")
            log.error(f"Try running: 'sudo chmod 666 {port}'")
            return False
        except Exception as e:
            log.error(f"Error initializing control box {box_id}: {e}")
            return False

    def _cleanup_connections(self):
        """Clean up serial connections"""
        for ser in getattr(self, '_serial_connections', []):
            try:
                if ser.is_open:
                    ser.close()
            except:
                pass
        self._serial_connections = []

    def _start_data_acquisition(self):
        """Start background data acquisition thread"""
        if self._data_thread is not None and self._data_thread.is_alive():
            return
            
        self._running = True
        self._data_thread = threading.Thread(target=self._data_acquisition_loop, daemon=True)
        self._data_thread.start()
        log.info("Started Paxini data acquisition thread")

    def _data_acquisition_loop(self):
        """Main data acquisition loop running in background thread"""
        while self._running:
            try:
                # Collect data from all control boxes
                all_data = self._collect_all_sensor_data()
                
                if all_data is not None:
                    # Update tactile data in thread-safe manner
                    self._lock.acquire()
                    self._tactile_data = all_data
                    self._time_stamp = time.perf_counter()
                    self._lock.release()
                
                # Sleep based on update frequency
                time.sleep(1.0 / self._update_frequency)
                
            except Exception as e:
                log.error(f"Error in data acquisition loop: {e}")
                time.sleep(0.1)

    def _collect_all_sensor_data(self) -> Optional[np.ndarray]:
        """Collect data from all sensor modules across all control boxes"""
        try:
            all_module_data = []
            
            # Iterate through all control boxes
            for box_id, box_config in enumerate(self._control_boxes):
                ser = self._serial_connections[box_id]
                sensors = box_config.get("sensors", [])
                
                # Collect data from all sensors on this box
                for sensor_config in sensors:
                    connect_ids = sensor_config.get("connect_ids", [])
                    for connect_id in connect_ids:
                        module_data = self._get_module_sensing_data(ser, connect_id)
                        if module_data is not None:
                            all_module_data.append(module_data)
            
            if all_module_data:
                # Stack all module data: shape = (n_modules, 120, 3)
                return np.stack(all_module_data, axis=0)
            else:
                return None
                
        except Exception as e:
            log.error(f"Error collecting sensor data: {e}")
            return None

    def close(self) -> bool:
        """Close all serial connections and stop data acquisition"""
        log.info("Closing Paxini serial sensor...")
        
        # Stop data acquisition thread
        self._running = False
        if self._data_thread and self._data_thread.is_alive():
            self._data_thread.join(timeout=2.0)
        
        # Close serial connections
        self._cleanup_connections()
        
        log.info("Paxini serial sensor closed")
        return True

    # ========== Paxini Protocol Implementation ==========
    
    def _calculate_lrc(self, data: List[int]) -> int:
        """Calculate LRC checksum for protocol"""
        checksum = sum(data) & 0xFF
        lrc = (~checksum + 1) & 0xFF
        return lrc

    def _build_protocol(self, fix_id: int, index: int, main_cmd: int, 
                       sub_cmd: List[int], length: List[int], data: List[int]) -> bytes:
        """Build Paxini protocol packet"""
        head = [0x55, 0xAA, 0x7B, 0x7B]
        tail = [0x55, 0xAA, 0x7D, 0x7D]
        
        lrc_packet = [fix_id, index, main_cmd] + sub_cmd + length + data
        lrc = self._calculate_lrc(lrc_packet)
        packet = head + [fix_id, index, main_cmd] + sub_cmd + length + data + [lrc]
        packet += tail
        return bytes(packet)

    def _check_response(self, response: bytes) -> bool:
        """Check protocol response for errors"""
        if len(response) == 0:
            raise ValueError("Empty response")
        if response[9] != 0:  # Error code
            raise ValueError(f"Response error code: {hex(response[9])}")
        return True

    def _extract_data(self, response: bytes) -> bytes:
        """Extract data from protocol response"""
        return response[12:-5]

    def _get_control_box_version(self, ser: serial.Serial) -> str:
        """Get control box version string"""
        fix_id = 0x0E
        index = 0x00
        main_cmd = 0x60
        sub_cmd = [0xA0, 0x01]
        length = [0x00, 0x00]
        data = []
        
        packet = self._build_protocol(fix_id, index, main_cmd, sub_cmd, length, data)
        ser.write(packet)
        time.sleep(0.01)
        
        response = ser.read(ser.in_waiting)
        self._check_response(response)
        return self._extract_data(response).decode("utf-8", errors="ignore")

    def _set_control_box_mode(self, ser: serial.Serial, mode: int) -> bool:
        """Set control box mode"""
        fix_id = 0x0E
        index = 0x00
        main_cmd = 0x70
        sub_cmd = [0xC0, 0x0C]
        length = [0x01, 0x00]
        data = [mode]
        
        packet = self._build_protocol(fix_id, index, main_cmd, sub_cmd, length, data)
        ser.write(packet)
        
        response = ser.read(17)
        return self._check_response(response)

    def _get_control_box_mode(self, ser: serial.Serial) -> str:
        """Get current control box mode"""
        fix_id = 0x0E
        index = 0x00
        main_cmd = 0x70
        sub_cmd = [0xC0, 0x0D]
        length = [0x00, 0x00]
        data = []
        
        packet = self._build_protocol(fix_id, index, main_cmd, sub_cmd, length, data)
        ser.write(packet)
        
        response = ser.read(18)
        self._check_response(response)
        return self._extract_data(response).hex()

    def _set_module_port(self, ser: serial.Serial, con_id: int) -> bool:
        """Select module port for communication"""
        fix_id = 0x0E
        index = 0x00
        main_cmd = 0x70
        sub_cmd = [0xB1, 0x0A]
        length = [0x01, 0x00]
        data = [decimal_to_hex((con_id - 1) * 3)]
        
        packet = self._build_protocol(fix_id, index, main_cmd, sub_cmd, length, data)
        ser.write(packet)
        
        response = ser.read(17)
        return self._check_response(response)

    def _recalibrate_sensor_module(self, ser: serial.Serial, con_id: int) -> bool:
        """Recalibrate a single sensor module"""
        # First select the module port
        if not self._set_module_port(ser, con_id):
            return False
        
        # Then recalibrate
        fix_id = 0x0E
        index = 0x00
        main_cmd = 0x70
        sub_cmd = [0xB0, 0x02]
        length = [0x02, 0x00]
        data = [0x03, 0x01]
        
        packet = self._build_protocol(fix_id, index, main_cmd, sub_cmd, length, data)
        ser.write(packet)
        
        response = ser.read(18)
        # Note: Error code 6 is expected for recalibration success per protocol
        return len(response) != 0 and response[9] == 6

    def _get_module_sensing_data(self, ser: serial.Serial, con_id: int) -> Optional[np.ndarray]:
        """Get sensing data from a single module"""
        try:
            # Select module port
            if not self._set_module_port(ser, con_id):
                return None
            
            # Read sensing data
            fix_id = 0x0E
            index = 0x00
            main_cmd = 0x70
            sub_cmd = [0xC0, 0x06]
            length = [0x05, 0x00]
            addr_begin = 1038
            addr_end = 1397
            num_bytes = addr_end - addr_begin + 1
            data = [0x7B] + extract_low_high_byte(addr_begin) + extract_low_high_byte(num_bytes)
            
            packet = self._build_protocol(fix_id, index, main_cmd, sub_cmd, length, data)
            ser.write(packet)
            
            response = ser.read(17 + 6 + num_bytes)
            self._check_response(response)
            
            # Extract and process sensing data
            response_data = self._extract_data(response)
            sensing_data = response_data[6:]
            sensing_data_array = np.array(list(map(int, sensing_data))).reshape(120, 3)
            
            # Convert x/y axis from 0~255 to -128~127 (two's complement)
            xy = sensing_data_array[:, 0:2]
            xy = np.where(xy >= 128, xy - 256, xy)
            sensing_data_array[:, 0:2] = xy
            
            return sensing_data_array
            
        except Exception as e:
            log.error(f"Error reading module {con_id} data: {e}")
            return None

    def get_sensor_info(self) -> Dict[str, Any]:
        """Get Paxini-specific sensor information"""
        info = super().get_sensor_info()
        
        # Add Paxini-specific information
        box_summary = []
        total_modules = 0
        for i, box_config in enumerate(self._control_boxes):
            sensors = box_config.get("sensors", [])
            box_modules = sum(len(s.get("connect_ids", [])) for s in sensors)
            total_modules += box_modules
            
            box_summary.append({
                "box_id": i,
                "port": box_config["port"],
                "control_mode": box_config["control_mode"],
                "sensor_models": [s["model"] for s in sensors],
                "total_modules": box_modules
            })
        
        info.update({
            "communication_type": "serial",
            "control_boxes_count": len(self._control_boxes),
            "total_modules": total_modules,
            "box_summary": box_summary
        })
        
        return info