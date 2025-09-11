#!/usr/bin/env python3
"""
Paxini Tactile Standalone Server

独立的Paxini触觉传感器ZMQ服务器，不依赖HIROLRobotPlatform平台代码。
支持配置文件，发送端和接收端配置一致。

Usage:
    # 使用默认配置
    python paxini_standalone_server.py

    # 使用配置文件
    python paxini_standalone_server.py --config server_config.yaml

    # 命令行参数
    python paxini_standalone_server.py --port /dev/ttyACM1 --zmq-port 5557
"""

import argparse
import signal
import sys
import time
import struct
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import serial
import zmq
import yaml


def extract_low_high_byte(d: int) -> List[int]:
    """Extract low and high byte from decimal number"""
    high_byte = (d >> 8) & 0xFF
    low_byte = d & 0xFF
    return [low_byte, high_byte]


def decimal_to_hex(d: int) -> int:
    """Convert decimal to hex (8-bit mask)"""
    return d & 0xFF


class PaxiniProtocol:
    """Paxini串口通信协议实现"""
    
    def __init__(self, serial_port: serial.Serial):
        self.ser = serial_port
    
    def _calculate_lrc(self, data: List[int]) -> int:
        """Calculate LRC checksum"""
        checksum = sum(data) & 0xFF
        lrc = (~checksum + 1) & 0xFF
        return lrc
    
    def _build_protocol(self, fix_id: int, index: int, main_cmd: int,
                       sub_cmd: List[int], length: List[int], data: List[int]) -> bytes:
        """Build protocol packet"""
        head = [0x55, 0xAA, 0x7B, 0x7B]
        tail = [0x55, 0xAA, 0x7D, 0x7D]
        
        lrc_packet = [fix_id, index, main_cmd] + sub_cmd + length + data
        lrc = self._calculate_lrc(lrc_packet)
        packet = head + [fix_id, index, main_cmd] + sub_cmd + length + data + [lrc] + tail
        return bytes(packet)
    
    def _check_response(self, response: bytes) -> bool:
        """Check protocol response"""
        if len(response) == 0:
            raise ValueError("Empty response")
        if response[9] != 0:  # Error code
            raise ValueError(f"Response error code: {hex(response[9])}")
        return True
    
    def _extract_data(self, response: bytes) -> bytes:
        """Extract data from protocol response"""
        return response[12:-5]
    
    def set_control_mode(self, mode: int) -> bool:
        """Set control box mode"""
        fix_id = 0x0E
        index = 0x00
        main_cmd = 0x70
        sub_cmd = [0xC0, 0x0C]
        length = [0x01, 0x00]
        data = [mode]
        
        packet = self._build_protocol(fix_id, index, main_cmd, sub_cmd, length, data)
        self.ser.write(packet)
        response = self.ser.read(17)
        return self._check_response(response)
    
    def set_module_port(self, con_id: int) -> bool:
        """Select module port"""
        fix_id = 0x0E
        index = 0x00
        main_cmd = 0x70
        sub_cmd = [0xB1, 0x0A]
        length = [0x01, 0x00]
        data = [decimal_to_hex((con_id - 1) * 3)]
        
        packet = self._build_protocol(fix_id, index, main_cmd, sub_cmd, length, data)
        self.ser.write(packet)
        response = self.ser.read(17)
        return self._check_response(response)
    
    def recalibrate_module(self, con_id: int) -> bool:
        """Recalibrate sensor module"""
        if not self.set_module_port(con_id):
            return False
        
        fix_id = 0x0E
        index = 0x00
        main_cmd = 0x70
        sub_cmd = [0xB0, 0x02]
        length = [0x02, 0x00]
        data = [0x03, 0x01]
        
        packet = self._build_protocol(fix_id, index, main_cmd, sub_cmd, length, data)
        self.ser.write(packet)
        response = self.ser.read(18)
        return len(response) != 0 and response[9] == 6
    
    def get_module_data(self, con_id: int) -> Optional[np.ndarray]:
        """Get tactile data from module"""
        try:
            if not self.set_module_port(con_id):
                return None
            
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
            self.ser.write(packet)
            response = self.ser.read(17 + 6 + num_bytes)
            self._check_response(response)
            
            response_data = self._extract_data(response)
            sensing_data = response_data[6:]
            sensing_array = np.array(list(map(int, sensing_data))).reshape(120, 3)
            
            # Convert x/y axis from 0~255 to -128~127
            xy = sensing_array[:, 0:2]
            xy = np.where(xy >= 128, xy - 256, xy)
            sensing_array[:, 0:2] = xy
            
            return sensing_array
            
        except Exception as e:
            print(f"Error reading module {con_id}: {e}")
            return None


class PaxiniStandaloneServer:
    """独立的Paxini ZMQ服务器 - 统一配置格式"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.running = False
        self.zmq_context = None
        self.zmq_socket = None
        
        # ZMQ configuration
        self.zmq_ip = config.get('ip', '*')
        self.zmq_port = config.get('port', 5556)
        self.zmq_topic = config.get('topic', 'PAXINI_TACTILE_STREAM')
        self.update_frequency = config.get('update_frequency', 100)
        
        # Parse control boxes configuration (same as network sensor)
        self._init_control_boxes(config)
        
        # Connection storage
        self.protocols = []

    def _init_control_boxes(self, config: Dict[str, Any]):
        """Initialize control box configuration (unified format only)"""
        control_boxes = config.get('control_boxes', [])
        
        if not control_boxes:
            raise ValueError("Server configuration must include 'control_boxes' array")
        
        # Parse control boxes and calculate total modules
        self.control_boxes = control_boxes
        total_modules = 0
        
        for i, box_config in enumerate(control_boxes):
            sensors = box_config.get("sensors", [])
            if not sensors:
                raise ValueError(f"Control box {i} requires 'sensors' list")
            
            # Count modules for this box
            box_modules = 0
            for sensor_config in sensors:
                connect_ids = sensor_config.get("connect_ids", [])
                box_modules += len(connect_ids)
            
            total_modules += box_modules
            print(f"📦 Control box {i}: {box_modules} modules")
        
        self.expected_total_modules = total_modules
        
        print(f"📋 Configuration: {len(control_boxes)} control boxes, {total_modules} total modules")
        print("🏷️  Per-box sensor_ids will be generated automatically")

    def initialize(self) -> bool:
        """Initialize all control boxes and ZMQ"""
        try:
            print("🚀 Initializing Paxini Standalone Server")
            print(f"  ZMQ: {self.zmq_ip}:{self.zmq_port}")
            print(f"  Topic: {self.zmq_topic}")
            print(f"  Broadcasting {len(self.control_boxes)} separate control box streams")
            
            # Initialize each control box
            for i, box_config in enumerate(self.control_boxes):
                success = self._init_control_box(i, box_config)
                if not success:
                    print(f"❌ Failed to initialize control box {i}")
                    self._cleanup_connections()
                    return False
            
            print(f"✅ All {len(self.control_boxes)} control boxes initialized")
            
            # Initialize ZMQ
            self.zmq_context = zmq.Context()
            self.zmq_socket = self.zmq_context.socket(zmq.PUB)
            self.zmq_socket.bind(f"tcp://{self.zmq_ip}:{self.zmq_port}")
            
            print("✅ Server initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            self._cleanup_connections()
            return False

    def _init_control_box(self, box_id: int, box_config: Dict[str, Any]) -> bool:
        """Initialize a single control box"""
        port = box_config.get('port', '/dev/ttyACM0')
        baudrate = box_config.get('baudrate', 460800)
        timeout = box_config.get('timeout', 1.0)
        control_mode = box_config.get('control_mode', 5)
        
        try:
            print(f"  📡 Connecting to control box {box_id} on {port}")
            
            # Open serial connection
            ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)
            protocol = PaxiniProtocol(ser)
            
            # Set control mode
            if not protocol.set_control_mode(control_mode):
                print(f"    ❌ Failed to set control mode {control_mode}")
                ser.close()
                return False
            
            # Calibrate all modules on this box
            sensors = box_config.get('sensors', [])
            all_connect_ids = []
            for sensor_config in sensors:
                connect_ids = sensor_config.get('connect_ids', [])
                all_connect_ids.extend(connect_ids)
            
            for con_id in all_connect_ids:
                success = protocol.recalibrate_module(con_id)
                print(f"    Module {con_id}: {'✅' if success else '❌'}")
            
            # Store successful connection
            self.protocols.append({
                'serial': ser,
                'protocol': protocol,
                'connect_ids': all_connect_ids,
                'box_id': box_id,
                'box_config': box_config
            })
            
            print(f"  ✅ Control box {box_id} initialized")
            return True
            
        except Exception as e:
            print(f"  ❌ Failed to initialize control box {box_id}: {e}")
            return False

    def _cleanup_connections(self):
        """Clean up all serial connections"""
        for protocol_info in self.protocols:
            try:
                if protocol_info['serial'].is_open:
                    protocol_info['serial'].close()
            except:
                pass
        self.protocols = []
    
    def pack_message(self, tactile_data: np.ndarray, timestamp_ms: int) -> bytes:
        """Pack tactile data into ZMQ message format"""
        message = bytearray()
        
        # Sensor count (1)
        message.extend(struct.pack('<i', 1))
        
        # Sensor ID
        sensor_id_bytes = self.sensor_id.encode('utf-8')
        message.extend(struct.pack('<i', len(sensor_id_bytes)))
        message.extend(sensor_id_bytes)
        
        # Timestamp
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
        
        return bytes(message)
    
    def pack_box_message(self, tactile_data: np.ndarray, timestamp_ms: int, sensor_id: str) -> bytes:
        """Pack tactile data into ZMQ message format for specific control box"""
        message = bytearray()
        
        # Sensor count (1)
        message.extend(struct.pack('<i', 1))
        
        # Sensor ID
        sensor_id_bytes = sensor_id.encode('utf-8')
        message.extend(struct.pack('<i', len(sensor_id_bytes)))
        message.extend(sensor_id_bytes)
        
        # Timestamp
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
        
        return bytes(message)
    
    def generate_box_sensor_id(self, box_config: Dict[str, Any], num_modules: int) -> str:
        """Generate sensor_id for specific control box"""
        # Use box name if provided, otherwise use box_id
        box_name = box_config.get("name", f"BOX{box_config.get('box_id', 0)}")
        return f"PAXINI_{box_name.upper()}_{num_modules:02d}_MODULES"
    
    def collect_data(self) -> Optional[Dict[int, np.ndarray]]:
        """Collect data from all control boxes, organized by box_id"""
        try:
            box_data = {}
            
            # Collect data from each control box separately
            for protocol_info in self.protocols:
                protocol = protocol_info['protocol']
                connect_ids = protocol_info['connect_ids']
                box_id = protocol_info['box_id']
                
                # Collect data from each module on this box
                box_module_data = []
                for con_id in connect_ids:
                    module_data = protocol.get_module_data(con_id)
                    if module_data is not None:
                        box_module_data.append(module_data)
                
                if box_module_data:
                    # Stack modules for this box: (n_modules_in_box, 120, 3)
                    box_data[box_id] = np.stack(box_module_data, axis=0)
            
            return box_data if box_data else None
            
        except Exception as e:
            print(f"❌ Data collection error: {e}")
            return None
    
    def run_server(self):
        """Main server loop"""
        print(f"🔄 Starting data broadcast at {self.update_frequency}Hz")
        print("Press Ctrl+C to stop\n")
        
        self.running = True
        dt = 1.0 / self.update_frequency
        next_time = time.perf_counter()
        stats_interval = 5.0
        last_stats = time.perf_counter()
        data_count = 0
        
        while self.running:
            try:
                loop_start = time.perf_counter()
                
                # Collect tactile data from all boxes
                box_data = self.collect_data()
                
                if box_data is not None:
                    timestamp_ms = int(time.time() * 1000)
                    
                    # Send separate message for each control box
                    for box_id, tactile_data in box_data.items():
                        # Find box config for this box_id
                        box_config = None
                        for protocol_info in self.protocols:
                            if protocol_info['box_id'] == box_id:
                                box_config = protocol_info['box_config']
                                break
                        
                        if box_config is not None:
                            # Generate box-specific sensor_id
                            num_modules = tactile_data.shape[0]
                            box_sensor_id = self.generate_box_sensor_id(box_config, num_modules)
                            
                            # Pack and send message for this box
                            message = self.pack_box_message(tactile_data, timestamp_ms, box_sensor_id)
                            self.zmq_socket.send_multipart([
                                self.zmq_topic.encode('utf-8'),
                                message
                            ])
                    
                    data_count += 1
                
                # Print statistics
                current_time = time.perf_counter()
                if current_time - last_stats >= stats_interval:
                    if box_data is not None:
                        # Calculate stats across all boxes
                        all_data = np.concatenate(list(box_data.values()), axis=0)
                        max_force = abs(all_data).max()
                        mean_force = abs(all_data).mean()
                        rate = data_count / (current_time - last_stats + stats_interval)
                        
                        print(f"📊 Stats - Rate: {rate:.1f}Hz | "
                              f"Max: {max_force:.1f} | "
                              f"Mean: {mean_force:.2f} | "
                              f"Boxes: {len(box_data)}")
                    data_count = 0
                    last_stats = current_time
                
                # Timing control
                next_time += dt
                sleep_time = next_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_time = time.perf_counter()
                
            except KeyboardInterrupt:
                print("\n🛑 Stopping server...")
                break
            except Exception as e:
                print(f"❌ Server error: {e}")
                time.sleep(0.1)
        
        self.running = False
    
    def close(self):
        """Cleanup all resources"""
        print("🔧 Cleaning up...")
        
        if self.zmq_socket:
            self.zmq_socket.close()
        if self.zmq_context:
            self.zmq_context.term()
        
        # Clean up all serial connections
        self._cleanup_connections()
        
        print("✅ Server stopped")


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file or use defaults"""
    # Default unified configuration (matches network sensor format)
    default_config = {
        'update_frequency': 100,
        'ip': '*',  # ZMQ bind IP
        'port': 5556,  # ZMQ port
        'topic': 'PAXINI_TACTILE_STREAM',
        'sensor_id': 'PAXINI_02_MODULES',
        'control_boxes': [
            {
                'port': '/dev/ttyACM0',
                'baudrate': 460800,
                'timeout': 1.0,
                'control_mode': 5,
                'sensors': [
                    {
                        'model': 'GEN2-IP-L5325',
                        'connect_ids': [5, 6]
                    }
                ]
            }
        ]
    }
    
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
            default_config.update(file_config)
            print(f"📄 Loaded config: {config_path}")
        except Exception as e:
            print(f"⚠️  Config load failed: {e}, using defaults")
    
    return default_config


def create_example_config():
    """Create example configuration files"""
    base_path = Path(__file__).parent
    
    # Server config (unified format)
    server_config = {
        'update_frequency': 100,
        'ip': '*',
        'port': 5556,
        'topic': 'PAXINI_TACTILE_STREAM',
        'sensor_id': 'PAXINI_02_MODULES',
        'control_boxes': [
            {
                'port': '/dev/ttyACM0',
                'baudrate': 460800,
                'timeout': 1.0,
                'control_mode': 5,
                'sensors': [
                    {
                        'model': 'GEN2-IP-L5325',
                        'connect_ids': [5, 6]
                    }
                ]
            }
        ]
    }
    
    # Client config (unified format) 
    client_config = {
        'communication_type': 'network',
        'taxel_nums': 120,
        'update_frequency': 100,
        'type': 'paxini_tactile',
        'ip': '192.168.1.100',  # Server IP
        'port': 5556,
        'topic': 'PAXINI_TACTILE_STREAM',
        'timeout': 10.0,
        'control_boxes': [
            {
                'sensors': [
                    {
                        'model': 'GEN2-IP-L5325',
                        'connect_ids': [5, 6]
                    }
                ]
            }
        ]
    }
    
    # Multi-box server config
    multibox_server_config = {
        'update_frequency': 100,
        'ip': '*',
        'port': 5556,
        'topic': 'PAXINI_TACTILE_STREAM',
        'sensor_id': 'PAXINI_04_MODULES',
        'control_boxes': [
            {
                'port': '/dev/ttyACM0',
                'baudrate': 460800,
                'timeout': 1.0,
                'control_mode': 5,
                'sensors': [
                    {
                        'model': 'GEN2-IP-L5325',
                        'connect_ids': [5, 6]
                    }
                ]
            },
            {
                'port': '/dev/ttyACM1',
                'baudrate': 460800,
                'timeout': 1.0,
                'control_mode': 5,
                'sensors': [
                    {
                        'model': 'GEN2-IP-L5325',
                        'connect_ids': [5, 6]
                    }
                ]
            }
        ]
    }
    
    # Write files
    try:
        with open(base_path / 'paxini_server_config.yaml', 'w') as f:
            yaml.dump(server_config, f, default_flow_style=False, sort_keys=False)
        
        with open(base_path / 'paxini_client_config.yaml', 'w') as f:
            yaml.dump(client_config, f, default_flow_style=False, sort_keys=False)
        
        with open(base_path / 'paxini_multibox_server_config.yaml', 'w') as f:
            yaml.dump(multibox_server_config, f, default_flow_style=False, sort_keys=False)
        
        print("✅ Created unified config files:")
        print("  - paxini_server_config.yaml (single control box)")
        print("  - paxini_client_config.yaml (network sensor)")  
        print("  - paxini_multibox_server_config.yaml (multiple control boxes)")
        
    except Exception as e:
        print(f"❌ Failed to create config files: {e}")


def main():
    parser = argparse.ArgumentParser(description="Paxini Standalone Server")
    parser.add_argument('--config', help="Configuration file path")
    parser.add_argument('--port', default='/dev/ttyACM0', help="Serial port")
    parser.add_argument('--zmq-port', type=int, default=5556, help="ZMQ port")
    parser.add_argument('--create-config', action='store_true',
                       help="Create example config files")
    
    args = parser.parse_args()
    
    # Create example configs if requested
    if args.create_config:
        create_example_config()
        return 0
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line args
    if args.port != '/dev/ttyACM0':
        config['serial_port'] = args.port
    if args.zmq_port != 5556:
        config['port'] = args.zmq_port
    
    # Create and run server
    server = PaxiniStandaloneServer(config)
    
    # Signal handler
    def signal_handler(sig, frame):
        print("\n🛑 Received shutdown signal")
        server.running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if not server.initialize():
            return 1
        
        server.run_server()
        
    except Exception as e:
        print(f"❌ Server error: {e}")
        return 1
    finally:
        server.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())