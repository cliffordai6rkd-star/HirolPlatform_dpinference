#!/usr/bin/env python3
"""
Paxini Network Sensor Demo Script

This script demonstrates the ZMQ network setup for Paxini tactile sensors:
1. Server mode: Streams data from local PaxiniSerialSensor via ZMQ
2. Client mode: Receives data as PaxiniNetworkSensor

Usage:
    # Terminal 1 (Server - computer with physical sensors):
    python test_network_demo.py --mode server --port /dev/ttyACM0

    # Terminal 2 (Client - remote computer):
    python test_network_demo.py --mode client --ip 192.168.1.100

For testing on single computer:
    # Terminal 1:
    python test_network_demo.py --mode server --port /dev/ttyACM0
    
    # Terminal 2:
    python test_network_demo.py --mode client --ip localhost
"""

import argparse
import time
import signal
import sys
import os

# Add parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paxini_serial_sensor import PaxiniSerialSensor
from paxini_network_sensor import PaxiniNetworkSensor, create_paxini_zmq_server


def run_server(port="/dev/ttyACM0", zmq_port=5556):
    """Run ZMQ server mode - streams sensor data over network"""
    
    print("=" * 60)
    print("PAXINI ZMQ SERVER MODE")
    print("=" * 60)
    print(f"Serial port: {port}")
    print(f"ZMQ port: {zmq_port}")
    print("Press Ctrl+C to stop")
    print()
    
    # Create serial sensor configuration
    config = {
        "communication_type": "serial",
        "taxel_nums": 120,
        "update_frequency": 100,
        "type": "paxini_tactile",
        "control_boxes": [
            {
                "port": port,
                "baudrate": 460800,
                "timeout": 1.0,
                "control_mode": 5,
                "sensors": [
                    {
                        "model": "GEN2-IP-L5325",
                        "connect_ids": [5, 6]  # Adjust as needed
                    }
                ]
            }
        ]
    }
    
    try:
        # Initialize serial sensor
        print("Initializing serial sensor...")
        sensor = PaxiniSerialSensor(config)
        
        if not sensor.is_sensor_connected():
            print("❌ Failed to connect to serial sensor")
            return 1
        
        print("✅ Serial sensor connected")
        
        # Print sensor info
        info = sensor.get_sensor_info()
        print(f"  Modules: {info['total_modules']}")
        print(f"  Control boxes: {info['control_boxes_count']}")
        
        # Start ZMQ server
        print(f"\nStarting ZMQ server on port {zmq_port}...")
        server_thread = create_paxini_zmq_server(
            sensor, 
            ip="*", 
            port=zmq_port,
            topic="PAXINI_TACTILE_STREAM"
        )
        
        print("✅ ZMQ server started")
        print("Broadcasting tactile data...")
        print("  Clients can connect with:")
        print(f"    python test_network_demo.py --mode client --ip <your_ip> --port {zmq_port}")
        print()
        
        # Main loop - show statistics
        start_time = time.time()
        last_stats_time = start_time
        
        while True:
            # Print statistics every 5 seconds
            current_time = time.time()
            if current_time - last_stats_time > 5.0:
                success, data, timestamp = sensor.read_tactile_data()
                if success and data is not None:
                    elapsed = current_time - start_time
                    max_force = abs(data).max()
                    mean_force = abs(data).mean()
                    print(f"Server stats - Running: {elapsed:.0f}s | "
                          f"Max force: {max_force:.1f} | "
                          f"Mean force: {mean_force:.2f}")
                last_stats_time = current_time
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        return 1
    finally:
        if 'sensor' in locals():
            sensor.close()
    
    return 0


def run_client(ip="localhost", zmq_port=5556, sensor_id="PAXINI_02_MODULES"):
    """Run ZMQ client mode - receives sensor data over network"""
    
    print("=" * 60)
    print("PAXINI ZMQ CLIENT MODE")
    print("=" * 60)
    print(f"Server IP: {ip}")
    print(f"ZMQ port: {zmq_port}")
    print(f"Sensor ID: {sensor_id}")
    print("Press Ctrl+C to stop")
    print()
    
    # Create network sensor configuration
    config = {
        "communication_type": "network",
        "taxel_nums": 120,
        "update_frequency": 100,
        "type": "paxini_tactile",
        "ip": ip,
        "port": zmq_port,
        "topic": "PAXINI_TACTILE_STREAM",
        "sensor_id": sensor_id,
        "timeout": 10.0
    }
    
    try:
        # Initialize network sensor
        print("Connecting to ZMQ server...")
        sensor = PaxiniNetworkSensor(config)
        
        if not sensor.is_sensor_connected():
            print("❌ Failed to connect to network sensor")
            print("  Check if:")
            print("    - Server is running")
            print("    - IP address is correct")
            print("    - Firewall allows connection")
            return 1
        
        print("✅ Network sensor connected")
        
        # Print sensor info
        info = sensor.get_sensor_info()
        print(f"  Sensor ID: {info['sensor_id']}")
        print(f"  Network: {info['ip']}:{info['port']}")
        print(f"  Topic: {info['topic']}")
        print(f"  Data shape: {info['data_shape']}")
        
        # Main receiving loop
        print("\nReceiving tactile data...")
        start_time = time.time()
        last_data_time = start_time
        data_count = 0
        
        while True:
            success, data, timestamp = sensor.read_tactile_data()
            
            if success and data is not None:
                data_count += 1
                last_data_time = time.time()
                
                # Print statistics every 5 seconds
                if data_count % 500 == 0:  # ~5s at 100Hz
                    elapsed = time.time() - start_time
                    rate = data_count / elapsed
                    max_force = abs(data).max()
                    mean_force = abs(data).mean()
                    
                    print(f"Client stats - Running: {elapsed:.0f}s | "
                          f"Rate: {rate:.1f}Hz | "
                          f"Max: {max_force:.1f} | "
                          f"Mean: {mean_force:.2f}")
            else:
                # Check for connection timeout
                if time.time() - last_data_time > 10.0:
                    print("⚠️  No data received for 10 seconds - connection may be lost")
                    last_data_time = time.time()
            
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\n🛑 Client stopped by user")
    except Exception as e:
        print(f"❌ Client error: {e}")
        return 1
    finally:
        if 'sensor' in locals():
            sensor.close()
    
    return 0


def main():
    parser = argparse.ArgumentParser(description="Paxini Network Sensor Demo")
    parser.add_argument("--mode", choices=["server", "client"], required=True,
                       help="Run as server (data source) or client (data receiver)")
    
    # Server options
    parser.add_argument("--port", default="/dev/ttyACM0",
                       help="Serial port for server mode")
    
    # Client options  
    parser.add_argument("--ip", default="localhost",
                       help="Server IP address for client mode")
    
    # Common options
    parser.add_argument("--zmq-port", type=int, default=5556,
                       help="ZMQ port number")
    parser.add_argument("--sensor-id", default="PAXINI_02_MODULES",
                       help="Sensor ID for client mode")
    
    args = parser.parse_args()
    
    # Setup signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\n🛑 Shutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run appropriate mode
    if args.mode == "server":
        return run_server(args.port, args.zmq_port)
    else:
        return run_client(args.ip, args.zmq_port, args.sensor_id)


if __name__ == "__main__":
    sys.exit(main())