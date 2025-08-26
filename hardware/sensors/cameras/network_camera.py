from hardware.base.camera import CameraBase
import threading
import warnings
import copy
import time
import struct

import zmq
import numpy as np
import cv2
import lz4.frame

def unpack_bundled_message(message_data):
    """
    解析從 ZMQ 接收到的打包訊息。
    與參考程式碼中的函式功能相同。
    """
    cameras = {}
    offset = 0
    
    try:
        # 解析消息头
        sync_group_count = struct.unpack('<i', message_data[offset:offset+4])[0]
        offset += 4
        
        camera_count = struct.unpack('<i', message_data[offset:offset+4])[0]
        offset += 4
        
        # 解析每个相机的数据
        for _ in range(camera_count):
            # 序列号长度和内容
            serial_len = struct.unpack('<i', message_data[offset:offset+4])[0]
            offset += 4
            serial = message_data[offset:offset+serial_len].decode('utf-8')
            offset += serial_len
            
            # 时间戳
            timestamp = struct.unpack('<q', message_data[offset:offset+8])[0]
            offset += 8
            
            # 彩色数据
            color_len = struct.unpack('<i', message_data[offset:offset+4])[0]
            offset += 4
            color_data = message_data[offset:offset+color_len]
            offset += color_len
            
            # 深度数据
            depth_len = struct.unpack('<i', message_data[offset:offset+4])[0]
            offset += 4
            depth_data = message_data[offset:offset+depth_len]
            offset += depth_len
            
            # 將解析出的資料存入字典
            cameras[serial] = {
                "color_data": color_data,
                "depth_data": depth_data,
                "timestamp": timestamp
            }
            
    except Exception as e:
        print(f"Error unpacking bundled message: {e}")
        return None
    
    return cameras

class NetworkCamera(CameraBase):
    """
    透過 ZMQ SUB Socket 從網路串流接收攝影機資料的實作。
    (已修改以處理打包訊息)
    """
    def __init__(self, config):
        # Network specific config
        self._ip = config['ip']
        self._port = config['port']
        self._topic = config.get('topic', 'D435i_STREAM')
        self._serial_number = config.get('serial_number') # Crucial for identifying the stream

        if not self._serial_number:
            raise ValueError("Configuration must include a 'serial_number'.")

        # ZMQ and Threading members
        self._context = None
        self._subscriber = None
        self._thread = None
        self._running = False
        
        print(f"Initializing NetworkCamera for SN: {self._serial_number}...")
        super().__init__(config)

    def initialize(self):
        """
        設定 ZMQ 連線並啟動背景執行緒來接收資料。
        """
        try:
            self._context = zmq.Context()
            self._subscriber = self._context.socket(zmq.SUB)
            connect_str = f"tcp://{self._ip}:{self._port}"
            self._subscriber.connect(connect_str)
            self._subscriber.setsockopt_string(zmq.SUBSCRIBE, self._topic)
            print(f"Subscriber for SN {self._serial_number} connected to {connect_str}")
            
            self._running = True
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

            # 等待第一幀資料來確認初始化成功
            timeout = time.time() + 5  # 5 秒超時
            while self._image_data is None and self._running and time.time() < timeout:
                time.sleep(0.1)
            
            if self._image_data is None:
                warnings.warn(f"Failed to receive first frame for SN {self._serial_number} within 5s.")
                self.close()
                return False

            print(f"NetworkCamera for SN: {self._serial_number} initialized successfully.")
            return True

        except Exception as e:
            print(f"Error initializing NetworkCamera for SN {self._serial_number}: {e}")
            self.close() # 初始化失敗時也應該清理資源
            return False

    def _run(self):
        """
        在背景執行緒中運行的主循環，用於接收和處理 ZMQ 打包訊息。
        """
        poller = zmq.Poller()
        poller.register(self._subscriber, zmq.POLLIN)

        while self._running:
            socks = dict(poller.poll(timeout=100))
            if self._subscriber in socks and socks[self._subscriber] == zmq.POLLIN:
                try:
                    # 接收打包消息 (topic, bundled_data)
                    topic, bundled_data = self._subscriber.recv_multipart()

                    # 解析打包的訊息，這會返回一個包含所有攝影機資料的字典
                    all_cameras_data = unpack_bundled_message(bundled_data)

                    if all_cameras_data is None or self._serial_number not in all_cameras_data:
                        continue # 如果解析失敗或此幀不包含本機資料，則跳過

                    # 根據序列號提取本攝影機的資料
                    camera_data = all_cameras_data[self._serial_number]

                    # 解碼彩色圖像
                    color_frame = cv2.imdecode(np.frombuffer(camera_data["color_data"], dtype=np.uint8), cv2.IMREAD_COLOR)
                    
                    # 解碼深度圖像
                    depth_frame = None
                    if self._contain_depth and camera_data["depth_data"]:
                        try:
                            h, w = self._img_shape
                            decompressed_depth = lz4.frame.decompress(camera_data["depth_data"])
                            depth_frame = np.frombuffer(decompressed_depth, dtype=np.uint16).reshape(h, w)
                        except Exception as e:
                            print(f"Error decompressing depth frame for {self._serial_number}: {e}")
                            depth_frame = None
                    
                    # 使用鎖安全地更新資料
                    if color_frame is not None:
                        with self._lock:
                            self._image_data = color_frame
                            self._depth_map_data = depth_frame
                            self._timestamp = camera_data["timestamp"]

                except zmq.ZMQError as e:
                    if e.errno == zmq.ETERM:
                        break # Context was terminated, exit loop
                    else:
                        print(f"ZMQ Error for SN {self._serial_number}: {e}")
                except Exception as e:
                    print(f"An error occurred in receiving thread for SN {self._serial_number}: {e}")
                    time.sleep(1)

    def close(self):
        """
        停止背景執行緒並關閉 ZMQ socket。
        """
        if self._running:
            print(f"Closing NetworkCamera for SN: {self._serial_number}...")
            self._running = False
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=2)
            
            if self._subscriber is not None:
                self._subscriber.close()
            
            if self._context is not None:
                self._context.term()
            
            self._subscriber = None
            self._context = None
            print(f"NetworkCamera for SN: {self._serial_number} closed.")


if __name__ == "__main__":
    # --- 攝影機設定 ---
    SERVER_IP = "192.168.252.82"
    
    config1 = {
        'ip': SERVER_IP,
        'port': 5555,
        'topic': 'D435i_STREAM',
        'serial_number': '243222076185',
        'image_shape': [480, 640],
        'contain_depth': True,
    }

    config2 = {
        'ip': SERVER_IP,
        'port': 5555,
        'topic': 'D435i_STREAM',
        'serial_number': '243322073850',
        'image_shape': [480, 640],
        'contain_depth': True,
    }

    # --- 初始化攝影機 ---
    cam1 = NetworkCamera(config1)
    cam2 = NetworkCamera(config2)

    # 檢查是否初始化成功
    if not cam1.is_initialized() or not cam2.is_initialized():
        print("One or more cameras failed to initialize. Exiting.")
        cam1.close()
        cam2.close()
        exit()

    try:
        while True:
            # --- 讀取資料 ---
            success1, image1 = cam1.read_image()
            success2, image2 = cam2.read_image()
            
            # --- 處理和顯示 ---
            display_frames = []
            
            if success1 and image1 is not None:
                latency1 = int(time.time() * 1000) - cam1.get_timestamp()
                cv2.putText(image1, f"SN: {cam1._serial_number}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(image1, f"Latency: {latency1}ms", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                display_frames.append(image1)
            else:
                h, w = config1['image_shape']
                placeholder = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(placeholder, f"Waiting for SN: {config1['serial_number']}", (w//4, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                display_frames.append(placeholder)

            if success2 and image2 is not None:
                latency2 = int(time.time() * 1000) - cam2.get_timestamp()
                cv2.putText(image2, f"SN: {cam2._serial_number}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(image2, f"Latency: {latency2}ms", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                display_frames.append(image2)
            else:
                h, w = config2['image_shape']
                placeholder = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(placeholder, f"Waiting for SN: {config2['serial_number']}", (w//4, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                display_frames.append(placeholder)

            # 將兩台攝影機的畫面垂直堆疊起來
            final_view = np.vstack(display_frames)
            cv2.imshow('Dual Network Camera Stream', final_view)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("'q' pressed, exiting...")
                break
    
    finally:
        # --- 清理資源 ---
        print("Cleaning up resources...")
        cam1.close()
        cam2.close()
        cv2.destroyAllWindows()
        print("Done.")