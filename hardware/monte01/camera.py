from hardware.base.camera import CameraBase

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

class Camera(CameraBase, Node):
    def __init__(self, config={}):
        rclpy.init(args=None)

        # Explicitly initialize both parent classes
        CameraBase.__init__(self, config)
        Node.__init__(self, 'camera_node')
        
        self.get_logger().info("影像顯示節點已啟動。")

        # 2. 設定 QoS Profile
        # 對於感測器數據（如攝影機），通常使用 Best Effort 和較小的歷史深度
        # 這樣可以確保我們處理的是最新的數據，即使有時會丟失一些影格
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # 3. 創建 CvBridge 實例
        # CvBridge 是 ROS Image 訊息和 OpenCV 影像之間轉換的橋樑
        self.bridge = CvBridge()

        # 4. 創建訂閱者 (Subscriber)
        self.subscription = self.create_subscription(
            Image,  # 訊息類型
            '/camera_head_front/color/image_raw',   
            # '/camera_head_front/color/video',  # todo:check these 
            # '/camera_head_front/depth/stream',  # Topic 名稱 
            self.image_callback,  # 收到訊息時要呼叫的回呼函式
            qos_profile  # 使用我們定義的 QoS 設定
        )
        self.subscription  # 避免 "unused variable" 警告

    def initialize(self):
        pass
    def close(self):
        pass
    def image_callback(self, msg: Image):
        """
        訂閱者的回呼函式，每當收到新的影像訊息時就會被執行。
        """
        self.get_logger().debug(f"接收到影像: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")
        try:
            # 5. 使用 cv_bridge 將 ROS Image 訊息轉換為 OpenCV 影像格式
            # "bgr8" 是 OpenCV 常用的顏色編碼
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge 轉換錯誤: {e}")
            return

        # 6. 使用 OpenCV 顯示影像
        cv2.imshow("ROS 2 Image Viewer", cv_image)
        
        # 7. 等待按鍵事件，這是讓視窗保持更新所必需的
        # 等待 1 毫秒，如果使用者按下 'q' 鍵，則準備關閉
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info("偵測到 'q' 鍵，準備關閉...")
            # 觸發節點關閉
            self.destroy_node()
            rclpy.shutdown()
    