#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

class ImageViewerNode(Node):
    """
    一個簡單的 ROS 2 節點，用來訂閱影像 topic 並使用 OpenCV 顯示。
    """
    def __init__(self):
        # 1. 初始化節點
        super().__init__('image_viewer_node')
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
            '/camera_head_front/color/image_raw',  # Topic 名稱
            self.image_callback,  # 收到訊息時要呼叫的回呼函式
            qos_profile  # 使用我們定義的 QoS 設定
        )
        self.subscription  # 避免 "unused variable" 警告

    def image_callback(self, msg: Image):
        """
        訂閱者的回呼函式，每當收到新的影像訊息時就會被執行。
        """
        self.get_logger().debug(f"接收到影像: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")
        print(f"...")
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

def main(args=None):
    try:
        # 初始化 rclpy
        rclpy.init(args=args)
        
        # 創建節點實例
        image_viewer_node = ImageViewerNode()
        
        # 進入 spin 模式，等待回呼函式被觸發
        # 當 destroy_node() 被呼叫時，spin 會結束
        rclpy.spin(image_viewer_node)

    except KeyboardInterrupt:
        print("偵測到 Ctrl+C，正在關閉節點...")
    finally:
        # 在結束時確保資源被釋放
        if 'image_viewer_node' in locals() and rclpy.ok():
            image_viewer_node.destroy_node()
            rclpy.shutdown()
        cv2.destroyAllWindows()
        print("程式已乾淨地關閉。")

if __name__ == '__main__':
    main()