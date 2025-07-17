import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time

class ImgSaver(Node):
    def __init__(self):
        super().__init__('image_saver')
        self.bridge = CvBridge()
       
        # Subscription for depth image
        self.depth_subscription = self.create_subscription(
            Image,
            '/camera_head_front/depth/stream',
            # '/camera/depth_to_color/image_raw',
            # '/camera_head_front/color/image_raw',
            self.depth_callback,
            10)
           
        # Subscription for color image
        self.color_subscription = self.create_subscription(
            Image,
            '/camera_head_front/color/video',
            self.color_callback,
            10)

    def depth_callback(self, msg):
        try:
            # Convert the ROS Image message to a CV2 Image for depth
            cv_image_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            # Save the depth image
            cv2.imwrite('/home/ubuntu/demo_monte_grasp/receive_ros2/data/depth.png', cv_image_depth)
            time.sleep(1)
            self.get_logger().info('Depth image saved to depth.png')
            # time.sleep(1)
        except Exception as e:
            self.get_logger().error(f'Failed to save depth image: {e}')

    def color_callback(self, msg):
        try:
            # Convert the ROS Image message to a CV2 Image for color
            cv_image_color = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # Save the color image
            cv2.imwrite('/home/ubuntu/demo_monte_grasp/receive_ros2/data/color.png', cv_image_color)
            time.sleep(1)
            self.get_logger().info('Color image saved to color.png')
            # time.sleep(1)
        except Exception as e:
            self.get_logger().error(f'Failed to save color image: {e}')

def main(args=None):
    rclpy.init(args=args)
    image_saver = ImgSaver()
    rclpy.spin(image_saver)
    image_saver.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import cv2
# import os
# from message_filters import ApproximateTimeSynchronizer, Subscriber
# import time


# class ImgSaver(Node):
#     def __init__(self):
#         super().__init__('image_saver')
#         self.bridge = CvBridge()
        
#         # 创建可配置的保存路径参数
#         self.declare_parameter('save_path', '/home/hanyu/monte_pick_and_place/receive_ros2/data')
#         self.save_path = self.get_parameter('save_path').get_parameter_value().string_value
        
#         # 确保保存目录存在
#         os.makedirs(self.save_path, exist_ok=True)
        
#         # 使用消息过滤器进行同步订阅
#         self.depth_sub = Subscriber(self, Image, '/camera_head_front/depth/image_raw')
#         self.color_sub = Subscriber(self, Image, '/camera_head_front/color/image_raw')
        
#         # 创建同步器，队列大小为10，时间同步容差为0.1秒
#         self.sync = ApproximateTimeSynchronizer(
#             [self.depth_sub, self.color_sub], 10, 1)
#         self.sync.registerCallback(self.synchronized_callback)
        
#         self.get_logger().info(f'图像将保存到: {self.save_path}')

#     def synchronized_callback(self, depth_msg, color_msg):
#         print(f'已保存同步图像，时间戳: {time.time()}')
#         return

#         try:
#             t1 = time.time()
#             # 转换深度图像
#             cv_image_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
#             # 转换彩色图像
#             cv_image_color = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            
#             # 保存图像
#             depth_path = os.path.join(self.save_path, 'depth.png')
#             color_path = os.path.join(self.save_path, 'color.png')
            
#             # cv2.imwrite(depth_path, cv_image_depth)
#             # cv2.imwrite(color_path, cv_image_color)
            
#             # self.get_logger().info(f'已保存同步图像，时间戳: {depth_msg.header.stamp.sec}.{depth_msg.header.stamp.nanosec}')
#             print(f'已保存同步图像，时间戳: {time.time()-t1}')
#         except cv2.error as e:
#             self.get_logger().error(f'OpenCV错误: {e}')
#         except Exception as e:
#             self.get_logger().error(f'保存图像失败: {e}')

# def main(args=None):
#     rclpy.init(args=args)
#     image_saver = ImgSaver()
#     rclpy.spin(image_saver)
#     image_saver.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()
