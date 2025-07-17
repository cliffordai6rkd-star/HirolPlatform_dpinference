import cv2
import pyrealsense2 as rs
import numpy as np

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        color_frame = param['color_frame']
        hsv_frame = param['hsv_frame']
        depth_frame = param['depth_frame']  # 获取深度帧
        
        # 获取当前帧的BGR和HSV图像
        color_frame = param['color_frame']
        hsv_frame = param['hsv_frame']
        
        # 获取颜色值（BGR格式）
        bgr_value = color_frame[y, x]
        hsv_value = hsv_frame[y, x]
        
        # 转换BGR为RGB格式显示
        rgb_value = (bgr_value[2], bgr_value[1], bgr_value[0])
        print(f"RGB值: R:{rgb_value[0]:3} G:{rgb_value[1]:3} B:{rgb_value[2]:3}")
        print(f"HSV值: H:{hsv_value[0]:3} S:{hsv_value[1]:3} V:{hsv_value[2]:3}")


        # 获取深度值（单位：米）
        depth_value = depth_frame.get_distance(x, y)  # 新增深度获取
        print(f"深度值: {depth_value:.3f} 米")

        # 在图像上显示数值
        display_frame = color_frame.copy()
        cv2.putText(display_frame, f"R:{rgb_value[0]} G:{rgb_value[1]} B:{rgb_value[2]}", 
           (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)  # 位置上移
        cv2.putText(display_frame, f"H:{hsv_value[0]} S:{hsv_value[1]} V:{hsv_value[2]}",
           (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)  # 位置上移
        cv2.putText(display_frame, f"Depth: {depth_value:.3f}m",  # 新增深度显示
           (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        cv2.imshow('Color Window', display_frame)

# 初始化RealSense
pipeline = rs.pipeline()
config = rs.config()
# 修改配置启用深度流
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 新增深度流配置

# 启动相机
pipeline.start(config)

# 创建OpenCV窗口
cv2.namedWindow('Color Window')
param = {'hsv_frame': None, 'color_frame': None}
cv2.setMouseCallback('Color Window', mouse_callback, param)

# 创建对齐对象（将深度对齐到彩色帧）
align_to = rs.stream.color
align = rs.align(align_to)  # 新增对齐对象

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)  # 对齐深度和彩色帧
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()  # 获取深度帧
        
        # 转换图像格式
        color_image = np.asanyarray(color_frame.get_data())
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        
        # 更新共享参数
        param['color_frame'] = color_image
        param['hsv_frame'] = hsv_image
        param['depth_frame'] = depth_frame  # 新增深度帧参数
        
        # 显示图像
        cv2.imshow('Color Window', color_image)
        
        # 退出机制
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()


