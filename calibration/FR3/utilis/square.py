import pyrealsense2 as rs
import numpy as np
import cv2

def start_realsense_camera():
    # 创建管道
    pipeline = rs.pipeline()
    
    # 创建配置并配置管道以进行流式传输
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # 开始流式传输
    pipeline.start(config)
    
    return pipeline

def capture_image(pipeline):
    # 等待连贯的帧
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    
    # 将图像转换为numpy数组
    color_image = np.asanyarray(color_frame.get_data())
    
    return color_image

def center_crop_image(color_frame, crop_size=(640, 640)):
    """中心裁剪彩色帧"""
    # 获取原始尺寸
    h, w = color_frame.shape[:2]
    new_w, new_h = crop_size
    
    # 确保裁剪尺寸不大于原始图像
    new_w = min(new_w, w)
    new_h = min(new_h, h)
    
    # 计算裁剪区域
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    right = left + new_w
    bottom = top + new_h
    print(f"裁剪区域: 上={top}, 下={bottom}, 左={left}, 右={right}")

    # 裁剪彩色帧（HWC格式）
    cropped_color = color_frame[top:bottom, left:right]
    
    return cropped_color, left, top  # 返回裁剪后的图像和偏移量

def detect_square(image):
    # 创建原始图像的副本用于显示
    display_image = image.copy()
    
    # 转换为灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 应用高斯模糊以减少噪音
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 使用Canny检测边缘
    edges = cv2.Canny(blurred, 50, 150)
    
    # 找到轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    found_squares = []
    
    # 检查每个轮廓
    for contour in contours:
        # 近似轮廓
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 如果轮廓有4个顶点，它可能是一个正方形
        if len(approx) == 4:
            # 获取边界矩形
            x, y, w, h = cv2.boundingRect(approx)
            
            # 检查它是否近似为正方形（宽高比接近1）
            aspect_ratio = float(w) / h
            if 0.9 <= aspect_ratio <= 1.1:
                # 绘制正方形轮廓
                cv2.drawContours(display_image, [approx], 0, (0, 255, 0), 3)
                
                # 计算中心点
                M = cv2.moments(approx)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = x + w//2, y + h//2
                
                # 添加标签
                cv2.putText(display_image, "Square", (cX - 30, cY), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 显示正方形的尺寸
                size_text = f"{w}x{h} pixels"
                cv2.putText(display_image, size_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                found_squares.append(approx)
    
    # 在图像上显示找到的正方形数量
    cv2.putText(display_image, f"Found {len(found_squares)} squares", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    return len(found_squares) > 0, found_squares, display_image

def main():
    try:
        # 启动相机
        print("启动RealSense相机...")
        pipeline = start_realsense_camera()
        
        # 等待几帧，让自动曝光稳定
        for _ in range(5):
            pipeline.wait_for_frames()
        
        # 捕获图像
        print("捕获图像...")
        original_image = capture_image(pipeline)
        
        # 中心裁剪图像
        print("中心裁剪图像...")
        crop_size = (480, 480)  # 可以根据需要调整裁剪尺寸
        cropped_image, offset_x, offset_y = center_crop_image(original_image, crop_size)
        
        # 保存原始和裁剪后的图像
        cv2.imwrite('original_image.jpg', original_image)
        cv2.imwrite('cropped_image.jpg', cropped_image)
        print(f"保存了原始图像和裁剪后的图像，裁剪偏移量: x={offset_x}, y={offset_y}")
        
        # 检测正方形
        print("检测正方形...")
        found_square, square_contours, result_image = detect_square(cropped_image)
        
        if found_square:
            print(f"检测到 {len(square_contours)} 个正方形!")
            # 保存结果图像
            cv2.imwrite('detected_squares.jpg', result_image)
            print("结果已保存为 'detected_squares.jpg'")
            
            # 显示原始图像和结果图像
            cv2.imshow('原始图像', original_image)
            cv2.imshow('裁剪后的图像', cropped_image)
            cv2.imshow('检测到的正方形', result_image)
            print("按任意键关闭窗口...")
            cv2.waitKey(0)
        else:
            print("未检测到正方形.")
            # 显示图像
            cv2.imshow('原始图像', original_image)
            cv2.imshow('裁剪后的图像', cropped_image)
            print("按任意键关闭窗口...")
            cv2.waitKey(0)
        
    finally:
        # 停止流式传输
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()