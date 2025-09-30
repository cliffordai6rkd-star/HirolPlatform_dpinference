import cv2
import numpy as np

def combine_image(img1:cv2.Mat, img2:cv2.Mat):
    # 找到两张图像中较小的尺寸
    min_height = min(img1.shape[0], img2.shape[0])
    min_width = min(img1.shape[1], img2.shape[1])
    
    # 将两张图像都resize到较小的尺寸
    img1_resized = cv2.resize(img1, (min_width, min_height))
    img2_resized = cv2.resize(img2, (min_width, min_height))

    # 将两张图像水平拼接
    concatenated_image = cv2.hconcat([img1_resized, img2_resized])
    return concatenated_image

def combine_images_2x2_grid(image_list, target_size=(400, 300)):
    """
    将图像列表按2x2网格排列，保持原始比例，缺失图像用黑色填充
    
    Args:
        image_list: 图像列表，最多4个图像
        target_size: 每个网格单元的目标尺寸 (width, height)
    
    Returns:
        拼接后的2x2网格图像
    """
    # 确保最多4个图像
    images = image_list[:4] if len(image_list) > 4 else image_list
    
    # 创建4个网格位置的图像列表
    grid_images = []
    
    for i in range(4):
        if i < len(images) and images[i] is not None:
            # 缩放图像到目标尺寸，保持比例
            img = images[i]
            h, w = img.shape[:2]
            target_w, target_h = target_size
            
            # 计算缩放比例，保持原始比例
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            # 缩放图像
            resized_img = cv2.resize(img, (new_w, new_h))
            
            # 创建目标尺寸的黑色背景
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            
            # 计算居中位置
            start_y = (target_h - new_h) // 2
            start_x = (target_w - new_w) // 2
            
            # 将缩放后的图像放置到画布中心
            canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized_img
            grid_images.append(canvas)
        else:
            # 创建黑色填充图像
            black_img = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            grid_images.append(black_img)
    
    # 将4个图像组合成2x2网格
    top_row = cv2.hconcat([grid_images[0], grid_images[1]])
    bottom_row = cv2.hconcat([grid_images[2], grid_images[3]])
    combined_grid = cv2.vconcat([top_row, bottom_row])
    
    return combined_grid

