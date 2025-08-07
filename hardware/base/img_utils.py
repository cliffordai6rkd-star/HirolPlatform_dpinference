import cv2

def combine_image(img1:cv2.Mat, img2:cv2.Mat):
    height_diff = abs(img1.shape[0] - img2.shape[0])
    if img1.shape[0] > img2.shape[0]:
        img2 = cv2.copyMakeBorder(img2, 0, height_diff, 0, 0, cv2.BORDER_CONSTANT)
    else:
        img1 = cv2.copyMakeBorder(img1, 0, height_diff, 0, 0, cv2.BORDER_CONSTANT)

    # 将两张图像水平拼接
    concatenated_image = cv2.hconcat([img1, img2])
    return concatenated_image

