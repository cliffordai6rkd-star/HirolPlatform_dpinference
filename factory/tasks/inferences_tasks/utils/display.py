"""
Image display utilities for inference tasks
"""

import numpy as np
import cv2
import glog as log


def calculate_grid_layout(num_images: int) -> tuple[int, int]:
    """Calculate optimal grid layout for images.

    Args:
        num_images: Number of images to display

    Returns:
        (rows, cols): Grid dimensions as close to square as possible
    """
    if num_images <= 0:
        return (0, 0)

    cols = int(np.ceil(np.sqrt(num_images)))
    rows = int(np.ceil(num_images / cols))
    return (rows, cols)


def create_image_grid(images_dict: dict[str, np.ndarray],
                        target_size: tuple[int, int] = (240, 320)) -> np.ndarray:
    """Create a grid of images for display.

    Args:
        images_dict: Dictionary of camera names to image arrays
        target_size: Target size for each image (height, width)

    Returns:
        Combined grid image

    Raises:
        ValueError: If images cannot be processed
    """
    if not images_dict:
        # Return a blank placeholder image
        placeholder = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(placeholder, "No Images", (80, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return placeholder

    num_images = len(images_dict)
    rows, cols = calculate_grid_layout(num_images)

    # Standardize all images
    processed_images = []
    for name, img in images_dict.items():
        try:
            # Ensure image is 3-channel BGR
            if len(img.shape) == 2:  # Grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

            # Resize to target size
            resized_img = cv2.resize(img, (target_size[1], target_size[0]),
                                    interpolation=cv2.INTER_LINEAR)

            # Add text label
            cv2.putText(resized_img, name, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            processed_images.append(resized_img)

        except cv2.error as e:
            log.warning(f"Failed to process image {name}: {e}")
            # Create placeholder
            placeholder = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
            cv2.putText(placeholder, f"Error: {name}", (10, target_size[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            processed_images.append(placeholder)

    # Fill remaining grid positions with blank images if needed
    total_positions = rows * cols
    while len(processed_images) < total_positions:
        blank = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
        processed_images.append(blank)

    # Create grid
    grid_height = rows * target_size[0]
    grid_width = cols * target_size[1]
    grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    for i, img in enumerate(processed_images):
        row = i // cols
        col = i % cols
        y_start = row * target_size[0]
        y_end = y_start + target_size[0]
        x_start = col * target_size[1]
        x_end = x_start + target_size[1]
        grid_image[y_start:y_end, x_start:x_end] = img

    return grid_image


def display_images(images_dict: dict[str, np.ndarray],
                   display_window_name: str) -> np.ndarray | None:
    """Display images in a unified OpenCV window.

    Args:
        images_dict: Dictionary of camera names to image arrays
        display_window_name: Name of the display window

    Returns:
        Combined grid image if successful, None if failed
    """
    try:
        grid_image = create_image_grid(images_dict)
        cv2.imshow(display_window_name, grid_image)
        cv2.waitKey(1)  # Non-blocking update
        return grid_image

    except cv2.error as e:
        log.warning(f"OpenCV display error: {e}")
        return None
    except ValueError as e:
        log.error(f"Image processing error: {e}")
        return None
    except Exception as e:
        log.error(f"Unexpected error in image display: {e}")
        return None