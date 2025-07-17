import pyrealsense2 as rs
import numpy as np
import cv2
import time

def circle_detection():
    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable color stream - highest resolution for better detection
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    
    # Start streaming
    profile = pipeline.start(config)
    
    # Get stream profile and camera intrinsics
    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    color_intrinsics = color_profile.get_intrinsics()
    
    # Print camera info
    print(f"Camera: {profile.get_device().get_info(rs.camera_info.name)}")
    print(f"Resolution: {color_intrinsics.width}x{color_intrinsics.height}")
    
    # Variables for FPS calculation
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    # Create named windows
    cv2.namedWindow('Original (640x640)', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Circle Detection', cv2.WINDOW_NORMAL)
    
    # Create trackbars for parameter tuning
    cv2.createTrackbar('Min Radius', 'Circle Detection', 10, 100, lambda x: None)
    cv2.createTrackbar('Max Radius', 'Circle Detection', 100, 200, lambda x: None)
    cv2.createTrackbar('Param1', 'Circle Detection', 100, 300, lambda x: None)
    cv2.createTrackbar('Param2', 'Circle Detection', 30, 100, lambda x: None)
    
    try:
        while True:
            # Start timing this frame
            frame_start_time = time.time()
            
            # Wait for a coherent frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
                
            # Convert image to numpy array
            color_image = np.asanyarray(color_frame.get_data())
            
            # Get dimensions of the original image
            height, width = color_image.shape[:2]
            
            # Crop image to 640x640 from the center
            start_y = (height - 640) // 2
            start_x = (width - 640) // 2
            cropped_image = color_image[start_y:start_y+640, start_x:start_x+640]
            
            # Make a copy of cropped image for displaying original
            original_display = cropped_image.copy()
            
            # Convert to grayscale for circle detection
            gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            
            # Apply some blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)
            
            # Get current trackbar values
            min_radius = cv2.getTrackbarPos('Min Radius', 'Circle Detection')
            max_radius = cv2.getTrackbarPos('Max Radius', 'Circle Detection')
            param1 = cv2.getTrackbarPos('Param1', 'Circle Detection')
            param2 = cv2.getTrackbarPos('Param2', 'Circle Detection')
            
            # Detect circles using Hough Circle Transform with current parameters
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,                   # Resolution ratio of accumulator to image
                minDist=50,             # Minimum distance between detected circles
                param1=param1,          # Higher threshold for Canny edge detector
                param2=param2,          # Threshold for circle detection
                minRadius=min_radius,   # Minimum radius of circles
                maxRadius=max_radius    # Maximum radius of circles
            )
            
            # Make a color copy for drawing results
            result_image = cropped_image.copy()
            
            # Draw a frame to show the detection area
            cv2.rectangle(color_image, (start_x, start_y), (start_x+640, start_y+640), (0, 255, 0), 2)
            
            # Create a blank image for status display
            status_image = np.zeros((640, 320, 3), dtype=np.uint8)
            
            # Add FPS and parameter info to status image
            cv2.putText(status_image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(status_image, f"Min Radius: {min_radius}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(status_image, f"Max Radius: {max_radius}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(status_image, f"Param1: {param1}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(status_image, f"Param2: {param2}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Circle detection results
            circle_count = 0
            if circles is not None:
                # Convert to integer coordinates
                circles = np.uint16(np.around(circles))
                circle_count = len(circles[0])
                
                for i, circle in enumerate(circles[0, :]):
                    # Get circle parameters
                    center_x, center_y, radius = circle
                    
                    # Draw the outer circle
                    cv2.circle(result_image, (center_x, center_y), radius, (0, 255, 0), 2)
                    # Draw the center of the circle
                    cv2.circle(result_image, (center_x, center_y), 2, (0, 0, 255), 3)
                    
                    # Add circle number
                    cv2.putText(result_image, f"#{i+1}", (center_x+10, center_y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                    # Add circle info to status display
                    y_pos = 180 + i * 60
                    if y_pos < 620:  # Ensure we don't go beyond status image
                        cv2.putText(status_image, f"Circle #{i+1}:", (10, y_pos), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(status_image, f"  Center: ({center_x}, {center_y})", (10, y_pos+30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                        cv2.putText(status_image, f"  Radius: {radius}", (10, y_pos+60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            # Display circle count
            cv2.putText(status_image, f"Circles Detected: {circle_count}", (10, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Combine result image and status display
            combined_result = np.hstack((result_image, status_image))
            
            # Display the original and result
            cv2.imshow('Original (640x640)', original_display)
            cv2.imshow('Circle Detection', combined_result)
            
            # Calculate FPS
            frame_count += 1
            if frame_count >= 10:
                end_time = time.time()
                fps = frame_count / (end_time - start_time)
                frame_count = 0
                start_time = time.time()
            
            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Camera stopped and resources released")

if __name__ == "__main__":
    circle_detection()