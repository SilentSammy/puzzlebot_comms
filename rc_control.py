import numpy as np
import math
import cv2
import requests
import time
import threading
from keybrd import is_pressed, is_toggled, ToggleManager  # your keybrd module for key state detection
from pb_http_client import PuzzlebotHttpClient  # your custom client for sending commands
from pose_estimation import find_arucos, estimate_marker_pose
import pose_estimation as pe
from simple_pid import PID

# Keyboard
radio_buttons = ToggleManager(['1', '2', '3'])

# Connection
# puzzlebot = PuzzlebotHttpClient("http://192.168.137.139:5000")
puzzlebot = PuzzlebotHttpClient("http://127.0.0.1:5000", safe_mode=False)

# Maximum values for throttle and yaw
max_yaw = math.radians(60)
max_thr = 0.5

# Navigation algorithms
def navigate_to_marker(frame):
    if not hasattr(navigate_to_marker, "pids"):
        navigate_to_marker.pids = {
            'yaw_pid': PID(Kp=1, Ki=0, Kd=0.1, setpoint=0.0, output_limits=(-max_yaw, max_yaw)),
            'throttle_pid': PID(Kp=2, Ki=0, Kd=0.1, setpoint=0.3, output_limits=(-max_thr, max_thr))
        }
    yaw_pid = navigate_to_marker.pids['yaw_pid']
    thr_pid = navigate_to_marker.pids['throttle_pid']
    yaw_threshold = 5.0  # The robot will start moving forward when the target is this many degrees from the center

    markers, ids = find_arucos(frame)
    if markers:
        marker = markers[0]
        # Remove extra nesting if necessary; otherwise, pass marker directly.
        _, _, _, cam_dist, _, cam_yaw = estimate_marker_pose( marker_corners=marker, frame=frame, ref_size=0.063300535, fov_x=math.radians(60) )
        print(f"Dist: {cam_dist:.2f}, Yaw: {math.degrees(cam_yaw):.2f}")
        yaw = yaw_pid(cam_yaw)
        
        # Compute interpolation factor 'alpha'
        # alpha = 0 when yaw error is >= threshold, alpha = 1 when yaw error is 0.
        alpha = 1 - (abs(cam_yaw) / math.radians(yaw_threshold)) if abs(cam_yaw) < math.radians(yaw_threshold) else 0

        # Interpolate the distance input: when alpha is 0, measured distance is set to the setpoint.
        measured_distance = (1 - alpha) * thr_pid.setpoint + alpha * cam_dist
        throttle = -thr_pid(measured_distance)
    else:
        throttle, yaw = 0, 0

    return throttle, yaw

def follow_line(frame):
    # Static variables
    max_yaw = math.radians(30)
    max_thr = 0.1
    if not hasattr(follow_line, "yaw_pid"):
        follow_line.yaw_pid = PID(Kp=0.25, Ki=0, Kd=0.1, setpoint=0.0, output_limits=(-max_yaw, max_yaw))

    if frame is None:
        return 0, 0

    # Convert to binary mask
    frame_height, frame_width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 85, 255, cv2.THRESH_BINARY_INV)
    
    # Shrink the vertical field of view to the lower part of the frame.
    v_fov = 0.25
    mask[:int(frame_height * (1-v_fov)), :] = 0
    
    # Erode the mask slightly to disconnect noisy connections.
    mask = cv2.erode(mask, kernel=np.ones((3, 3), np.uint8), iterations=3)

    # Find contours in the modified mask, filtering out noise.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    MIN_AREA = 100 # Minimum area of contours to consider
    contours = [c for c in contours if cv2.contourArea(c) > MIN_AREA]
    
    throttle, yaw = 0, 0
    if contours:
        # Sort contours by: lowest (largest y+h), then center-most.
        def contour_key(c):
            x, y, w, h = cv2.boundingRect(c)
            bottom = y + h
            center = x + w/2
            center_err = abs(center - (frame_width/2))
            return (-bottom, center_err)
            
        sorted_contours = sorted(contours, key=contour_key)
        line_contour = sorted_contours[0]
        x, y, w, h = cv2.boundingRect(line_contour)
        
        # Draw the outline of line_contour on the original frame
        cv2.drawContours(frame, [line_contour], -1, (0, 255, 0), 2)

        # If the contour isn't touching the bottom of the frame, do nothing.
        if (y + h) >= int(frame_height * 0.80):
            # Get the X position of the cropped area using moments.
            M = cv2.moments(line_contour)
            center_x = int(M["m10"] / M["m00"]) if M["m00"] != 0 else x + w//2
            normalized_x = (center_x - (frame_width/2)) / (frame_width/2)
            print(f"Normalized center X (lower 20%): {normalized_x:.2f}")
            
            # Calculate the yaw and throttle values to follow the line.
            yaw = follow_line.yaw_pid(normalized_x)
            alignment = 1 - abs(normalized_x)
            align_thres = 0.6  # adjust as needed
            throttle = max_thr * ((alignment - align_thres) / (1 - align_thres)) if alignment >= align_thres else 0
    
    return throttle, yaw
 
def show_frame(frame, name="Frame", max_size=400):
    h, w = frame.shape[:2]
    if h > w:
        new_h = max_size
        new_w = int(w * (max_size / h))
    else:
        new_w = max_size
        new_h = int(h * (max_size / w))
    resized_frame = cv2.resize(frame, (new_w, new_h))
    cv2.imshow(name, resized_frame)
    cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)
    cv2.setWindowProperty(name, cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_NORMAL)  # Make the window resizable
    if cv2.waitKey(1) & 0xFF == 27:
        raise KeyboardInterrupt

try:
    while True:
        # Inputs and outputs
        frame = puzzlebot.get_frame()
        throttle, yaw = 0, 0

        # Safe mode selection
        puzzlebot.safe_mode = not is_toggled('z') # Safe by default

        # Control mode selection
        state = radio_buttons.get_active()
        if state == '2':
            throttle, yaw = navigate_to_marker(frame)
        elif state == '3':
            throttle, yaw = follow_line(frame)
        
        # Always allow manual control
        throttle += (1 if is_pressed('w') else -1 if is_pressed('s') else 0) * max_thr
        yaw += (1 if is_pressed('a') else -1 if is_pressed('d') else 0) * max_yaw

        # Send control commands to the robot
        puzzlebot.send_vel(throttle, yaw, wait_for_completion=False)
        
        if frame is not None:
            show_frame(frame, "Puzzlebot Stream", 400)
except KeyboardInterrupt:
    print("Exiting...")
finally:
    puzzlebot._stop_stream()
    cv2.destroyAllWindows()