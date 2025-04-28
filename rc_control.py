import numpy as np
import math
import cv2
import requests
import time
import threading
from keybrd import is_pressed, is_toggled  # your keybrd module for key state detection
from pb_http_client import PuzzlebotHttpClient  # your custom client for sending commands
from pose_estimation import find_arucos, estimate_marker_pose
from simple_pid import PID

# Connection
# puzzlebot = PuzzlebotHttpClient("http://192.168.137.139:5000")
puzzlebot = PuzzlebotHttpClient("http://127.0.0.1:5000")

# Control
heading_pid = PID(Kp=1, Ki=0, Kd=0.1, setpoint=0.0)
heading_pid.output_limits = (-math.radians(60), math.radians(60))
distance_pid = PID(Kp=2, Ki=0, Kd=0.1, setpoint=0.3)
distance_pid.output_limits = (-0.2, 0.2)
yaw_threshold = 5.0 # The robot will start moving forward when the target is this many degrees from the center of the image

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
    if cv2.waitKey(1) & 0xFF == ord('q'):
        raise KeyboardInterrupt

try:
    # puzzlebot._start_stream()
    while True:
        frame = puzzlebot.get_frame()

        # Process control commands
        yaw_mag = math.radians(45)
        throttle_mag = 0.15
        if not is_toggled('m'): # Manual mode
            throttle = (1 if is_pressed('w') else -1 if is_pressed('s') else 0) * throttle_mag
            yaw = (1 if is_pressed('a') else -1 if is_pressed('d') else 0) * yaw_mag
        else: # Auto mode
            markers, ids = find_arucos(frame)
            if markers:
                marker = markers[0]
                _, _, _, cam_dist, _, cam_yaw = estimate_marker_pose(marker_corners=marker, frame=frame, ref_size=0.063300535, fov_x=math.radians(60))
                print(f"Dist: {cam_dist:.2f}, Yaw: {math.degrees(cam_yaw):.2f}")
                yaw = heading_pid(cam_yaw)
                factor = 1 - (abs(cam_yaw) / math.radians(yaw_threshold)) if abs(cam_yaw) < math.radians(yaw_threshold) else 0
                throttle = factor * (-distance_pid(cam_dist))
            else:
                throttle = 0
                yaw = 0
        
        # Send control commands to the robot
        puzzlebot.send_vel(throttle, yaw, wait_for_completion=True)

        if frame is not None:
            show_frame(frame, "Puzzlebot Stream", 400)
        
except KeyboardInterrupt:
    print("Exiting...")
finally:
    # Stop the video stream and release resources
    puzzlebot._stop_stream()
    cv2.destroyAllWindows()