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

def show_frame(frame, name="Frame", width=400):
    resized_frame = cv2.resize(frame, (width, int(frame.shape[0] * (width / frame.shape[1]))))
    cv2.imshow(name, resized_frame)
    cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey(1) # So that the window is responsive

# Connection
puzzlebot = PuzzlebotHttpClient("http://192.168.137.139:5000")

# Store RC Car states
throttle, yaw = 0, 0
prev_throttle, prev_yaw = 0, 0

# Camera parameters
dist_coeffs = np.zeros((5, 1), dtype=np.float32)
camera_matrix = np.array([
    [1047.47,    0.0,   640.0],
    [   0.0,  729.04,  355.59],
    [   0.0,    0.0,    1.0 ]
], dtype=np.float32)

# Control
heading_pid = PID(Kp=1, Ki=0, Kd=0.1, setpoint=0.0)
heading_pid.output_limits = (-math.radians(60), math.radians(60))
distance_pid = PID(Kp=2, Ki=0, Kd=0.1, setpoint=0.3)
distance_pid.output_limits = (-0.2, 0.2)

try:
    puzzlebot.start_stream()
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
                # print(np.linalg.norm(marker[0][0] - marker[0][1]) / frame.shape[1])
                _, _, _, cam_dist, _, cam_yaw = estimate_marker_pose(marker_corners=marker, frame=frame, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, ref_size=0.063300535, fov_x=1, fov_y=1)
                print(f"Dist: {cam_dist:.2f}, Yaw: {math.degrees(cam_yaw):.2f}")
                yaw = heading_pid(cam_yaw)
                yaw_threshold = 10.0
                factor = max(0, 1 - (abs(cam_yaw) / yaw_threshold))
                throttle = factor * (-distance_pid(cam_dist))
            else:
                throttle = 0
                yaw = 0
        
        if yaw != prev_yaw or throttle != prev_throttle:
            puzzlebot.send_vel_async(throttle, yaw)
            print(f"Throttle: {throttle:.2f}, Yaw: {yaw:.2f}")
            prev_throttle, prev_yaw = throttle, yaw

        if frame is not None:
            show_frame(frame, "Puzzlebot Stream", 600)
except KeyboardInterrupt:
    print("Exiting...")
finally:
    # Stop the video stream and release resources
    puzzlebot.stop_stream()
    cv2.destroyAllWindows()