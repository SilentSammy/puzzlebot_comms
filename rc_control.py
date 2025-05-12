import numpy as np
import math
import cv2
import requests
import time
import threading
from input_man import is_pressed, is_toggled, rising_edge  # your keybrd module for key state detection
from pb_http_client import PuzzlebotHttpClient  # your custom client for sending commands
from pose_estimation import find_arucos, estimate_marker_pose
import pose_estimation as pe
from simple_pid import PID
import visual_navigation as vn

# Connection
# puzzlebot = PuzzlebotHttpClient("http://192.168.137.139:5000", safe_mode=True)
puzzlebot = PuzzlebotHttpClient("http://127.0.0.1:5000", safe_mode=False)

# Maximum values for throttle and yaw
max_yaw = math.radians(90)
max_thr = 0.75

def manual_control():
    from input_man import get_axis
    slow_thr = 0.2
    slow_yaw = math.radians(45)

    # Get keyboard input
    keyvert = 1 if is_pressed('w') else -1 if is_pressed('s') else 0
    keyhor = 1 if is_pressed('a') else -1 if is_pressed('d') else 0
    keyboost = 1 if is_pressed('c') else 0

    # Get controller input
    joyver = get_axis('LY')
    joyhor = -get_axis('LX')
    conboost = max(get_axis('RT'), get_axis('LT'))

    # Calculate the higher of the two absolute values
    thr = keyvert if abs(keyvert) > abs(joyver) else joyver
    yaw = keyhor if abs(keyhor) > abs(joyhor) else joyhor
    boost = max(conboost, keyboost)  # Use the higher value between rt and sh

    if not puzzlebot.safe_mode:
        boost = 0 # Disable boost in unsafe mode

    # Interpolate from slow to max_thr based on the boost value
    thr *= slow_thr + (max_thr - slow_thr) * boost
    yaw *= slow_yaw + (max_yaw - slow_yaw) * boost

    # print(f"Throttle: {thr:.2f}, Yaw: {yaw:.2f}")
    return thr, yaw

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

def reset_nav_mode():
    global nav_mode
    nav_mode = 1
    print("Control mode: Manual")
    
nav_mode = 1
try:
    while True:
        # Inputs and outputs
        frame = puzzlebot.get_frame()
        drawing_frame = frame.copy()
        throttle, yaw = 0, 0

        # Optional screenshot
        if rising_edge('p'):
            cv2.imwrite("screenshot.png", frame)
            print("Screenshot taken")

        # Safe mode selection
        if rising_edge('z', 'Y'):
            puzzlebot.safe_mode = not puzzlebot.safe_mode
            print(f"Safe mode: {puzzlebot.safe_mode}")

        # Control mode selection
        if rising_edge('1', 'X'):
            nav_mode = 1
            print("Control mode: Manual")
        elif rising_edge('2', 'A'):
            nav_mode = 2
            print("Control mode: Navigate track")
        elif rising_edge('3', 'B'):
            nav_mode = 3
            print("Control mode: Navigate to marker")
        elif rising_edge('4'):
            nav_mode = 4
            print("Control mode: Follow line")
        elif rising_edge('5'):
            nav_mode = 5
            print("Control mode: Preprogrammed sequence")
        
        # Control
        if nav_mode == 2:
            throttle, yaw = vn.navigate_track(frame, drawing_frame)
        elif nav_mode == 3:
            throttle, yaw = vn.navigate_to_marker(frame, drawing_frame)
        elif nav_mode == 4:
            throttle, yaw = vn.follow_line(frame, drawing_frame)
        elif nav_mode == 5:
            throttle, yaw = vn.sequence(when_done=reset_nav_mode)
        
        # Always allow manual control
        thr, yw = manual_control()
        throttle += thr
        yaw += yw

        # Send control commands to the robot
        puzzlebot.send_vel(throttle, yaw, wait_for_completion=False)
        
        show_frame(drawing_frame, "Puzzlebot Stream", 400)
except KeyboardInterrupt:
    print("Exiting...")
finally:
    puzzlebot._stop_stream()
    cv2.destroyAllWindows()