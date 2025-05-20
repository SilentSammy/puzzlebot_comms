import os
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
puzzlebot = PuzzlebotHttpClient("http://192.168.137.9:5000", safe_mode=True)
# puzzlebot = PuzzlebotHttpClient("http://127.0.0.1:5000", safe_mode=False)

# Maximum values for throttle and yaw
max_yaw = math.radians(180)
max_thr = 0.6

def manual_control():
    from input_man import get_axis
    slow_thr = 0.2
    slow_yaw = math.radians(90)

    # Get keyboard input
    keyvert = 1 if is_pressed('w') else -1 if is_pressed('s') else 0
    keyhor = 1 if is_pressed('a') else -1 if is_pressed('d') else 0
    keyboost = 1 if is_pressed('c') else 0

    # Get controller input
    joyver = get_axis('LY')
    # joyhor = -get_axis('LX')
    joyhor = -get_axis('RX')
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

def show_frame(img, name, scale=1):
    show_frame.first_time = show_frame.first_time if hasattr(show_frame, 'first_time') else True
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)
    if show_frame.first_time:
        cv2.resizeWindow(name, int(img.shape[1]*scale), int(img.shape[0]*scale))
        show_frame.first_time = False
    cv2.imshow(name, img)
    if cv2.waitKey(1) & 0xFF == 27:
        raise KeyboardInterrupt

def reset_nav_mode():
    global nav_mode
    nav_mode = 1
    print("Control mode: Manual")

def choose_direction(intersection, drawing_frame=None, time_limit=3):
    if not hasattr(choose_direction, 'tmr'):
        choose_direction.tmr = time.time()

    dir_labels = ["back", "left", "right", "front"]
    chosen_idx = -1
    avail_idxs = [i for i, d in enumerate(intersection) if d is not None]
    keys = [('k', 'DPAD_DOWN'), ('j', 'DPAD_LEFT'), ('l', 'DPAD_RIGHT'), ('i', 'DPAD_UP')]
    if drawing_frame is not None:
        for i, d in enumerate(intersection):
            if d is not None:
                cv2.putText(drawing_frame, keys[i][0].upper(), d[1], cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

    for i, key in enumerate(keys):
        if is_pressed(*key):
            if i in avail_idxs:
                chosen_idx = i
                break
    
    # If the time limit is reached, choose a random direction
    if time_limit >= 0:
        print(f"Time remaining: {time_limit - (time.time() - choose_direction.tmr):.2f} seconds")
        if time.time() - choose_direction.tmr > time_limit:
            chosen_idx = np.random.choice(avail_idxs)
    
    if chosen_idx != -1:
        print(f"Chosen direction: {dir_labels[chosen_idx]}")
        del choose_direction.tmr
    return chosen_idx

def screenshot(frame):
    import os

    # Static variables
    screenshot.last_time = screenshot.last_time if hasattr(screenshot, 'last_time') else None
    screenshot.dir_path = screenshot.dir_path if hasattr(screenshot, 'dir_path') else "./screenshots/"+time.strftime("%Y-%m-%d_%H-%M-%S")
    screenshot.count = screenshot.count if hasattr(screenshot, 'count') else 0

    # If less than n seconds have passed since the last screenshot, return
    if screenshot.last_time is not None and time.time() - screenshot.last_time < 0.2:
        return
    screenshot.last_time = time.time()

    # Make the directory if it doesn't exist
    os.makedirs(screenshot.dir_path, exist_ok=True)

    # Save the image
    filename = os.path.join(screenshot.dir_path, f"screenshot_{screenshot.count:03}.png")
    cv2.imwrite(filename, frame)
    print(f"Screenshot saved: {filename}")

    # Increment the count
    screenshot.count += 1

def record(frame):
    """
    If frame is a valid image (numpy array), write it to the video file.
    If frame is None, release the VideoWriter if it exists.
    """
    # Close the video if frame is None
    if frame is None:
        if hasattr(record, "vw"):
            record.vw.release()
            print("Video recording closed.")
            del record.vw
        return

    # If VideoWriter hasn't been created yet, initialize it now
    if not hasattr(record, "vw"):
        fps = 30  # desired frame rate
        height, width = frame.shape[:2]
        
        # Create a directory for video output if needed
        record.dir_path = record.dir_path if hasattr(record, "dir_path") else "./resources/videos"
        os.makedirs(record.dir_path, exist_ok=True)
        
        # Create a timestamped file name 
        filename = os.path.join(record.dir_path, time.strftime("output_%Y-%m-%d_%H-%M-%S.mp4"))
        
        # Choose a codec that works well (e.g., 'mp4v')
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        record.vw = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        print(f"Video recording started: {filename}")
    
    # Append the frame to the video file
    record.vw.write(frame)

nav_mode = 1
try:
    while True:
        # Inputs and outputs
        frame = puzzlebot.get_frame()
        # frame = vn.undistort_fisheye(frame)
        drawing_frame = frame.copy()
        throttle, yaw = 0, 0

        # Optional screenshot or recording
        if rising_edge('p'):
            screenshot(frame)
        record(frame if is_toggled('o') else None)

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
            print("Control mode: Waypoint navigation")
        elif rising_edge('4'):
            nav_mode = 4
            print("Control mode: Follow line")
        elif rising_edge('5'):
            nav_mode = 5
            print("Control mode: Preprogrammed sequence")
        elif rising_edge('6'):
            nav_mode = 6
            print("Testing mode")
        
        # Control
        if nav_mode == 2:
            throttle, yaw = vn.navigate_track(frame, drawing_frame, lambda i: choose_direction(i, drawing_frame, -1))
        elif nav_mode == 3:
            throttle, yaw = vn.waypoint_navigation(frame, drawing_frame)
        elif nav_mode == 4:
            throttle, yaw = vn.follow_line(frame, drawing_frame, max_thr=0.25, align_thres=0.2)
        elif nav_mode == 5:
            throttle, yaw = vn.sequence(speed_factor=2)
        
        # Always allow manual control
        thr, yw = manual_control()
        throttle += thr
        yaw += yw

        # Disable output for debugging
        if rising_edge('0'):
            print("Output" + (" disabled" if is_toggled('0') else " enabled"))
        if is_toggled('0'):
            throttle = 0
            yaw = 0

        # Send control commands to the robot
        puzzlebot.send_vel(throttle, yaw, wait_for_completion=False)

        # Show the frame
        show_frame(drawing_frame, "Puzzlebot Stream")

except KeyboardInterrupt:
    print("Exiting...")
finally:
    puzzlebot.send_vel(0, 0)
    puzzlebot._stop_stream()
    cv2.destroyAllWindows()
