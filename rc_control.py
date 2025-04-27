import math
import cv2
import requests
import time
import threading
from keybrd import is_pressed, is_toggled  # your keybrd module for key state detection
from pb_http_client import PuzzlebotHttpClient  # your custom client for sending commands

# Connection
BASE_URL = "http://192.168.137.139:5000"
ENABLE_STREAM = True  # Set to False to disable video stream

puzzlebot = PuzzlebotHttpClient(BASE_URL)

# cv2.namedWindow("RC Car", cv2.WINDOW_NORMAL)

# Store RC Car states
throttle, yaw = 0, 0
prev_throttle, prev_yaw = 0, 0

try:
    # Start the video stream
    puzzlebot.start_stream()

    while True:
        # Process control commands
        yaw_mag = math.radians(45)
        throttle_mag = 0.15
        if not is_toggled('m'):  # Manual mode
            throttle = (1 if is_pressed('w') else -1 if is_pressed('s') else 0) * throttle_mag
            yaw = (1 if is_pressed('a') else -1 if is_pressed('d') else 0) * yaw_mag
        else:  # Auto mode (circular trajectory)
            throttle = throttle_mag
            yaw = yaw_mag

        if yaw != prev_yaw or throttle != prev_throttle:
            puzzlebot.send_vel_async(throttle, yaw)
            print(f"Throttle: {throttle:.2f}, Yaw: {yaw:.2f}")
            prev_throttle, prev_yaw = throttle, yaw

        # Grab one frame from the stream on each iteration
        frame = puzzlebot.get_frame()
        if frame is not None:
            width = 400
            resized_frame = cv2.resize(frame, (width, int(frame.shape[0] * (width / frame.shape[1]))))
            cv2.imshow("RC Car", resized_frame)
            cv2.setWindowProperty("RC Car", cv2.WND_PROP_TOPMOST, 1)
    
        print(puzzlebot.get_pose())
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Exiting...")
finally:
    # Stop the video stream and release resources
    puzzlebot.stop_stream()
    cv2.destroyAllWindows()