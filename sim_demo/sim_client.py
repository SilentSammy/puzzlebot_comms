import sys, os
sys.path[:0] = [os.path.abspath(os.path.join(os.path.dirname(__file__), p)) for p in ('..', '../..')]
import math
import cv2
import requests
import time
import threading
from keybrd import is_pressed, is_toggled  # your keybrd module for key state detection

def send_vel(linear, angular):
    """Send velocity command to the simulation webserver."""
    try:
        params = {"v": linear, "w": angular}
        response = requests.get(f"{BASE_URL}/cmd_vel", params=params)
    except Exception as ex:
        print("Error sending velocity:", ex)

def send_vel_async(linear, angular):
    """Dispatch send_vel asynchronously."""
    threading.Thread(target=send_vel, args=(linear, angular), daemon=True).start()

# Connection
BASE_URL = "http://127.0.0.1:5000"

# Open the MJPEG stream once.
cap = cv2.VideoCapture(f"{BASE_URL}/car_cam")
if not cap.isOpened():
    print("Error: Could not open video stream!")
    exit(1)

# Store RC Car states
throttle, yaw = 0, 0
prev_throttle, prev_yaw = 0, 0

try:
    while True:
        # Process control commands
        yaw_mag = math.radians(45)
        throttle_mag = 0.3
        if not is_toggled('m'):  # Manual mode
            throttle = (1 if is_pressed('w') else -1 if is_pressed('s') else 0) * throttle_mag
            yaw = (1 if is_pressed('a') else -1 if is_pressed('d') else 0) * yaw_mag
        else:  # Auto mode (circular trajectory)
            throttle = throttle_mag
            yaw = yaw_mag

        if yaw != prev_yaw or throttle != prev_throttle:
            send_vel_async(throttle, yaw)
            print(f"Throttle: {throttle:.2f}, Yaw: {yaw:.2f}")
            prev_throttle, prev_yaw = throttle, yaw

        # Grab one frame from the stream on each iteration
        ret, frame = cap.read()
        if ret:
            cv2.namedWindow("RC Car", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("RC Car", cv2.WND_PROP_TOPMOST, 1)
            cv2.resizeWindow("RC Car", 400, 400)
            cv2.imshow("RC Car", frame)

        # Check if 'q' was pressed to exit.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Exiting...")
finally:
    cap.release()
    cv2.destroyAllWindows()