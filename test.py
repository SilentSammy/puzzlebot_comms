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
BASE_URL = "http://192.168.137.139:5000"
ENABLE_STREAM = False  # Set to False to disable video stream

# Open the MJPEG stream once.
if ENABLE_STREAM:
    cap = cv2.VideoCapture(f"{BASE_URL}/car_cam")
    if not cap.isOpened():
        print("Error: Could not open video stream!")
        exit(1)
else:
    cap = None

# Store RC Car states
throttle, yaw = 0, 0
prev_throttle, prev_yaw = 0, 0

try:
    while True:
        # Process control commands
        yaw_mag = math.radians(45)
        throttle_mag = 0.15
        if not is_toggled('m'):  # Manual mode
            throttle = (1 if is_pressed('w') else -1 if is_pressed('s') else 0) * throttle_mag
            yaw = (1 if is_pressed('a') else -1 if is_pressed('d') else 0) * yaw_mag
        else:  # Auto mode (circular trajectory)
            t = time.time() % 2
            throttle = throttle_mag * math.sin(t * math.pi)
            yaw = 0

        if yaw != prev_yaw or throttle != prev_throttle:
            send_vel_async(throttle, yaw)
            print(f"Throttle: {throttle:.2f}, Yaw: {yaw:.2f}")
            prev_throttle, prev_yaw = throttle, yaw

        # Grab one frame from the stream on each iteration
        if ENABLE_STREAM and cap is not None:
            ret, frame = cap.read()
            if not ret:
                cap.release()
                cap = cv2.VideoCapture(f"{BASE_URL}/car_cam")
                continue
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