import cv2
from robomaster import robot
import time
from keybrd import *

# Initialize the drone
tl_drone = robot.Drone()
tl_drone.initialize()
tl_flight = tl_drone.flight
tl_camera = tl_drone.camera
tl_camera.start_video_stream(display=False)
tl_camera.set_fps("high")
tl_camera.set_resolution("high")
tl_camera.set_bitrate(6)

# Store drone states
roll, pitch, yaw, throttle = 0, 0, 0, 0
prev_roll, prev_pitch, prev_yaw, prev_throttle = 0, 0, 0, 0

# Store flight state
flight = True

try:
    tl_flight.takeoff().wait_for_completed()
    while True:
        # Takeoff and landing control
        if rising_edge('t'):
            flight = not flight
            if flight:
                tl_flight.takeoff().wait_for_completed()
                print("Takeoff")
            else:
                tl_flight.land().wait_for_completed()
                print("Land")

        # Drone Control
        if flight:
            mag = 50
            yaw_mag = 100
            throttle_mag = 75
            if not is_toggled('m'): # Manual mode
                pitch = (1 if is_pressed('w') else -1 if is_pressed('s') else 0) * mag
                roll = (1 if is_pressed('d') else -1 if is_pressed('a') else 0) * mag
                yaw = (1 if is_pressed('e') else -1 if is_pressed('q') else 0) * yaw_mag
                throttle = (1 if is_pressed('z') else -1 if is_pressed('x') else 0) * throttle_mag
            else: # Auto mode
                # Circular trajectory
                pitch = 30
                yaw = 50
                roll = 0
                throttle = 0

        # Only update if there's a change in roll or pitch
        if roll != prev_roll or pitch != prev_pitch or yaw != prev_yaw or throttle != prev_throttle:
            tl_flight.rc(a=roll, b=pitch, c=throttle, d=yaw)
            print(f"Roll: {roll}, Pitch: {pitch}, Yaw: {yaw}, Throttle: {throttle}")
            prev_roll, prev_pitch, prev_yaw, prev_throttle = roll, pitch, yaw, throttle
        
        # Read and display the camera image
        img = tl_camera.read_cv2_image()
        cv2.namedWindow("Drone", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Drone", cv2.WND_PROP_TOPMOST, 1)
        cv2.resizeWindow("Drone", 400, 400)  # Set the window size to 800x600
        cv2.imshow("Drone", img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        # time.sleep(0.1)
finally:
    cv2.destroyAllWindows()
    tl_flight.land().wait_for_completed()
    tl_drone.close()
