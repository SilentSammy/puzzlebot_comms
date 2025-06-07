import threading
import cv2
from cam_navigation import visual_navigation as vn
from cam_navigation import buzzer
from cam_navigation import gpt
from cam_navigation import led
import math
# from cam_navigation import yolo
import time

def success_action():
    buzzer.play_melody_nonblocking(buzzer.melodies["custom_success_chime"])
    
def decision_action(action_id):
    if action_id == 0:
        buzzer.play_melody_nonblocking(buzzer.melodies["3_lows"])
    elif action_id == 1:
        led.run_led_sequence_nonblocking(led.sequences["left_turn"])
        buzzer.play_melody_nonblocking(buzzer.melodies["falling_tone"])
    elif action_id == 2:
        led.run_led_sequence_nonblocking(led.sequences["right_turn"])
        buzzer.play_melody_nonblocking(buzzer.melodies["rising_tone"])
    elif action_id == 3:
        buzzer.play_melody_nonblocking(buzzer.melodies["3_highs"])

def navigate(frame, drawing_frame=None):
    global last_time
    # thr, yaw = track_nav.navigate(frame, drawing_frame)
    thr, yaw = int_nav.navigate(frame, drawing_frame)
    current_time = time.time()
    if last_time is not None:
        fps = 1.0 / (current_time - last_time)
        print(f"FPS: {fps:.2f}")
    last_time = current_time
    return thr, yaw

last_time = time.time()
line_foll = vn.LineFollower()
sl_det = vn.StoplightDetector()
fl_det = vn.FlagDetector(pattern_size=(6, 3), square_size=0.05)
in_det = vn.IntersectionDetector(undistort=True)
sl_nav = vn.StoplightNavigator(
    line_follower=line_foll, 
    stoplight_detector=sl_det, 
    flag_detector=fl_det,
    end_action=lambda: print("Stoplight navigation completed!")  # Optional end action
)
int_nav = vn.IntersectionNavigator(
    line_follower=line_foll,
    intersection_detector=in_det,
    decision_func=lambda _: 1,
)
ol_con = vn.OpenLoopController(
    linear_factor=1.0,
    angular_factor=1.0,
)
sg_det = vn.SignDetector(lambda f: yolo.get_signs)
track_nav = vn.TrackNavigator(
    intersection_navigator=int_nav,
    sign_detector=sg_det,
    flag_detector=fl_det,
)
