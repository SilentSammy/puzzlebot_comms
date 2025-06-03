import threading
import cv2
from cam_navigation import visual_navigation as vn
from cam_navigation import buzzer
from cam_navigation import gpt
from cam_navigation import led
import math

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
    thr, yaw = track_nav.navigate(frame, drawing_frame)
    return thr, yaw

# Navigation and detection objects + settings
line_foll = vn.LineFollower()
sl_det = vn.StoplightDetector()
fl_det = vn.FlagDetector()
in_det = vn.IntersectionDetector(undistort=True)
sl_nav = vn.StoplightNavigator(
    line_follower=line_foll, 
    stoplight_detector=sl_det, 
    flag_detector=fl_det,
    end_action=lambda: success_action()
)
track_nav = vn.TrackNavigator(
    line_follower=line_foll,
    intersection_detector=in_det,
    decision_func=lambda f: 2,
    decision_action=decision_action,
    # turn_left = [(0.1, 0, 2), (0.4, math.radians(90))], # needs tuning, using defaults for now
    # turn_right = [(0.1, 0, 2), (0.65, -math.radians(90))],
)
ol_con = vn.OpenLoopController(
    linear_factor=1.1,
    angular_factor=1.1,
)