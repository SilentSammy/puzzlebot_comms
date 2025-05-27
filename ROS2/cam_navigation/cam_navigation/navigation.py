import threading
import cv2
from cam_navigation import visual_navigation as vn
from cam_navigation import buzzer
from cam_navigation import gpt

def navigate(frame, drawing_frame=None):
    def success_action():
        buzzer.play_melody_nonblocking(buzzer.melodies["custom_success_chime"])
    
    def decision_action(action_id):
        if action_id == 0:
            buzzer.play_melody_nonblocking(buzzer.melodies["3_lows"])
        elif action_id == 1:
            buzzer.play_melody_nonblocking(buzzer.melodies["falling_tone"])
        elif action_id == 2:
            buzzer.play_melody_nonblocking(buzzer.melodies["rising_tone"])
        elif action_id == 3:
            buzzer.play_melody_nonblocking(buzzer.melodies["3_highs"])

    thr, yaw = vn.navigate_track(frame, drawing_frame=drawing_frame,
        undistort=True,
        decision_action=decision_action,
        decision_func=gpt.choose_direction_nb,
        # decision_func=lambda f: 2,
    )
    return thr, yaw
