import os
import math
import cv2
import time
from input_manager.input_man import is_pressed, is_toggled, rising_edge, get_axis
import pb_http_client  # your custom client for sending commands
import visual_navigation as vn
import yolo
from collections import deque
from videoplayer import VideoRecorder
from buzzer import melodies
import random

# Recorders
raw_recorder = VideoRecorder()
annotated_recorder = VideoRecorder()

# Connection
puzzlebot = pb_http_client.PuzzlebotHttpClient("http://192.168.137.165:5000", safe_mode=True, id = 0)
# puzzlebot = pb_http_client.PuzzlebotHttpClient("http://127.0.0.1:5001", safe_mode=False, id = 1)

def end_sequence():
    contr = vn.OpenLoopController()
    # do a little wiggle dance at the end
    wiggle_actions = [ (0, -math.radians(20), 0.5), (0, math.radians(20), 0.5), ]
    wiggle_actions*= 2
    wiggle_actions.insert(0, (0, math.radians(20), 0.75)) # Start with a slightly longer left turn

    def start():
        print("Starting end sequence...")
        puzzlebot.play_buzzer(melodies["custom_success_chime"])
        contr.start(wiggle_actions)
    
    def ongoing():
        if contr.running():
            print("Dancing...")
            return contr.execute()
    
    return start, ongoing

def signs_action(signs):
    from visual_navigation import SignType, Sign
    signs:list[Sign] = signs
    if not signs:
        return
    
    # Get the closest sign
    closest_sign = min(signs, key=lambda s: s.approx_dist)
    if closest_sign.approx_dist < 0.65 and closest_sign.type.name in melodies:
        print(f"Detected sign: {closest_sign.type.name} at {closest_sign.approx_dist:.2f}m")
        melody = melodies[closest_sign.type.name]
        puzzlebot.play_buzzer(melody)

def decision_action(decision):
    if decision == 0:  # backward
        puzzlebot.play_buzzer(melodies["3_lows"])
    elif decision == 1:  # left
        puzzlebot.play_buzzer(melodies["falling_tone"])
    elif decision == 2:  # right
        puzzlebot.play_buzzer(melodies["rising_tone"])
    elif decision == 3:  # forward
        puzzlebot.play_buzzer(melodies["3_highs"])

end_seq = end_sequence()
turns = deque([3, 1, 2])

# Navigation components
line_foll = vn.LineFollower()
sl_det = vn.StoplightDetector(
    hsv_val_range=(128, 225),
    chain_length=3,
)
fl_det = vn.FlagDetector(pattern_size=(4, 3), square_size=0.025)
in_det = vn.IntersectionDetector(
    undistort=puzzlebot.id == 0,
    setpoint=0.675,
    max_thr=0.1
)
sl_nav = vn.StoplightNavigator(
    line_follower=line_foll, 
    stoplight_detector=sl_det, 
    flag_detector=fl_det,
    end_action=lambda: print("Stoplight navigation completed!")  # Optional end action
)
int_nav = vn.IntersectionNavigator(
    line_follower=line_foll,
    intersection_detector=in_det,
    decision_func=lambda _: turns.popleft() if turns else -1, # Use deque to cycle through turns
    decision_action=decision_action,
    turn_left = [
        (0.4, 0, 2.0),
        (0, math.radians(90), 5.0),
        (0.4, 0, 2.0),
    ],
)
ol_con = vn.OpenLoopController(
    linear_factor=1.2 if puzzlebot.id == 0 else 1.0,  # Adjust linear factor based on robot ID
    angular_factor=1.1 if puzzlebot.id == 0 else 1.0,  # Adjust angular factor based on robot ID
)
sg_det = vn.SignDetector(
    get_signs_func=lambda f: yolo.get_signs(f),  # Uncomment this line to use YOLO for sign detection
    signs_action=lambda s: signs_action(s),  # Optional action for detected signs
)
track_nav = vn.TrackNavigator(
    intersection_navigator=int_nav,
    sign_detector=sg_det,
    flag_detector=fl_det,
    stoplight_detector=sl_det,
    end_action=lambda: end_seq[0](),  # Optional end action
    ongoing_end_action=lambda: end_seq[1](),  # Optional ongoing end action
)

# Maximum values for throttle and yaw

def manual_control():
    max_yaw = math.radians(180)
    max_thr = 0.6
    slow_thr = 0.2
    slow_yaw = math.radians(60)

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

    # if not puzzlebot.safe_mode:
    #     boost = 0 # Disable boost in unsafe mode

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

def screenshot(frame):
    import os

    # Static variables
    screenshot.last_time = screenshot.last_time if hasattr(screenshot, 'last_time') else None
    screenshot.dir_path = screenshot.dir_path if hasattr(screenshot, 'dir_path') else "./resources/screenshots/dated/"+time.strftime("%Y-%m-%d_%H-%M-%S")
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

modes = [
    (('1', 'X'), "Manual Control", lambda: None),
    (('2', 'A'), "Follow Line", lambda: line_foll.follow_line(frame, drawing_frame)),
    (('3', 'B'), "Track Navigator", lambda: track_nav.navigate(frame, drawing_frame)),
    (('4'), "Follow Line with Stoplight", lambda: sl_nav.navigate(frame, drawing_frame)),
    (('5'), "Follow Line with Intersection", lambda: int_nav.navigate(frame, drawing_frame)),
    (('6'), "Stop at intersection", lambda: in_det.stop_at_intersection(frame, drawing_frame)),
    (('7'), "SignDetector", lambda: sg_det.get_confirmed_signs_nb(frame, drawing_frame)),
]

mode = 0
try:
    while True:
        
        # Inputs and outputs
        frame = puzzlebot.get_frame()
        drawing_frame = frame.copy()
        if frame is None:
            continue  # Skip if no frame is received
        throttle, yaw = 0, 0
        
        # Optionally play the buzzer
        if is_pressed('b', 'Y'):
            melody_name = list(melodies.keys())[:3]
            puzzlebot.play_buzzer(melodies[random.choice(melody_name)])

        # Optional screenshot or recording
        if rising_edge('p'):
            screenshot(frame)
        if is_toggled('o'):
            if not raw_recorder.is_recording():
                raw_recorder.start(frame)
            raw_recorder.write(frame)
        else:
            if raw_recorder.is_recording():
                raw_recorder.stop()

        # Choose mode
        for i, m in enumerate(modes):
            if rising_edge(*m[0]):
                mode = i
                print(f"Control mode: {m[1]}")
        
        # Execute the current mode
        t2 = time.time()
        func = modes[mode][2]
        if func:
            result = func()
            if ( isinstance(result, tuple) and len(result) == 2 and all(isinstance(x, float) for x in result) ):
                throttle, yaw = result

        # Disable output for debugging
        if rising_edge('0'):
            print("Output" + (" disabled" if is_toggled('0') else " enabled"))
        if is_toggled('0'):
            throttle = yaw = 0

        # Always allow manual control
        thr, yw = manual_control()
        throttle += thr
        yaw += yw

        # Send control commands to the robot
        puzzlebot.send_vel(throttle, yaw, wait_for_completion=False)

        # Optionally record the already annotated frame
        if is_toggled('i'):
            if not annotated_recorder.is_recording():
                annotated_recorder.start(drawing_frame)
            annotated_recorder.write(drawing_frame)
        else:
            if annotated_recorder.is_recording():
                annotated_recorder.stop()

        # Show the frame
        show_frame(drawing_frame, "Puzzlebot Stream")

except KeyboardInterrupt:
    print("Exiting...")
finally:
    puzzlebot.send_vel(0, 0)
    puzzlebot._stop_stream()
    cv2.destroyAllWindows()
