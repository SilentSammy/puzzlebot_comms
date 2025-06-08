import os
import math
import cv2
import time
from input_manager.input_man import is_pressed, is_toggled, rising_edge, get_axis
from pb_http_client import PuzzlebotHttpClient  # your custom client for sending commands
import visual_navigation as vn
import yolo
from collections import deque

turns = deque([3, 1, 2])

# Connection
# puzzlebot = PuzzlebotHttpClient("http://192.168.137.165:5000", safe_mode=True, id = 0)
puzzlebot = PuzzlebotHttpClient("http://127.0.0.1:5001", safe_mode=False, id = 1)

# Navigation components
line_foll = vn.LineFollower()
sl_det = vn.StoplightDetector(
    hsv_val_range=(128, 255)
)
fl_det = vn.FlagDetector(pattern_size=(6, 3), square_size=0.05)
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
    turn_left = [
        (0.35, 0, 2.0),
        (0, math.radians(90), 5.0),
        (0.35, 0, 2.0),
    ],
)
ol_con = vn.OpenLoopController(
    linear_factor=1.2 if puzzlebot.id == 0 else 1.0,  # Adjust linear factor based on robot ID
    angular_factor=1.1 if puzzlebot.id == 0 else 1.0,  # Adjust angular factor based on robot ID
)
sg_det = vn.SignDetector(
    get_signs_func=yolo.get_signs,  # Uncomment this line to use YOLO for sign detection
)
track_nav = vn.TrackNavigator(
    intersection_navigator=int_nav,
    # sign_detector=sg_det,
    flag_detector=fl_det,
    stoplight_detector=sl_det,
)

# Maximum values for throttle and yaw
max_yaw = math.radians(180)
max_thr = 0.6

def manual_control():
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

def choose_direction(frame):
    from visual_navigation import choose_direction_nb
    return choose_direction_nb(frame)

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

modes = [
    (('1', 'X'), "Manual Control", lambda: None),
    (('2', 'A'), "Follow Line with Stoplight", lambda: sl_nav.navigate(frame, drawing_frame)),
    (('3'), "Follow Line", lambda: line_foll.follow_line(frame, drawing_frame)),
    (('4'), "Follow Line with Intersection", lambda: int_nav.navigate(frame, drawing_frame)),
    (('5'), "Stop at intersection", lambda: in_det.stop_at_intersection(frame, drawing_frame)),
    (('6'), "SignDetector", lambda: sg_det.get_confirmed_signs_nb(frame, drawing_frame)),
    (('7'), "Track Navigator", lambda: track_nav.navigate(frame, drawing_frame)),
]

mode = 0
try:
    while True:
        # Inputs and outputs
        frame = puzzlebot.get_frame()
        drawing_frame = frame.copy()
        throttle, yaw = 0, 0

        # Optional screenshot or recording
        if rising_edge('p'):
            screenshot(frame)
        record(frame if is_toggled('o') else None)

        # Choose mode
        for i, m in enumerate(modes):
            if rising_edge(*m[0]):
                mode = i
                print(f"Control mode: {m[1]}")
        
        # Execute the current mode
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

        # Show the frame
        show_frame(drawing_frame, "Puzzlebot Stream")

except KeyboardInterrupt:
    print("Exiting...")
finally:
    puzzlebot.send_vel(0, 0)
    puzzlebot._stop_stream()
    cv2.destroyAllWindows()
