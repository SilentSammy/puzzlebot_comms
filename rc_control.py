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
# puzzlebot = PuzzlebotHttpClient("http://192.168.137.139:5000", safe_mode=True)
puzzlebot = PuzzlebotHttpClient("http://127.0.0.1:5000", safe_mode=False)

# Maximum values for throttle and yaw
max_yaw = math.radians(90)
max_thr = 0.75

# Navigation algorithms
def manual_control():
    from input_man import get_axis
    slow_thr = 0.2
    slow_yaw = math.radians(45)

    # Get keyboard input
    keyvert = 1 if is_pressed('w') else -1 if is_pressed('s') else 0
    keyhor = 1 if is_pressed('a') else -1 if is_pressed('d') else 0
    keyboost = 1 if is_pressed('c') else 0

    # Get controller input
    joyver = get_axis('LY')
    joyhor = -get_axis('LX')
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

def navigate_to_marker(frame):
    # Static variables
    max_yaw = math.radians(30)
    max_thr = 0.2
    yaw_threshold = 5.0  # The robot will start moving forward when the target is this many degrees from the center
    if not hasattr(navigate_to_marker, "pids"):
        navigate_to_marker.pids = {
            'yaw_pid': PID(Kp=1, Ki=0, Kd=0.1, setpoint=0.0, output_limits=(-max_yaw, max_yaw)),
            'throttle_pid': PID(Kp=2, Ki=0, Kd=0.1, setpoint=0.3, output_limits=(-max_thr, max_thr))
        }
    yaw_pid = navigate_to_marker.pids['yaw_pid']
    thr_pid = navigate_to_marker.pids['throttle_pid']

    markers, ids = find_arucos(frame)
    if markers and ids is not None:
        # Find the marker with the lowest ID
        min_id_index = np.argmin(ids)
        marker = markers[min_id_index]
        # Remove extra nesting if necessary; otherwise, pass marker directly.
        _, _, _, cam_dist, _, cam_yaw = estimate_marker_pose( marker_corners=marker, frame=frame, ref_size=0.063300535, fov_x=math.radians(60) )
        print(f"Dist: {cam_dist:.2f}, Yaw: {math.degrees(cam_yaw):.2f}")
        yaw = yaw_pid(cam_yaw)
        
        # Compute interpolation factor 'alpha'
        # alpha = 0 when yaw error is >= threshold, alpha = 1 when yaw error is 0.
        alpha = 1 - (abs(cam_yaw) / math.radians(yaw_threshold)) if abs(cam_yaw) < math.radians(yaw_threshold) else 0

        # Interpolate the distance input: when alpha is 0, measured distance is set to the setpoint.
        measured_distance = (1 - alpha) * thr_pid.setpoint + alpha * cam_dist
        throttle = -thr_pid(measured_distance)
    else:
        throttle, yaw = 0, 0
        # Write "Searching for ArUco" on the frame
        cv2.putText(frame, "Searching for ArUco", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return throttle, yaw

def follow_line(frame):
    if frame is None:
        return 0, 0

    frame_height, frame_width = frame.shape[:2]

    # Static variables
    max_yaw = math.radians(60)
    max_thr = 0.2
    if not hasattr(follow_line, "yaw_pid"):
        follow_line.yaw_pid = PID(Kp=0.6, Ki=0, Kd=0.1, setpoint=0.0, output_limits=(-max_yaw, max_yaw))

    def get_line_candidates():
        # Convert to binary mask, only keeping pixels darker than dark_thres.
        dark_thres = 100
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, dark_thres, 255, cv2.THRESH_BINARY_INV)
        
        # Shrink the vertical field of view to the lower part of the frame.
        v_fov = 0.4
        mask[:int(frame_height * (1-v_fov)), :] = 0
        
        # Erode and dilate to remove noise and fill gaps.
        mask = cv2.erode(mask, kernel=np.ones((3, 3), np.uint8), iterations=3)
        mask = cv2.dilate(mask, kernel=np.ones((3, 3), np.uint8), iterations=3)

        # Find contours in the modified mask, filtering out noise.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        MIN_AREA = 1000 # Minimum area of contours to consider
        contours = [c for c in contours if cv2.contourArea(c) > MIN_AREA]
    
        return contours

    def get_contour_line(c, fix_vert=True):
        vx, vy, cx, cy = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        scale = 100  # Adjust scale as needed for visualization.
        pt1 = (int(cx - vx * scale), int(cy - vy * scale))
        pt2 = (int(cx + vx * scale), int(cy + vy * scale))
        angle = math.degrees(math.atan2(vy, vx))

        if fix_vert:
            # angle = angle # Placeholder for any angle correction logic.
            angle = angle - 90 * np.sign(angle)

        return pt1, pt2, angle, cx, cy

    contours = get_line_candidates()

    # Show the direction of the line candidates
    for i, c in enumerate(contours):
        pt1, pt2, angle, cx, cy = get_contour_line(c)
        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        # cv2.putText(frame, str(i), (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        # print(f"Contour {i}: Angle: {angle:.2f} degrees") # Optionally print angles for debugging

    throttle, yaw = 0, 0
    if contours:
        def contour_key(c):
            # Define the maximum angle for clamping
            max_angle = 80  # Adjust this value as needed
            # Get the direction of the line and its centroid
            _, _, angle, cx, cy = get_contour_line(c)
            angle = max(min(angle, max_angle), -max_angle)
            # Compute ref_x based on angle: 0° -> center, +max_angle -> left, -max_angle -> right.
            ref_x = (frame_width / 2) + (angle / max_angle) * (frame_width / 2)
            # Draw ref_x on the frame for debugging
            # cv2.line(frame, (int(ref_x), 0), (int(ref_x), frame_height), (0, 0, 255), 2)
            # Compute the error between the centroid and our adjusted reference.
            x_err = abs(cx - ref_x)
            # Return a tuple for sorting: first sort by lowest centroid (i.e. largest cy) then by x error.
            return (x_err)
            
        # Choose the best candidate
        sorted_contours = sorted(contours, key=contour_key)
        line_contour = sorted_contours[0]
        
        # Draw the best candidate in green and the others in red.
        cv2.drawContours(frame, [line_contour], -1, (0, 255, 0), 2)
        for c in sorted_contours[1:]:
            cv2.drawContours(frame, [c], -1, (0, 0, 255), 2)

        # Get the X position of the line in the frame.
        x, y, w, h = cv2.boundingRect(line_contour)
        center_x = x + w // 2
        normalized_x = (center_x - (frame_width/2)) / (frame_width/2)
        cv2.line(frame, (center_x, 0), (center_x, frame_height), (255, 0, 0), 2)
        
        # Adjust yaw to keep the line centered in the frame.
        yaw = follow_line.yaw_pid(normalized_x)
        
        # Decrease throttle as the line moves away from the center.
        alignment = 1 - abs(normalized_x) # 1 when centered, 0 when at the edge.
        align_thres = 0.3 # Throttle will be max_thr when aligned, 0 at the threshold, and negative below the threshold.
        throttle = max_thr * ((alignment - align_thres) / (1 - align_thres))
    else:
        # Write "Searching for line" on the frame
        cv2.putText(frame, "Searching for line", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    return throttle, yaw  # Comment this line to disable output
    return 0, 0

def sequence():
    # Static variables
    if not hasattr(sequence, "start_time"):
        sequence.start_time = time.time()
    
    # Define the sequence of actions
    actions = [ # v, w, t
        (0.15, 0, 2), # Move 30cm forward
        (0, math.radians(30), 3), # 90° turn
        (0.15, 0, 2), # Move 30cm forward
    ]

    # Get the elapsed time
    elapsed_time = time.time() - sequence.start_time

    action = None
    ac_time = 0
    for i, act in enumerate(actions):
        v, w, t = act
        ac_time += t
        if elapsed_time < ac_time:
            action = act
            break
    
    done = action is None
    throttle, yaw, _ = action if not done else (0, 0, True)
    if done:
        del sequence.start_time
    return throttle, yaw, done

def show_frame(frame, name="Frame", max_size=400):
    h, w = frame.shape[:2]
    if h > w:
        new_h = max_size
        new_w = int(w * (max_size / h))
    else:
        new_w = max_size
        new_h = int(h * (max_size / w))
    resized_frame = cv2.resize(frame, (new_w, new_h))
    cv2.imshow(name, resized_frame)
    cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)
    cv2.setWindowProperty(name, cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_NORMAL)  # Make the window resizable
    if cv2.waitKey(1) & 0xFF == 27:
        raise KeyboardInterrupt

try:
    nav_mode = 1
    while True:
        # Inputs and outputs
        frame = puzzlebot.get_frame()
        throttle, yaw = 0, 0

        # Optional screenshot
        if rising_edge('p'):
            cv2.imwrite("screenshot.png", frame)
            print("Screenshot taken")

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
            print("Control mode: Follow line")
        elif rising_edge('3', 'B'):
            nav_mode = 3
            print("Control mode: Navigate to marker")
        elif rising_edge('4'):
            nav_mode = 4
            print("Preprogrammed sequence")
        elif rising_edge('5'):
            nav_mode = 5
            print("Navigate intersection")
        if nav_mode == 2:
            throttle, yaw = follow_line(frame)
        elif nav_mode == 3:
            throttle, yaw = navigate_to_marker(frame)
        elif nav_mode == 4:
            throttle, yaw, done = sequence()
            if done:
                print("Sequence completed")
                nav_mode = 1
        elif nav_mode == 5:
            throttle, yaw = vn.navigate_intersection(frame)
        
        # Always allow manual control
        thr, yw = manual_control()
        throttle += thr
        yaw += yw

        # Send control commands to the robot
        puzzlebot.send_vel(throttle, yaw, wait_for_completion=False)
        
        show_frame(frame, "Puzzlebot Stream", 400)
except KeyboardInterrupt:
    print("Exiting...")
finally:
    puzzlebot._stop_stream()
    cv2.destroyAllWindows()