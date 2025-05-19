import sys
import os
sys.path[:0] = [os.path.abspath(os.path.join(os.path.dirname(__file__), p)) for p in ('..', '../..')]
from simple_pid import PID
import numpy as np
import math
import cv2
import time

def adaptive_thres(frame, drawing_frame=None,
    blur_kernel_size=(7, 7),  # Kernel size for GaussianBlur
    adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Adaptive thresholding method
    threshold_type=cv2.THRESH_BINARY_INV,  # Thresholding type
    block_size=141,  # Size of the neighborhood used for thresholding (must be odd)
    c_value=6,  # Constant subtracted from the mean or weighted mean (the higher the value, the darker the pixels need to be to be considered black)
):
    # Processing
    drawing_frame = drawing_frame if drawing_frame is not None else frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, blur_kernel_size, 0)
    mask = cv2.adaptiveThreshold(gray, 255, adaptive_method, threshold_type, block_size, c_value)
    drawing_frame[:] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return mask

def get_gray_mask(frame, drawing_frame=None,
                  saturation_thresh=112,
                  value_thresh=(40, 255)):
    """
    Returns a binary mask keeping only near-grayscale pixels.
    - saturation_thresh: max saturation to be considered gray
    - value_thresh: (min, max) brightness range
    If drawing_frame is provided (or defaults to a copy of frame),
    it will be overwritten with the mask visualized in BGR.
    """
    # Prepare drawing_frame
    drawing_frame = drawing_frame if drawing_frame is not None else frame.copy()
    
    # Convert and split
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create mask: low saturation + within brightness range
    lower = (0, 0,       value_thresh[0])
    upper = (179, saturation_thresh, value_thresh[1])
    mask = cv2.inRange(hsv, lower, upper)
    
    # Visualize mask on drawing_frame
    drawing_frame[:] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    return mask

def refined_mask(frame, drawing_frame=None):
    adaptive_mask = adaptive_thres(frame)
    gray_mask = get_gray_mask(frame)

    refined_mask = cv2.bitwise_and(adaptive_mask, gray_mask)

    if drawing_frame is not None:
        drawing_frame[:] = cv2.cvtColor(refined_mask, cv2.COLOR_GRAY2BGR)
    return refined_mask

def get_line_mask(frame, drawing_frame=None,
    v_fov = 0.4,  # Bottom field of view (0.4 = 40% of the frame height)
    morph_kernel = np.ones((5, 5), np.uint8),  # Kernel for morphological operations
    erode_iterations = 4,  # Number of iterations for erosion
    dilate_iterations = 6,  # Number of iterations for dilation
):
    # Get mask
    # mask = adaptive_thres(frame)
    mask = refined_mask(frame, drawing_frame=drawing_frame)

    # Only keep the lower part of the mask, filling the upper part with black.
    mask[:int(frame.shape[:2][0] * (1-v_fov)), :] = 0

    # Erode and dilate to remove noise and fill gaps.
    mask = cv2.erode(mask, kernel=morph_kernel, iterations=erode_iterations)
    mask = cv2.dilate(mask, kernel=morph_kernel, iterations=dilate_iterations)

    # Overwrite the drawing frame with the mask for debugging.
    if drawing_frame is not None:
        drawing_frame[:] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return mask

def get_line_candidates(frame, drawing_frame=None,
    min_area=2000,
    min_length=90,
):
    # Helper function to get line-related info from a contour
    def get_contour_line_info(c, fix_vert=True):
        # Fit a line to the contour
        vx, vy, cx, cy = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        
        # Project contour points onto the line's direction vector.
        projections = [((pt[0][0] - cx) * vx + (pt[0][1] - cy) * vy) for pt in c]
        min_proj = min(projections)
        max_proj = max(projections)
        
        # Compute endpoints from the extreme projection values.
        pt1 = (int(cx + vx * min_proj), int(cy + vy * min_proj))
        pt2 = (int(cx + vx * max_proj), int(cy + vy * max_proj))
        
        # Calculate the line angle in degrees.
        angle = math.degrees(math.atan2(vy, vx))
        if fix_vert:
            angle = angle - 90 * np.sign(angle)
        
        # Calculate the line length given pt1 and pt2.
        length = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
        
        return pt1, pt2, angle, cx, cy, length

    mask = get_line_mask(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > min_area]

    lines = [ get_contour_line_info(c) for c in contours ]
    lines = zip(contours, lines)
    lines = [l for l in lines if l[1][5] > min_length]  # Filter by length
    if drawing_frame is not None:
        contours = [l[0] for l in lines]

        # Draw the lines on the drawing frame
        for i, l in enumerate(lines):
            contour, (pt1, pt2, angle, cx, cy, _) = l # Unpack the tuple
            cv2.drawContours(drawing_frame, [contour], -1, (0, 255, 255), 2)
            cv2.line(drawing_frame, pt1, pt2, (0, 255, 255), 2)
            cv2.putText(drawing_frame, str(i), (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
    return zip(*lines) if lines else ([], [])

def get_middle_line(frame, drawing_frame=None):
    # Helper function to sort the contours
    def line_key(l):
        l = l[1]
        # Define the maximum angle for clamping
        max_angle = 80  # Adjust this value as needed
        # Get the direction of the line and its centroid
        _, _, angle, cx, cy, _ = l
        angle = max(min(angle, max_angle), -max_angle)
        # Compute ref_x based on angle: 0° -> center, +max_angle -> left, -max_angle -> right.
        ref_x = (frame_width / 2) + (angle / max_angle) * (frame_width / 2)
        # Draw ref_x on the frame for debugging
        # cv2.line(drawing_frame, (int(ref_x), 0), (int(ref_x), frame_height), (0, 0, 255), 2)
        # Compute the error between the centroid and our adjusted reference.
        x_err = abs(cx - ref_x)
        # Return a tuple for sorting: first sort by lowest centroid (i.e. largest cy) then by x error.
        return (x_err)

    # Frame size    
    frame_height, frame_width = frame.shape[:2]

    # Get the line candidates
    contours, lines = get_line_candidates(frame)
    lines = zip(contours, lines)

    if contours:
        # Sort by key
        lines = sorted(lines, key=line_key)

        # Choose the best candidate
        best_line = lines[0]
        
        # Draw the best candidate in green and the others in red.
        if drawing_frame is not None:
            cv2.drawContours(drawing_frame, [best_line[0]], -1, (0, 255, 0), 2)
            cv2.drawContours(drawing_frame, [c[0] for c in lines[1:]], -1, (0, 0, 255), 2)

        # Return the zipped line
        return best_line

def follow_line(frame, drawing_frame=None,
    Kp=0.6, Ki=0, Kd=0.1, # PID parameters
    max_yaw=math.radians(60), # Maximum yaw in radians
    max_thr=0.2, # Maximum throttle
    align_thres = 0.3 # Throttle will be max_thr when aligned, 0 at the threshold, and negative below the threshold.
):
    """ Follow the line in the frame """

    # Static variables
    follow_line.yaw_pid = follow_line.yaw_pid if hasattr(follow_line, "yaw_pid") else PID(Kp=Kp, Ki=Ki, Kd=Kd, setpoint=0.0, output_limits=(-max_yaw, max_yaw))

    # Get and unpck the line
    line = get_middle_line(frame, drawing_frame=drawing_frame)

    throttle, yaw = 0, 0
    if line:
        contour, (pt1, pt2, cx, cy, angle, length) = line
        # Get the X position of the line in the frame.
        frame_height, frame_width = frame.shape[:2]
        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w // 2
        normalized_x = (center_x - (frame_width/2)) / (frame_width/2)
        
        # Adjust yaw to keep the line centered in the frame.
        yaw = follow_line.yaw_pid(normalized_x)
        
        # Decrease throttle as the line moves away from the center.
        alignment = 1 - abs(normalized_x) # 1 when centered, 0 when at the edge.
        x =  ((alignment - align_thres) / (1 - align_thres)) # From 1 to -1
        T = 0.7
        m = 0.5
        # thr_factor = m * x if x <= T else m * T + m * (x - T) + ((1 - m) / ((1 - T) ** 2)) * (x - T) ** 2
        thr_factor = x
        throttle = max_thr * thr_factor

        # Optionally draw stats on the frame
        if drawing_frame is not None:
            cv2.line(drawing_frame, (center_x, 0), (center_x, frame_height), (255, 0, 0), 2)
            cv2.putText(drawing_frame, f"v: {throttle:.2f} m/s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(drawing_frame, f"w: {math.degrees(yaw):.2f} deg/s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        # Write "Searching for line" on the frame
        cv2.putText(drawing_frame, "Searching for line", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    return throttle, yaw  # Comment this line to disable output
    return 0, 0
