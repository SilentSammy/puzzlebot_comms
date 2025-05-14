from simple_pid import PID
import numpy as np
import math
import cv2
import time
import pose_estimation as pe
import image_test as it
import keybrd

def navigate_to_marker(frame, drawing_frame=None):
    drawing_frame = drawing_frame if drawing_frame is not None else frame.copy()
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

    markers, ids = pe.find_arucos(frame, drawing_frame=drawing_frame)
    if markers and ids is not None:
        # Find the marker with the lowest ID
        min_id_index = np.argmin(ids)
        marker = markers[min_id_index]
        # Remove extra nesting if necessary; otherwise, pass marker directly.
        _, _, _, cam_dist, _, cam_yaw = pe.estimate_marker_pose( marker_corners=marker, frame=frame, ref_size=0.063300535, fov_x=math.radians(60) )
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
        cv2.putText(drawing_frame, "Searching for ArUco", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return throttle, yaw

def follow_line(frame, drawing_frame=None):
    # If drawing_frame is provided, use it for drawing; otherwise, draw on a throwaway copy of the frame (inneficient but I'll fix it later).
    drawing_frame = drawing_frame if drawing_frame is not None else frame.copy()

    frame_height, frame_width = frame.shape[:2]

    # Static variables
    max_yaw = math.radians(60)
    max_thr = 0.2
    if not hasattr(follow_line, "yaw_pid"):
        follow_line.yaw_pid = PID(Kp=0.6, Ki=0, Kd=0.1, setpoint=0.0, output_limits=(-max_yaw, max_yaw))

    def get_line_candidates():
        # Convert to binary mask, only keeping pixels darker than dark_thres.
        dark_thres = 120
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, dark_thres, 255, cv2.THRESH_BINARY_INV)
        
        # Shrink the vertical field of view to the lower part of the frame.
        v_fov = 0.4
        mask[:int(frame_height * (1-v_fov)), :] = 0
        
        # Erode and dilate to remove noise and fill gaps.
        mask = cv2.erode(mask, kernel=np.ones((3, 3), np.uint8), iterations=5)
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
        cv2.line(drawing_frame, pt1, pt2, (0, 255, 0), 2)
        # cv2.putText(drawing_frame, str(i), (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
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
            # cv2.line(drawing_frame, (int(ref_x), 0), (int(ref_x), frame_height), (0, 0, 255), 2)
            # Compute the error between the centroid and our adjusted reference.
            x_err = abs(cx - ref_x)
            # Return a tuple for sorting: first sort by lowest centroid (i.e. largest cy) then by x error.
            return (x_err)
            
        # Choose the best candidate
        sorted_contours = sorted(contours, key=contour_key)
        line_contour = sorted_contours[0]
        
        # Draw the best candidate in green and the others in red.
        cv2.drawContours(drawing_frame, [line_contour], -1, (0, 255, 0), 2)
        for c in sorted_contours[1:]:
            cv2.drawContours(drawing_frame, [c], -1, (0, 0, 255), 2)

        # Get the X position of the line in the frame.
        x, y, w, h = cv2.boundingRect(line_contour)
        center_x = x + w // 2
        normalized_x = (center_x - (frame_width/2)) / (frame_width/2)
        cv2.line(drawing_frame, (center_x, 0), (center_x, frame_height), (255, 0, 0), 2)
        
        # Adjust yaw to keep the line centered in the frame.
        yaw = follow_line.yaw_pid(normalized_x)
        
        # Decrease throttle as the line moves away from the center.
        alignment = 1 - abs(normalized_x) # 1 when centered, 0 when at the edge.
        align_thres = 0.3 # Throttle will be max_thr when aligned, 0 at the threshold, and negative below the threshold.
        throttle = max_thr * ((alignment - align_thres) / (1 - align_thres))
    else:
        # Write "Searching for line" on the frame
        cv2.putText(drawing_frame, "Searching for line", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    return throttle, yaw  # Comment this line to disable output
    return 0, 0

def identify_intersection(frame, drawing_frame=None):
    drawing_frame = drawing_frame if drawing_frame is not None else frame.copy()
    def get_dotted_lines(frame, drawing_frame=None):
        def find_dots(frame, drawing_frame=None):
            drawing_frame = drawing_frame if drawing_frame is not None else frame.copy()
            # Adaptive thresholding to create a binary mask
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            mask = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 5)

            # Find quadrilateral contours in the mask with sufficient area
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [c for c in contours if cv2.contourArea(c) > 20]
            dots = []

            # Maximum allowed aspect ratio (long side divided by short side)
            max_aspect_ratio = 10.0

            for cnt in contours:
                # Approximate the contour to a polygon
                epsilon = 0.03 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                # Check if the approximated contour has 4 points (quadrilateral) and is convex.
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    x, y, w, h = cv2.boundingRect(approx)
                    # Filter out quadrilaterals that are too elongated
                    if min(w, h) == 0 or max(w, h) / min(w, h) > max_aspect_ratio:
                        continue
                    center = (x + w // 2, y + h // 2)
                    dots.append(center)
                    # Optionally, draw the detected dot on the image
                    cv2.circle(drawing_frame, center, 5, (0, 0, 255), -1)
                    cv2.polylines(drawing_frame, [approx], True, (0, 255, 0), 2)

            return dots

        def cluster_collinear_points(dots, min_points=5, threshold=5, outlier_factor=10.0):
            """ 
            Groups dots that are roughly collinear.
            - threshold: maximum perpendicular distance (in pixels) allowed from the candidate line.
            - outlier_factor: if a point's closest neighbor distance is greater than outlier_factor times
            the average gap between points on the candidate line, then that point is considered an outlier.
            """
            remaining = dots.copy()
            groups = []

            while len(remaining) >= min_points:
                best_group = []
                best_line = None

                # Test every pair as a candidate line.
                for i in range(len(remaining)):
                    for j in range(i+1, len(remaining)):
                        p1 = remaining[i]
                        p2 = remaining[j]
                        # Compute line parameters in the form: ax + by + c = 0.
                        if p2[0] - p1[0] == 0:
                            a = 1
                            b = 0
                            c = -p1[0]
                            direction = (0, 1)
                        else:
                            slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
                            a = -slope
                            b = 1
                            c = -(a * p1[0] + b * p1[1])
                            dx = p2[0] - p1[0]
                            dy = p2[1] - p1[1]
                            L = math.hypot(dx, dy)
                            direction = (dx / L, dy / L)

                        # Gather candidate dots that lie within the perpendicular distance threshold.
                        candidate_group = []
                        for p in remaining:
                            dist = abs(a * p[0] + b * p[1] + c) / math.sqrt(a**2 + b**2)
                            if dist < threshold:
                                candidate_group.append(p)

                        # Remove outliers based on the neighbors along the candidate line.
                        if len(candidate_group) >= 2:
                            # Project each point onto the candidate line.
                            projections = [((pt[0] * direction[0] + pt[1] * direction[1]), pt)
                                        for pt in candidate_group]
                            projections.sort(key=lambda x: x[0])
                            
                            # Compute average gap between consecutive projected points.
                            proj_values = [proj for proj, _ in projections]
                            if len(proj_values) > 1:
                                avg_gap = (proj_values[-1] - proj_values[0]) / (len(proj_values) - 1)
                            else:
                                avg_gap = 0

                            filtered = []
                            for idx, (proj, pt) in enumerate(projections):
                                if idx == 0:
                                    gap = proj_values[1] - proj
                                elif idx == len(proj_values) - 1:
                                    gap = proj - proj_values[idx - 1]
                                else:
                                    gap = min(proj - proj_values[idx - 1], proj_values[idx + 1] - proj)
                                # If the point's nearest neighbor distance is within the acceptable range, keep it.
                                if gap <= outlier_factor * avg_gap:
                                    filtered.append(pt)
                            candidate_group = filtered

                        if len(candidate_group) > len(best_group):
                            best_group = candidate_group
                            best_line = (a, b, c)

                if len(best_group) >= min_points:
                    groups.append(best_group)
                    # Remove grouped dots from further consideration.
                    remaining = [p for p in remaining if p not in best_group]
                else:
                    break

            return groups
        
        def find_line_endpoints(points):
            endpoint1, endpoint2 = None, None
            max_distance = 0
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    d = math.hypot(points[j][0] - points[i][0], points[j][1] - points[i][1])
                    if d > max_distance:
                        max_distance = d
                        endpoint1, endpoint2 = points[i], points[j]
            return endpoint1, endpoint2

        dots = find_dots(frame, drawing_frame=drawing_frame)
        groups = cluster_collinear_points(dots)
        dotted_lines = [find_line_endpoints(group) for group in groups]

        # Optionally draw the lines on the image
        for line in dotted_lines:
            cv2.line(drawing_frame, line[0], line[1], (255, 0, 0), 2)

        return dotted_lines

    dotted_lines = get_dotted_lines(frame, drawing_frame=drawing_frame)
    centers = [((l[0][0] + l[1][0]) // 2, (l[0][1] + l[1][1]) // 2) for l in dotted_lines]
    angles = [((math.degrees(math.atan2(l[1][1] - l[0][1], l[1][0] - l[0][0])) + 90) % 180) - 90 for l in dotted_lines]
    dotted_lines = list(zip(dotted_lines, centers, angles))
    
    # Separate into vertical and horizontal lines by angle
    vert_thres = 30
    verticals = [dl for dl in dotted_lines if abs(dl[2]) > vert_thres]
    horizontals = [dl for dl in dotted_lines if abs(dl[2]) <= vert_thres]

    # Identify the back, left, right, and front lines
    proximity_sorted = [hor for hor in horizontals if hor[1][1] / frame.shape[0] >= 0.3]
    proximity_sorted = iter(sorted(proximity_sorted, key=lambda x: x[1][1], reverse=True))
    mid_x = frame.shape[1] / 2  # Half the frame width
    left_verticals = [dl for dl in verticals if dl[1][0] < mid_x]
    left = next(iter(sorted(left_verticals, key=lambda x: x[1][0], reverse=True)), None)
    right_verticals = [dl for dl in verticals if dl[1][0] > mid_x and dl != left]
    right = next(iter(sorted(right_verticals, key=lambda x: x[1][0])), None)
    back = next(proximity_sorted, None)
    front = next(proximity_sorted, None)
    proximity_sorted = iter(sorted(dotted_lines, key=lambda x: x[1][1], reverse=True))
    if back != next(proximity_sorted, None): # If the back line is not the first in the sorted list, it means we have a front line
        front = back
        back = None

    # Draw a triangle for each available direction. One of the vertices is the center of the line
    directions = [back, left, right, front]
    for dl in filter(None, directions):
        line, center, angle = dl
        angle_rad_base = math.radians(angle * 1.8 + (90 if dl != back else -90))
        angle_rad1 = angle_rad_base + math.radians(15)
        angle_rad2 = angle_rad_base - math.radians(15)
        h = 50
        pt1 = int(center[0] + h * math.cos(angle_rad1)), int(center[1] + h * math.sin(angle_rad1))
        pt2 = int(center[0] + h * math.cos(angle_rad2)), int(center[1] + h * math.sin(angle_rad2))
        cv2.fillPoly(drawing_frame, [np.array([center, pt1, pt2], np.int32)], (0, 255, 0)) # Draw the triangle as a filled polygon
    return directions

def stop_at_intersection(frame, drawing_frame=None, intersection=None):
    # Static variables
    max_yaw = math.radians(30)
    max_thr = 0.2
    yaw_threshold = 5.0  # The robot will start translation when the target is this many degrees from the center
    if not hasattr(stop_at_intersection, "pids"):
        stop_at_intersection.pids = {
            "w_pid": PID(2.0, 0, 0.1, setpoint=0, output_limits=(-max_yaw, max_yaw)),
            "v_pid": PID(0.5, 0, 0.1, setpoint=0.8, output_limits=(-max_thr, max_thr)),
        }
    w_pid = stop_at_intersection.pids["w_pid"]
    v_pid = stop_at_intersection.pids["v_pid"]
    throttle, yaw = 0, 0

    if intersection:
        back, left, right, front = intersection
    else:
        back, left, right, front = identify_intersection(frame, drawing_frame=drawing_frame)

    # Align the robot with the intersection
    if back or front:
        angle = back[2] if back else front[2]
        error = math.radians(angle)
        yaw = w_pid(error)
        alpha = 1 - (abs(error) / yaw_threshold) if abs(error) < yaw_threshold else 0
        norm_y = back[1][1] / frame.shape[0] if back else 1.0
        measured_distance = (1 - alpha) * v_pid.setpoint + alpha * norm_y
        throttle = v_pid(measured_distance)

    return throttle, yaw

def navigate_track(frame, drawing_frame=None, decision_func=None):
    def random_decision(intersection):
        avail_idxs = [i for i, d in enumerate(intersection) if d is not None and i != 0]
        chosen_idx = np.random.choice(avail_idxs)
        dir_labels = ["back", "left", "right", "front"]
        print(f"Available directions: {[dir_labels[i] for i in avail_idxs]}; Going: {dir_labels[chosen_idx]}")
        return chosen_idx

    # Define sequences of actions (v, w, t)
    backward = [ (0, math.radians(30), 6) ]
    turn_left = [
        (0.15, 0, 2), # Move 30cm forward
        (0, math.radians(30), 3), # left 90° turn
        (0.15, 0, 2), # Move 30cm forward
    ]
    turn_right = [
        (0.15, 0, 2), # Move 30cm forward
        (0, -math.radians(30), 3), # right 90° turn
        (0.15, 0, 2), # Move 30cm forward
    ]
    forward = [ (0.15, 0, 4) ]
    actions = [backward, turn_left, turn_right, forward]
    decision_func = decision_func or random_decision

    # Static variables
    navigate_track.tmr = navigate_track.tmr if hasattr(navigate_track, "tmr") else 0
    navigate_track.action_index = navigate_track.action_index if hasattr(navigate_track, "action_index") else -1
    navigate_track.stoplight = navigate_track.stoplight if hasattr(navigate_track, "stoplight") else 2

    # If an action is in progress execute it.
    if navigate_track.action_index != -1:
        def reset_action():
            navigate_track.action_index = -1
        thr, yaw = sequence(actions=actions[navigate_track.action_index], when_done=reset_action)
        return thr, yaw

    # Determine the speed factor based on the stoplight.
    stoplight = identify_stoplight(frame, drawing_frame=drawing_frame)
    if stoplight is not None and stoplight != 1: # If red or green, remember it
        navigate_track.stoplight = stoplight
    speed_factor = (stoplight or navigate_track.stoplight) * 0.5

    # Attempt to identify an intersection.
    # intersection = identify_intersection(frame, drawing_frame=drawing_frame)
    intersection = [None, None, None, None]

    # Are we close enough to the intersection?
    inter_hor = intersection[0] or intersection[3] if intersection else None
    inter_hor_y = inter_hor[1][1] / frame.shape[0] if inter_hor else 0
    close_enough = inter_hor_y >= (1 - 0.4)

    if close_enough: # If the robot is close to the intersection, stop and choose a random direction.
        thr, yaw = stop_at_intersection(frame=frame, drawing_frame=drawing_frame, intersection=intersection) # Passing the detected intersection to avoid double processing.
        
        # Wait for the robot to stabilize and for the stoplight to be green.
        if not (abs(thr) < 0.02 and abs(thr) < 0.02) or speed_factor != 1:
            navigate_track.tmr = time.time() # Reset the timer (keep waiting)
        
        # If the robot has been stable for 2 seconds choose a random index from the available directions
        if time.time() - navigate_track.tmr > 1:
            navigate_track.action_index = decision_func(intersection) # Poll the decision function

    else: # No intersection detected, so we should follow the line.
        thr, yaw = follow_line(frame, drawing_frame=drawing_frame)
        navigate_track.tmr = time.time() # Reset the timer
        thr *= speed_factor
    return thr, yaw

def waypoint_navigation(frame, drawing_frame=None):
    waypoint_navigation.stoplight = waypoint_navigation.stoplight if hasattr(waypoint_navigation, "stoplight") else 2
    waypoint_navigation.last_time = waypoint_navigation.last_time if hasattr(waypoint_navigation, "last_time") else time.time()
    waypoint_navigation.elapsed_time = waypoint_navigation.elapsed_time if hasattr(waypoint_navigation, "elapsed_time") else 0
    
    # Determine the speed factor based on the stoplight.
    stoplight = identify_stoplight(frame, drawing_frame=drawing_frame)
    if stoplight is not None and stoplight != 1: # If red or green, remember it
        waypoint_navigation.stoplight = stoplight
    speed_factor = (stoplight or waypoint_navigation.stoplight) * 0.5
    s = speed_factor

    elapsed = time.time() - waypoint_navigation.last_time
    waypoint_navigation.elapsed_time += elapsed * s
    waypoint_navigation.last_time = time.time()


    # Define the sequence of actions (v, w, t)
    actions = [
        (0.2, 0, 2), # Move 50cm forward
        (0, -math.radians(90), 2), # 180° turn
    ]
    actions = [(v*s, w*s, t) for v, w, t in actions] # Scale the actions by s
    total_time = sum([t for _, _, t in actions])
    waypoint_navigation.elapsed_time = waypoint_navigation.elapsed_time % total_time
    print(waypoint_navigation.elapsed_time)
    throttle, yaw = sequence(actions=actions, elapsed_time=waypoint_navigation.elapsed_time)

    return throttle, yaw

def identify_stoplight(frame, drawing_frame = None):
    def find_color_ellipses(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ellipses = []
        for cnt in contours:
            # 1. Filter out contours with fewer than n points.
            if len(cnt) < 5:
                continue
            # 2. Fit an ellipse.
            ellipse = cv2.fitEllipse(cnt)
            (center_x, center_y), (axis1, axis2), angle = ellipse
            # 3. Skip invalid ellipses.
            if axis1 <= 0 or axis2 <= 0:
                continue
            # 4. Create a filled mask for the ellipse.
            ellipse_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.ellipse(ellipse_mask, ellipse, 255, thickness=-1)
            # 5. Count pixels in the ellipse that also appear in the original mask.
            filled = cv2.countNonZero(cv2.bitwise_and(mask, mask, mask=ellipse_mask))
            # 6. Compute the theoretical ellipse area.
            ellipse_area = math.pi * (axis1 / 2) * (axis2 / 2)
            fill_ratio = filled / ellipse_area if ellipse_area > 0 else 0
            # 7. Accept only if fill ratio is >= 0.9.
            if fill_ratio < 0.8:
                continue
            # 8. Filter out ellipses where the minor axis is less than 0.5 of the major axis.
            if min(axis1, axis2) < 0.5 * max(axis1, axis2):
                continue
            ellipses.append(ellipse)
        return ellipses

    # Define hsv ranges
    red = ((0, 150, 100), (10, 255, 255))
    green = ((50, 150, 100), (70, 255, 255))
    yellow = ((25, 150, 100), (35, 255, 255))

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks for each color.
    red_mask = cv2.inRange(hsv, red[0], red[1])
    green_mask = cv2.inRange(hsv, green[0], green[1])
    yellow_mask = cv2.inRange(hsv, yellow[0], yellow[1])

    # Dilate just a bit
    red_mask = cv2.dilate(red_mask, kernel=np.ones((3, 3), np.uint8), iterations=1)
    green_mask = cv2.dilate(green_mask, kernel=np.ones((3, 3), np.uint8), iterations=1)
    yellow_mask = cv2.dilate(yellow_mask, kernel=np.ones((3, 3), np.uint8), iterations=1)

    # Combine masks for displaying purposes.
    # combined_mask = cv2.addWeighted(red_mask, 1, green_mask, 1, 0)
    # combined_mask = cv2.addWeighted(combined_mask, 1, yellow_mask, 1, 0)
    # if drawing_frame is not None:
    #     combined_mask_color = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
    #     drawing_frame[:] = combined_mask_color

    # Process each color.
    red_ellipses = find_color_ellipses(red_mask)
    green_ellipses = find_color_ellipses(green_mask)
    yellow_ellipses = find_color_ellipses(yellow_mask)

    # Draw ellipses on the drawing frame.
    if drawing_frame is not None:
        for ellipse in red_ellipses:
            cv2.ellipse(drawing_frame, ellipse, (255, 255, 255), 2)
        for ellipse in green_ellipses:
            cv2.ellipse(drawing_frame, ellipse, (255, 255, 255), 2)
        for ellipse in yellow_ellipses:
            cv2.ellipse(drawing_frame, ellipse, (255, 255, 255), 2)
    
    if red_ellipses:
        return 0
    elif yellow_ellipses:
        return 1
    elif green_ellipses:
        return 2
    else:
        return None

def sequence(actions=None, when_done=None, elapsed_time=None):
    # Static variables
    if not hasattr(sequence, "start_time"):
        sequence.start_time = time.time()
    
    # Define the sequence of actions or use default
    actions = actions or [ # v, w, t
        (0.15, 0, 2), # Move 30cm forward
        (0, -math.radians(30), 3), # 90° turn
        (0.15, 0, 2), # Move 30cm forward
    ]

    # Get the elapsed time
    elapsed_time = elapsed_time or (time.time() - sequence.start_time)

    action = None
    ac_time = 0
    for i, act in enumerate(actions):
        ac_time += act[2]
        if elapsed_time < ac_time:
            action = act
            break
    
    throttle, yaw = 0, 0
    if action is None: # done
        del sequence.start_time
        if when_done:
            when_done()
    else:
        throttle, yaw, _ = action
    return throttle, yaw

def undistort_fisheye(img):
    h, w = img.shape[:2]
    sensor_w_mm = 3.68
    sensor_h_mm = 2.76
    f_mm = 3.15
    fx = (f_mm / sensor_w_mm) * w
    fy = (f_mm / sensor_h_mm) * h
    cx = w / 2.0
    cy = h / 2.0
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float64)
    D = np.array([0, 0, 0, 0], dtype=np.float64)

    # Undistort the image
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=0.0
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
    )
    undistorted = cv2.remap(
        img, map1, map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )
    return undistorted

def color_correct(img):
    """
    Removes a purple-ish tint around the borders by applying an inverse lens shading correction.
    Assumes that at the image edges the gains are approximately:
      - Red gain ~ 4.2
      - Green gain ~ 3.26
      - Blue gain ~ 3.12
    This function builds a radial correction map that is 1 at the center and transitions linearly toward 1/edge_gain at the corners.
    The full map is then re-normalized to preserve overall brightness.
    """
    # Convert to float for processing.
    img_float = img.astype(np.float32)
    height, width = img.shape[:2]
    cx, cy = width / 2.0, height / 2.0
    
    # Maximum radial distance (from center to corner)
    max_dist = np.sqrt(cx**2 + cy**2)

    # Create a grid of (x,y) coordinates and compute normalized radial distance (0 at center, 1 at corner)
    x = np.arange(width)
    y = np.arange(height)
    xv, yv = np.meshgrid(x, y)
    dist = np.sqrt((xv - cx)**2 + (yv - cy)**2)
    r = dist / max_dist

    # Define the edge gains from calibration/lens shading tables.
    red_edge_gain = 4.2
    green_edge_gain = 3.26
    blue_edge_gain = 3.12

    # Compute per-channel correction factors:
    # At r=0, factor is 1.
    # At r=1, factor is 1/edge_gain.
    corr_red = 1.0 - r * (1.0 - (1.0 / red_edge_gain))
    corr_green = 1.0 - r * (1.0 - (1.0 / green_edge_gain))
    corr_blue = 1.0 - r * (1.0 - (1.0 / blue_edge_gain))
    
    # Build the correction map (note: OpenCV uses BGR order)
    raw_map = np.stack([corr_blue, corr_green, corr_red], axis=-1)
    
    # Re-normalize the map so that its mean is 1, preserving overall brightness.
    norm_factor = np.mean(raw_map)
    correction_map = raw_map / norm_factor

    # Apply the correction map.
    corrected = img_float * correction_map

    # Clip to valid 8-bit range and convert back to uint8.
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return corrected

def correct_purple(frame: np.ndarray,
                   strength: float = 1.0,
                   radius_ratio: float = 0.8) -> np.ndarray:
    """
    Reduce purple-ish hue at frame edges by radially desaturating magenta.
    - strength: how aggressively to desaturate (0…1).
    - radius_ratio: inner radius (as fraction of diag) before desaturation starts.
    """
    h, w = frame.shape[:2]
    # Create normalized radial mask
    xv, yv = np.meshgrid(np.linspace(0, w-1, w),
                         np.linspace(0, h-1, h))
    dx = xv - (w/2); dy = yv - (h/2)
    dist = np.sqrt(dx*dx + dy*dy)
    max_dist = np.sqrt((w/2)**2 + (h/2)**2)
    # mask = 0 at center, 1 at edges beyond radius_ratio*max_dist
    mask = np.clip((dist - radius_ratio*max_dist) /
                   (max_dist - radius_ratio*max_dist), 0, 1)
    mask = cv2.GaussianBlur(mask, (101,101), 0)  # smooth transition

    # Convert to HSV for saturation control
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    # Desaturate edges
    hsv[...,1] *= (1.0 - strength * mask)
    hsv[...,1] = np.clip(hsv[...,1], 0, 255)

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

if __name__ == "__main__":
    images = [
        "./resources/screenshots/intersection4.png",
        "./resources/screenshots/irl_intersection.png",
        "./resources/screenshots/irl_intersection2.png",
        "./resources/screenshots/sl_green.png",
        "./resources/screenshots/sl_red.png",
        "./resources/screenshots/sl_yellow.png",
        "./resources/screenshots/green_circle.png",
        "./resources/screenshots/red_circle.png",
        "./resources/screenshots/yellow_circle.png",
    ]
    frame = cv2.imread(images[-2])
    # frame = undistort_fisheye(frame)
    # frame = color_correct(frame)
    frame = correct_purple(frame, strength=2.0, radius_ratio=0.9)
    # identify_intersection(frame, frame)
    print(identify_stoplight(frame, frame))

    # Display the result.
    cv2.imshow("Display", frame)
    # cv2.imshow("Display2", frame2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
