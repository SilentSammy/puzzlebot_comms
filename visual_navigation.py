from simple_pid import PID
import numpy as np
import math
import cv2
import time
import pose_estimation as pe

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
            # Load the image and convert to grayscale
            dark_thres = 100
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, dark_thres, 255, cv2.THRESH_BINARY_INV)

            # Find quadrilateral contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            dots = []

            # Maximum allowed aspect ratio (long side divided by short side)
            max_aspect_ratio = 10.0

            for cnt in contours:
                # Approximate the contour to a polygon
                epsilon = 0.03 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                # Check if the approximated contour has 4 points (quadrilateral)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    # Filter out quadrilaterals that are too elongated
                    if min(w, h) == 0 or max(w, h) / min(w, h) > max_aspect_ratio:
                        continue
                    center = (x + w // 2, y + h // 2)
                    dots.append(center)
                    # cv2.circle(drawing_frame, center, 5, (0, 0, 255), -1) # Optionally, Draw the detected dot on the image
            return dots

        def cluster_collinear_points(dots, min_points=4, threshold=3, outlier_factor=2.0):
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
    left = next(iter(sorted(verticals, key=lambda x: x[1][0], reverse=True)), None)
    right = next(iter(sorted(verticals, key=lambda x: x[1][0])), None)
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

def stop_at_intersection(frame, drawing_frame=None):
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

def navigate_track(frame, drawing_frame=None):
    throttle, yaw = 0, 0
    thr, yw = follow_line(frame, drawing_frame=drawing_frame)
    throttle = thr
    yaw = yw
    return throttle, yaw

if __name__ == "__main__":
    # Load intesection.png
    frame = cv2.imread("intersection4.png")

    # For visualization, draw a line connecting the endpoints for each detected dotted line group.
    stop_at_intersection(frame)

    # Display the result.
    cv2.imshow("Display", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
