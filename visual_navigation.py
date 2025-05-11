from simple_pid import PID
import numpy as np
import math
import cv2
import time

def navigate_intersection(frame):
    # Static variables
    max_yaw = math.radians(30)
    max_thr = 0.2
    yaw_threshold = 5.0  # The robot will start moving forward when the target is this many degrees from the center
    if not hasattr(navigate_intersection, "pids"):
        navigate_intersection.pids = {
            "w_pid": PID(2.0, 0, 0.1, setpoint=0, output_limits=(-max_yaw, max_yaw)),
            "v_pid": PID(0.5, 0, 0.1, setpoint=0.8, output_limits=(-max_thr, max_thr)),
        }
    w_pid = navigate_intersection.pids["w_pid"]
    v_pid = navigate_intersection.pids["v_pid"]
    throttle, yaw = 0, 0

    def get_dotted_lines(frame):
        def find_dots(frame):
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
                    # cv2.circle(frame, center, 5, (0, 0, 255), -1) # Optionally, Draw the detected dot on the image
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

        dots = find_dots(frame)
        groups = cluster_collinear_points(dots)
        dotted_lines = [find_line_endpoints(group) for group in groups]
        return dotted_lines
    
    def identify_directions(dotted_lines):
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
        if not front:
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
            cv2.fillPoly(frame, [np.array([center, pt1, pt2], np.int32)], (0, 255, 0)) # Draw the triangle as a filled polygon
        return directions

    dotted_lines = get_dotted_lines(frame)
    back, left, right, front = identify_directions(dotted_lines)

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

if __name__ == "__main__":
    # Load intesection.png
    frame = cv2.imread("intersection4.png")

    # For visualization, draw a line connecting the endpoints for each detected dotted line group.
    navigate_intersection(frame)

    # Display the result.
    cv2.imshow("Display", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
