import threading
from simple_pid import PID
import numpy as np
import math
import cv2
import time
from collections import deque
from itertools import combinations, count

class LineFollower:
    def __init__(
        self,
        blur_kernel_size=(7, 7),
        adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        threshold_type=cv2.THRESH_BINARY_INV,
        block_size=141,
        c_value=6,
        v_fov=0.4,
        morph_kernel_size=(5, 5),
        erode_iterations=4,
        dilate_iterations=6,
        min_area=2000,
        min_length=90,
        lifespan=5,
        max_yaw=60,  # degrees
        max_thr=0.2,
        align_thres=0.2,
        Kp = 0.6,
        Ki = 0,
        Kd = 0.1
    ):
        # Adaptive thresholding parameters
        self.blur_kernel_size = blur_kernel_size
        self.adaptive_method = adaptive_method
        self.threshold_type = threshold_type
        self.block_size = block_size
        self.c_value = c_value

        # Line mask parameters
        self.v_fov = v_fov
        self.morph_kernel = np.ones(morph_kernel_size, np.uint8)
        self.erode_iterations = erode_iterations
        self.dilate_iterations = dilate_iterations

        # Line candidate parameters
        self.min_area = min_area
        self.min_length = min_length

        # Line tracking static variables
        self._id_gen = count(0)  # ID generator for line candidates
        self._old_lines = []  # List of old lines for tracking

        # Persistent line parameters
        self.lifespan = lifespan  # Number of frames to keep the line before resetting

        # Persistent line static variables
        self._chosen_id = -1  # ID of the currently chosen line
        self._upcoming_id = -1  # ID of the upcoming line
        self._upcoming_count = 0  # Count of frames the upcoming line has been seen

        # PID controller for line following
        self.max_thr = max_thr  # Maximum throttle
        self.align_thres = align_thres  # Throttle will be max_thr when aligned, 0 at the threshold, and negative below the threshold.
        self.yaw_pid = PID(Kp=Kp, Ki=Ki, Kd=Kd, setpoint=0.0, output_limits=(-math.radians(max_yaw), math.radians(max_yaw)))

    def adaptive_thres(self, frame, drawing_frame=None):
        mask = adaptive_thres(frame, drawing_frame=drawing_frame,
                              blur_kernel_size=self.blur_kernel_size,
                              adaptive_method=self.adaptive_method,
                              threshold_type=self.threshold_type,
                              block_size=self.block_size,
                              c_value=self.c_value)
        return mask

    def get_line_mask(self, frame, drawing_frame=None):
        # Get mask
        mask = self.adaptive_thres(frame)

        # Only keep the lower part of the mask, filling the upper part with black.
        mask[:int(frame.shape[:2][0] * (1-self.v_fov)), :] = 0

        # Erode and dilate to remove noise and fill gaps.
        mask = cv2.erode(mask, kernel=self.morph_kernel, iterations=self.erode_iterations)
        mask = cv2.dilate(mask, kernel=self.morph_kernel, iterations=self.dilate_iterations)

        # Overwrite the drawing frame with the mask for debugging.
        if drawing_frame is not None:
            drawing_frame[:] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return mask
    
    def get_line_candidates(self, frame, drawing_frame=None):
        mask = self.get_line_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > self.min_area]

        lines = [ get_contour_line_info(c) for c in contours ]
        lines = zip(contours, lines)
        lines = [l for l in lines if l[1][4] > self.min_length]  # Filter by length
        if drawing_frame is not None:
            contours = [l[0] for l in lines]

            # Draw the lines on the drawing frame
            for i, l in enumerate(lines):
                contour, (pt1, pt2, center, angle, length) = l # Unpack the tuple
                cv2.drawContours(drawing_frame, [contour], -1, (0, 255, 255), 2)
                cv2.line(drawing_frame, pt1, pt2, (0, 255, 255), 2)
        return zip(*lines) if lines else ([], [])

    def id_line_candidates(self, frame, drawing_frame=None):
        # Get line candidates
        contours, lines = self.get_line_candidates(frame, drawing_frame=drawing_frame)
        lines = zip(contours, lines)

        # Create a dictionary of line tuples -> contours for easy lookup
        # line_dict = {tuple(l): c for l, c in zip(lines, contours)}

        # Assign IDs to the new lines based on their proximity to old lines
        new_lines = [{'id': None, 'line': l, 'contour': c} for c, l in lines]
        self._old_lines, lost_lines = assign_tracked_ids(
            new_objs=new_lines,
            tracked_objs=self._old_lines,
            id_gen=lambda: next(self._id_gen),
            get_id=lambda obj: obj['id'],
            set_id=lambda obj, id: obj.update({'id': id}),
            get_pos=lambda obj: obj['line'][2], # Use the center of the line as the position
            upd_obj=lambda old_obj, new_obj: old_obj.update(new_obj),
            threshold_px=100,
            persist=True
        )

        # Clear lost objects
        clear_lost_objects(
            tracked_objs=self._old_lines,
            lost_objs=lost_lines,
            lost_timeout=0.1,
            is_lost=lambda obj: 'lost_time' in obj,
            get_id=lambda obj: obj['id'],
            get_lost_time=lambda obj: obj['lost_time'],
            set_lost_time=lambda obj, t: obj.update({'lost_time': t}),
            refind=lambda obj: obj.pop('lost_time', None)
        )

        if drawing_frame is not None:
            # Draw the lines on the drawing frame
            for i, l in enumerate(self._old_lines):
                pt1, pt2, center, angle, length = l["line"] # Unpack the tuple
                cv2.putText(drawing_frame, str(l["id"]), center, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        ids = [l["id"] for l in self._old_lines]
        lines = [l["line"] for l in self._old_lines]
        # contours = [line_dict[tuple(l)] for l in lines]
        contours = [l["contour"] for l in self._old_lines]

        return contours, lines, ids

    def get_middle_line(self, frame, drawing_frame=None, line_candidates=None):
        # Helper function to sort the contours
        def line_key(l):
            l = l[1]
            # Define the maximum angle for clamping
            max_angle = 80  # Adjust this value as needed
            # Get the direction of the line and its centroid
            _, _, center, angle, _ = l
            cx = center[0]
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
        if line_candidates is None:
            contours, lines, ids = self.id_line_candidates(frame, drawing_frame=drawing_frame)
        else:
            contours, lines, ids = line_candidates
        lines = zip(contours, lines, ids)

        if lines:
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

    def get_persistent_line(self, frame, drawing_frame=None):
        # Get the line candidates for this frame
        line_candidates = self.id_line_candidates(frame)
        if not line_candidates[0]:
            return None
        
        # Get the best line for this frame
        best_contour, best_line, best_id = self.get_middle_line(frame, line_candidates=line_candidates)
        
        # If a new best line is found, reset the upcoming line
        if best_id != self._upcoming_id:
            self._upcoming_id = best_id
            self._upcoming_count = 0
        
        # If the best line is the same as the upcoming line, increment the count
        if best_id == self._upcoming_id:
            self._upcoming_count += 1
        
        # If the upcoming line has been seen for 3 frames, choose it
        if self._upcoming_count >= self.lifespan:
            self._chosen_id = self._upcoming_id
        
        # If chosen_id is within the current candidates, use it
        chosen_line = next(( (c, l, id) for c, l, id in zip(*line_candidates) if id == self._chosen_id ), None)
        
        if drawing_frame is not None:
            # Draw the others in red
            for contour, line, id in zip(*line_candidates):
                cv2.drawContours(drawing_frame, [contour], -1, (0, 0, 255), 2)
                cv2.putText(drawing_frame, str(id), line[2], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Draw the chosen line in green
            if chosen_line is not None:
                cv2.drawContours(drawing_frame, [chosen_line[0]], -1, (0, 255, 0), 2)
            
            # Write the best ID in green
            if best_line is not None:
                cv2.putText(drawing_frame, str(best_id), best_line[2], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Return the chosen line, the best line, and the chosen ID
        if chosen_line is not None:
            return chosen_line[0], chosen_line[1], self._chosen_id

    def follow_line(self, frame, drawing_frame=None, authority=1.0):
        """ Follow the line in the frame """
        # Ensure authority is within [0, 1] range
        authority = max(0, min(authority, 1.0))

        # Get and unpack the line
        # line = get_middle_line(frame, drawing_frame=drawing_frame)
        line = self.get_persistent_line(frame, drawing_frame=drawing_frame)

        throttle, yaw = 0, 0
        if line:
            contour, (pt1, pt2, center, angle, length), id = line
            # Get the X position of the line in the frame.
            frame_height, frame_width = frame.shape[:2]
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            normalized_x = (center_x - (frame_width/2)) / (frame_width/2) # Normalize to [-1, 1] range
            
            # Adjust yaw to keep the line centered in the frame.
            yaw = self.yaw_pid(normalized_x * authority)  # Use authority to scale the PID output
            
            # Decrease throttle as the line moves away from the center.
            alignment = 1 - abs(normalized_x) # 1 when centered, 0 when at the edge.
            x =  ((alignment - self.align_thres) / (1 - self.align_thres)) # From 1 to -1
            thr_factor = x
            throttle = self.max_thr * thr_factor
            throttle *= authority  # Scale throttle by authority

            # Optionally draw stats on the frame
            if drawing_frame is not None:
                cv2.line(drawing_frame, (center_x, 0), (center_x, frame_height), (255, 0, 0), 2)
                cv2.putText(drawing_frame, f"v: {throttle:.2f} m/s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(drawing_frame, f"w: {math.degrees(yaw):.2f} deg/s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            # Write "Searching for line" on the frame
            cv2.putText(drawing_frame, "Searching for line", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        return throttle, yaw  # Comment this line to disable output
        return 0.0, 0.0

class StoplightDetector:
    def __init__(self,
                 red_target_hue=0,
                 red_hue_tol=10,
                 red_sat_thresh=60,
                 yellow_hue=30,
                 yellow_hue_tol=12,
                 yellow_sat_thresh=80,
                 green_hue=65,
                 green_hue_tol=20,
                 green_sat_thresh=30,
                 morph_kernel_size=(3, 3),
                 erode_iterations=3,
                 dilate_iterations=2,
                 min_fill_ratio=0.85,
                 max_outside_ratio=0.15,
                 max_tilt=0.5,
                 max_norm_major=0.8,
                 min_norm_major=0.02,
                 edge_thres_x=0.3,
                 edge_thres_y=0.0,
                 chain_length=4,
                 max_chain_gap=1,
                 history_len=5
        ):

        # Adaptive color thresholding parameters
        self.red_target_hue = red_target_hue
        self.red_hue_tol = red_hue_tol
        self.red_sat_thresh = red_sat_thresh
        self.yellow_hue = yellow_hue
        self.yellow_hue_tol = yellow_hue_tol
        self.yellow_sat_thresh = yellow_sat_thresh
        self.green_hue = green_hue
        self.green_hue_tol = green_hue_tol
        self.green_sat_thresh = green_sat_thresh

        # Ellipse mask parameters
        self.morph_kernel = np.ones(morph_kernel_size, np.uint8)
        self.erode_iterations = erode_iterations
        self.dilate_iterations = dilate_iterations
        self.min_fill_ratio = min_fill_ratio
        self.max_outside_ratio = max_outside_ratio
        self.max_tilt = max_tilt
        self.max_norm_major = max_norm_major
        self.min_norm_major = min_norm_major

        # Identify stoplight params
        self.edge_thres_x = edge_thres_x  # Fraction of the frame width
        self.edge_thres_y = edge_thres_y  # Fraction of the frame height
        self.chain_length = chain_length  # Consecutive frames to consider a color as seen
        self.max_chain_gap = max_chain_gap  # Maximum gap in frames to consider a color as seen

        # Identify stoplight static variables
        color_history_len = max(history_len, chain_length + max_chain_gap)
        self.red_history = deque(maxlen=color_history_len)
        self.yellow_history = deque(maxlen=color_history_len)
        self.green_history = deque(maxlen=color_history_len)

    def ellipse_mask(self, mask, drawing_frame=None):
        # Erode and dilate to remove noise and fill gaps.
        mask = cv2.erode(mask, kernel=self.morph_kernel, iterations=self.erode_iterations)
        mask = cv2.dilate(mask, kernel=self.morph_kernel, iterations=self.dilate_iterations)

        if drawing_frame is not None:
            drawing_frame[:] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Find ellipses in the mask
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        valid_contours = []
        ellipses = []
        valid_mask  = np.zeros_like(mask)
        for cnt in contours:
            # 1. Filter out contours with fewer than 5 points.
            if len(cnt) < 5:
                cv2.drawContours(drawing_frame, [cnt], -1, (0, 0, 255), 1)  # Draw contour in red
                continue
            # 2. Fit an ellipse.
            ellipse = cv2.fitEllipse(cnt)
            (center_x, center_y), (axis1, axis2), angle = ellipse

            # 2.1 Skip invalid ellipses
            if axis1 <= 0 or axis2 <= 0:
                continue

            # 3. Filter out ellipses with extreme aspect ratios.
            if min(axis1, axis2) < self.max_tilt * max(axis1, axis2):
                if drawing_frame is not None:
                    cv2.drawContours(drawing_frame, [cnt], -1, (255, 0, 255), 1)  # Draw contour in magenta
                continue

            # 4. Filter out ellipses too small or too large.
            norm_major = max(axis1, axis2) / mask.shape[1]
            if norm_major < self.min_norm_major or norm_major > self.max_norm_major:
                if drawing_frame is not None:
                    cv2.drawContours(drawing_frame, [cnt], -1, (0, 255, 255), 1)  # Draw contour in cyan
                continue

            # 5. Accept only if fill ratio is >= assumed threshold.
            ellipse_mask_img = np.zeros(mask.shape, dtype=np.uint8)
            cv2.ellipse(ellipse_mask_img, ellipse, 255, thickness=-1)
            filled = cv2.countNonZero(cv2.bitwise_and(mask, mask, mask=ellipse_mask_img))
            ellipse_pixels = cv2.countNonZero(ellipse_mask_img)  # Actual pixel count in the ellipse
            if ellipse_pixels == 0:
                continue
            fill_ratio = filled / ellipse_pixels
            if fill_ratio < self.min_fill_ratio:
                if drawing_frame is not None:
                    cv2.ellipse(drawing_frame, ellipse, (0, 255, 255), 2)  # Draw ellipse in yellow
                continue

            # 6. Outside-area ratio filter
            outside_pixels = ellipse_pixels - filled   # Pixels inside ellipse but not in the mask
            outside_ratio = outside_pixels / ellipse_pixels
            if outside_ratio > self.max_outside_ratio:
                if drawing_frame is not None:
                    cv2.ellipse(drawing_frame, ellipse, (255, 0, 0), 2)  # Draw ellipse in blue
                continue

            # Record the ellipse and the contour
            valid_mask = cv2.bitwise_or(valid_mask, ellipse_mask_img)
            ellipses.append(ellipse)
            valid_contours.append(cnt)
        
        if drawing_frame is not None:
            for ellipse in ellipses: # Draw valid ellipses in green
                cv2.ellipse(drawing_frame, ellipse, (0, 255, 0), 2)

        return valid_mask, ellipses

    def stoplight_mask(self, frame, drawing_frame=None):
        red_mask = adaptive_color_thresh( frame, target_hue=self.red_target_hue, hue_tol=self.red_hue_tol, sat_thresh=self.red_sat_thresh )
        yellow_mask = adaptive_color_thresh( frame, target_hue=self.yellow_hue, hue_tol=self.yellow_hue_tol, sat_thresh=self.yellow_sat_thresh )
        green_mask = adaptive_color_thresh( frame, target_hue=self.green_hue, hue_tol=self.green_hue_tol, sat_thresh=self.green_sat_thresh )

        red_mask, red_ellipses = self.ellipse_mask( red_mask )
        green_mask, green_ellipses = self.ellipse_mask( green_mask )
        yellow_mask, yellow_ellipses = self.ellipse_mask( yellow_mask )

        # AND all the masks together
        stoplight_mask = cv2.bitwise_or(red_mask, green_mask)
        stoplight_mask = cv2.bitwise_or(stoplight_mask, yellow_mask)

        # Overwrite the drawing frame with the mask for debugging.
        if drawing_frame is not None:
            # create a blank color image
            colored = np.zeros_like(frame)
            # apply color per mask
            colored[red_mask > 0]    = (0,   0,   255)  # BGR red
            colored[yellow_mask > 0] = (0,   255, 255)  # BGR yellow
            colored[green_mask > 0]  = (0,   255, 0)    # BGR green
            drawing_frame[:] = colored
        return stoplight_mask, red_ellipses, yellow_ellipses, green_ellipses
    
    def identify_stoplight(self, frame, drawing_frame=None):
        def is_near_edge(ellipse):
            # Unpack the ellipse: ((x, y), (a, b), angle)
            (x, y), (a, b), angle = ellipse

            frame_width = frame.shape[1]
            frame_height = frame.shape[0]

            # Normalize the center to [-1, 1]
            x_norm = (x - frame_width / 2) / (frame_width / 2)
            y_norm = (y - frame_height / 2) / (frame_height / 2)
            
            # Assume a perfect circle with diameter 'a'
            normalized_radius_x = a / frame_width
            normalized_radius_y = a / frame_height

            # Check if any part of the circle touches near the edges.
            near_x = abs(x_norm) + normalized_radius_x >= (1 - self.edge_thres_x)
            near_y = abs(y_norm) + normalized_radius_y >= (1 - self.edge_thres_y)
            
            return near_x or near_y
        
        def is_confirmed(history):
            return sum(history) >= (self.chain_length - self.max_chain_gap)
        
        # Process the frame
        mask, red_ellipses, yellow_ellipses, green_ellipses = self.stoplight_mask(frame)
        if drawing_frame is not None:
            grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grayscale = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
            drawing_frame[:] = grayscale
            for ellipse in red_ellipses:
                cv2.ellipse(drawing_frame, ellipse, (0, 0, 255), -1)
            for ellipse in yellow_ellipses:
                cv2.ellipse(drawing_frame, ellipse, (0, 255, 255), -1)
            for ellipse in green_ellipses:
                cv2.ellipse(drawing_frame, ellipse, (0, 255, 0), -1)
        
        # Filter out only red ellipses that are near the edge of the frame
        red_ellipses_fltrd = [e for e in red_ellipses if not is_near_edge(e)]
        
        # Update color histories with booleans
        self.red_history.append(bool(red_ellipses_fltrd))
        self.yellow_history.append(bool(yellow_ellipses))
        self.green_history.append(bool(green_ellipses))

        # Find all confirmed colors and their counts
        confirmed = []
        ellipses_by_color = [red_ellipses_fltrd, yellow_ellipses, green_ellipses]
        for color, history in enumerate([self.red_history, self.yellow_history, self.green_history]):
            if is_confirmed(history):
                count = sum(history)
                confirmed.append((color, count))
        # Return the confirmed color with the most appearances
        if confirmed:
            confirmed.sort(key=lambda x: x[1], reverse=True)
            chosen_color = confirmed[0][0]
            ellipses = ellipses_by_color[chosen_color]
            # Draw a white ellipse around the largest ellipse of the confirmed color
            if ellipses and drawing_frame is not None:
                # Find the largest by major axis
                largest = max(ellipses, key=lambda e: max(e[1]))
                cv2.ellipse(drawing_frame, largest, (255, 255, 255), 3)
            return chosen_color
        return None

class FlagDetector:
    def __init__(self,
            pattern_size=(4, 3),
            square_size=0.025,
            dist_thres=0.4,
        ):
        # Flag detection parameters
        self.pattern_size = pattern_size    # Chessboard pattern size
        self.square_size = square_size      # Size of each square in meters

        # Multi-threading static variables
        self._worker = None
        self._lock = threading.Lock()
        self._last_result = None
        self._last_drawing_frame = None

        self.dist_thres = dist_thres  # Distance threshold in meters to consider the flag close
        self.end_reached = False  # Flag to indicate if the end has been reached

    def get_flag_distance(self, frame, drawing_frame=None):
        """
        Estima Z (m) usando solo la altura en píxeles del patrón.
        Devuelve (Z, h_pix).
        """
        ret, corners = cv2.findChessboardCorners(frame, self.pattern_size, None)
        if not ret:
            return None
        f_y = K[1,1]
        ys = corners[:,:,1].flatten()
        h_pix = ys.max() - ys.min()
        # Altura real entre la primer y última fila de esquinas internas
        H_real = self.square_size * (self.pattern_size[1] - 1)
        dist = (f_y * H_real) / h_pix
        if drawing_frame is not None:
            # Draw text above the topmost chessboard corner
            cv2.drawChessboardCorners(drawing_frame, self.pattern_size, corners, ret)
            top_y = int(ys.min())
            left_x = int(corners[:,:,0].flatten().min())
            text1 = f"Z: {dist:.2f} m"
            text2 = f"h_pix: {h_pix:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            (tw1, th1), _ = cv2.getTextSize(text1, font, font_scale, thickness)
            (tw2, th2), _ = cv2.getTextSize(text2, font, font_scale, thickness)
            x = left_x
            y1 = max(top_y - 10, th1 + 5)
            y2 = y1 + th2 + 10
            cv2.putText(drawing_frame, text1, (x, y1), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
            cv2.putText(drawing_frame, text2, (x, y2), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
        return dist
    
    def get_flag_distance_nb(self, frame, drawing_frame=None):
        with self._lock:
            result = self._last_result
            annotated_frame = self._last_drawing_frame
            
        
        # If processing is not ongoing, process in background
        if self._worker is None or not self._worker.is_alive():
            def worker_func(frame_copy, drawing_frame):
                dist = self.get_flag_distance( frame_copy, drawing_frame=drawing_frame )
                with self._lock:
                    self._last_result = dist
                    self._last_drawing_frame = drawing_frame
        
            # Start the worker thread
            t = threading.Thread(
                target=worker_func,
                args=(frame.copy(), np.zeros_like(frame)),
            )
            t.daemon = True
            t.start()
            self._worker = t
        
        if drawing_frame is not None and annotated_frame is not None:
            # Overwrite the drawing_frame pixels with annotated_frame pixels wherever the mask is True.
            non_black_mask = np.any(annotated_frame != 0, axis=2)
            drawing_frame[non_black_mask] = annotated_frame[non_black_mask]

        return result

    def flag_reached(self, frame, drawing_frame=None, non_blocking=True):
        if not self.end_reached:
            if non_blocking:
                dist = self.get_flag_distance_nb(frame, drawing_frame=drawing_frame)
            else:
                dist = self.get_flag_distance(frame, drawing_frame=drawing_frame)
                
            if dist is not None and dist <= self.dist_thres:
                self.end_reached = True
        
        if drawing_frame is not None and self.end_reached:
            cv2.putText( drawing_frame, "Flag reached", (drawing_frame.shape[1] // 2 - 80, drawing_frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA )
        
        return self.end_reached

class StoplightNavigator:
    def __init__(self, line_follower=None, stoplight_detector=None, flag_detector=None, end_action=None):
        self.lf = line_follower or LineFollower()
        self.sd = stoplight_detector or StoplightDetector()
        self.fd = flag_detector or FlagDetector()
        self.end_action=end_action
        
        # Static variables
        self.tmr = 0  # Timer for stoplight detection
        self.stoplight = 2  # Default stoplight state (2 = yellow)
        self.end_reached = False  # Flag to indicate if the end has been reached
        self.stoplight
    
    def navigate(self, frame, drawing_frame=None):
        thr = yaw = 0
        if not self.fd.end_reached:
            # Determine the speed factor based on the stoplight.
            stoplight = self.sd.identify_stoplight(frame, drawing_frame=drawing_frame)
            if stoplight is not None and stoplight != 1: # If red or green, remember it
                self.stoplight = stoplight
            speed_factor = (stoplight or self.stoplight) * 0.5

            self.fd.flag_reached(frame, drawing_frame=drawing_frame, non_blocking=True)
            if not self.fd.end_reached:
                thr, yaw = self.lf.follow_line(frame, drawing_frame=drawing_frame, authority=speed_factor)
            elif self.end_action:
                self.end_action()
        else:
            self.fd.flag_reached(frame, drawing_frame=drawing_frame, non_blocking=True)
        return thr, yaw

class IntersectionDetector:
    def __init__(self,
        undistort=True,
        v_fov=0.55,
        morph_kernel_size=(3, 3),
        erode_iterations=3,
        dilate_iterations=2,
        max_aspect_ratio=10.0,
        min_area=20,
        ep=0.035,
        min_points=5,
        setpoint=0.7,
        max_yaw=30.0,
        max_thr=0.15,
        w_Kp=2.0,
        w_Ki=0.0,
        w_Kd=0.1,
        v_Kp=0.5,
        v_Ki=0.0,
        v_Kd=0.1,
    ):
        # Dark mask parameters
        self.undistort = undistort
        self.v_fov = v_fov  # Bottom field of view (0.6 = 60% of the frame height)
        self.morph_kernel = np.ones(morph_kernel_size, np.uint8)  # Kernel for morphological operations
        self.erode_iterations = erode_iterations  # Number of iterations for erosion
        self.dilate_iterations = dilate_iterations  # Number of iterations for dilation

        # Find dots parameters
        self.max_aspect_ratio = max_aspect_ratio
        self.min_area = min_area
        self.ep = ep  # Approximation factor for contour approximation

        # Dotted line parameters
        self.min_points = min_points  # Minimum number of points to consider a line

        # Stopping parameters
        self.yaw_threshold = 5.0
        self.w_pid = PID(w_Kp, w_Ki, w_Kd, setpoint=0, output_limits=(-max_yaw, max_yaw))
        self.v_pid = PID(v_Kp, v_Ki, v_Kd, setpoint=setpoint, output_limits=(-max_thr, max_thr))

    def get_dark_mask(self, frame, drawing_frame=None):
        # Undistort the frame if needed
        valid_mask = None
        if self.undistort:
            frame, valid_mask = undistort_fisheye(frame, zoom=False)
        
        # Find dark areas using adaptive thresholding
        mask = adaptive_thres(frame)

        # Crop out the upper part of the mask to keep only the lower part of the frame.
        mask[:int(frame.shape[:2][0] * (1-self.v_fov)), :] = 0

        # Crop out invalid areas due to undistortion
        if valid_mask is not None:
            mask = cv2.bitwise_and(mask, mask, mask=valid_mask)

        # Erode and dilate to remove noise and fill gaps.
        mask = cv2.erode(mask, kernel=self.morph_kernel, iterations=self.erode_iterations)
        mask = cv2.dilate(mask, kernel=self.morph_kernel, iterations=self.dilate_iterations)
        
        # Overwrite the drawing frame with the mask for debugging.
        if drawing_frame is not None:
            drawing_frame[:] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        return mask

    def find_dots(self, frame, drawing_frame=None):
        mask = self.get_dark_mask(frame)
        
        if drawing_frame is not None:
            drawing_frame[:] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Find quadrilateral contours in the mask with sufficient area
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > self.min_area]
        dots = []
        
        # Maximum allowed aspect ratio (long side divided by short side)
        for cnt in contours:
            # Approximate the contour to a polygon
            epsilon = self.ep * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # Check if the approximated contour has 4 points (quadrilateral) and is convex.
            if len(approx) == 4 and cv2.isContourConvex(approx):
                x, y, w, h = cv2.boundingRect(approx)
                # Filter out quadrilaterals that are too elongated
                if min(w, h) == 0 or max(w, h) / min(w, h) > self.max_aspect_ratio:
                    continue
                pt1, pt2, center, angle, length = get_contour_line_info(approx, fix_vert=False)
                dots.append(center)
                line = ((pt1[0], pt1[1]), (pt2[0], pt2[1]))
                # Optionally, draw the detected dot on the image
                cv2.circle(drawing_frame, center, 5, (0, 0, 255), -1)
                cv2.polylines(drawing_frame, [approx], True, (0, 255, 0), 2)
            else:
                cv2.polylines(drawing_frame, [approx], True, (255, 0, 0), 2)
        return dots

    def find_dotted_lines(self, frame, drawing_frame=None):
        dots = self.find_dots(frame, drawing_frame=drawing_frame)
        groups = group_dotted_lines_simple(dots, min_inliers=self.min_points)
        dotted_lines = [(group[0], group[-1]) for group in groups if len(group) >= 2]
        line_centers = [((line[0][0] + line[1][0]) // 2, (line[0][1] + line[1][1]) // 2) for line in dotted_lines]
        angles = [((math.degrees(math.atan2(l[1][1] - l[0][1], l[1][0] - l[0][0])) + 90) % 180) - 90 for l in dotted_lines]

        # Optionally draw the lines and their centers on the image
        for i, line in enumerate(dotted_lines):
            cv2.line(drawing_frame, line[0], line[1], (255, 0, 0), 2)
            cv2.circle(drawing_frame, line_centers[i], 8, (0, 255, 0), -1)
        return dotted_lines, line_centers, angles
    
    def find_intersection(self, frame, drawing_frame=None):
        dotted_lines, centers, angles = self.find_dotted_lines(frame)
        dotted_lines = zip(dotted_lines, centers, angles)

        # Find the line with the longest distance between endpoints
        def line_length(line):
            (pt1, pt2), center, angle = line
            return math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
        dotted_lines = sorted(dotted_lines, key=line_length, reverse=True)
        best_line = next(iter(dotted_lines), None)

        if drawing_frame is not None and best_line is not None:
            line, center, angle = best_line
            cv2.drawMarker(drawing_frame, center, (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

        return best_line

    def stop_at_intersection(self, frame, drawing_frame=None, intersection=None):
        throttle, yaw = 0, 0

        # Get the intersection
        intersection = self.find_intersection(frame, drawing_frame=drawing_frame) if intersection is None else intersection

        # Align the robot with the intersection
        if intersection is not None:
            line, center, angle = intersection
            error = math.radians(angle)
            yaw = self.w_pid(error)
            alpha = 1 - (abs(error) / self.yaw_threshold) if abs(error) < self.yaw_threshold else 0
            norm_y = center[1] / frame.shape[0]
            measured_distance = (1 - alpha) * self.v_pid.setpoint + alpha * norm_y
            throttle = self.v_pid(measured_distance)

        return throttle, yaw

class TrackNavigator:
    def __init__(self,
        line_follower=None,
        intersection_detector=None,
        controller=None,
        decision_func=None,
        decision_action=None,
        backward = None,
        turn_left = None,
        turn_right = None,
        forward = None,
    ):
        self.lf = line_follower or LineFollower()
        self.id = intersection_detector or IntersectionDetector()
        self.controller = controller or OpenLoopController()

        # Parameters
        self.decision_func = decision_func      # Function to decide the next action
        self.decision_action = decision_action  # Action to perform after decision

        # Static variables
        self.tmr = 0  # Timer for decision making

        # Actions
        backward = backward or [ (0, math.radians(180)) ]
        turn_left = turn_left or [
            (0.35, 0, 2.0),
            (0, math.radians(90), 5.0),
            (0.15, 0, 2.0),
        ]
        turn_right = turn_right or [
            (0.35, 0, 2.0),
            (0, -math.radians(90), 5.0),
            (0.15, 0, 2.0),
        ]
        forward = forward or [ (0.5, 0) ]
        self.actions = [backward, turn_left, turn_right, forward] # 0 = backward, 1 = left, 2 = right, 3 = forward
    
    def navigate(self, frame, drawing_frame=None):
        decision_func = self.decision_func or (lambda frame: 2) # Default decision function
        
        # If an action is in progress execute it.
        if self.controller.running():
            thr, yaw = self.controller.execute()
            return thr, yaw

        # Attempt to find intersection
        intersection = self.id.find_intersection(frame, drawing_frame=drawing_frame)

        # If intersection is found, stop at it
        if intersection is not None:
            thr, yaw = self.id.stop_at_intersection(frame, drawing_frame=drawing_frame, intersection=intersection)

            # Wait for the robot to stabilize
            if not (abs(thr) < 0.02 and abs(thr) < 0.02):
                self.tmr = time.time() # Reset the timer (keep waiting)
            
            # If the robot has been stable for n seconds poll the decision function
            if time.time() - self.tmr > 2:
                print("Polling decision function...")
                action_index = decision_func(frame)


                if action_index != -1:
                    action_labels = ["backward", "turn_left", "turn_right", "forward"]
                    print(f"Decision made: {action_labels[action_index]}")
                    if self.decision_action:
                        self.decision_action(action_index)
                    # Start the action sequence
                    self.controller.start(steps=self.actions[action_index], loop=False)

        else:
            # If no intersection is found, follow the line
            thr, yaw = self.lf.follow_line(frame, drawing_frame=drawing_frame)
        return thr, yaw

class OpenLoopController:
    def __init__(self,
        linear_factor=1.0,
        angular_factor=1.0,
        loop=False,
        default_duration=5.0,
        steps=None,
    ):
        self.linear_factor = linear_factor
        self.angular_factor = angular_factor

        self.steps = steps or []

        self.start_time = None  # Start time of the sequence
        self.loop = loop        # Whether to loop the sequence
        self.default_duration = default_duration
    
    def running(self):
        return self.start_time is not None

    def elapsed(self):
        if self.start_time is None:
            return float('inf')  # If not running, return infinite elapsed time
        return time.time() - self.start_time

    def unpack_step(self, step):
        if len(step) == 2:
            # If the step is a tuple of (x, theta)
            x, theta = step
            t = self.default_duration
        elif len(step) == 3:
            x, theta, t = step
            t = t or self.default_duration
        return x, theta, t

    def get_current_index(self):
        if not self.running():
            return -1
        elapsed = self.elapsed()
        total = self.total_time()
        if self.loop and total > 0:
            elapsed = elapsed % total
        unpacked_steps = [self.unpack_step(step) for step in self.steps]
        for i, (x, theta, t) in enumerate(unpacked_steps):
            if elapsed < t:
                return i
            elapsed -= t
        return len(self.steps) - 1
    
    def total_time(self):
        unpacked_steps = [self.unpack_step(step) for step in self.steps]
        return sum(t for _, _, t in unpacked_steps)

    def stop(self):
        self.start_time = None

    def start(self, steps=None, loop=None):
        self.loop = loop if loop is not None else self.loop
        self.steps = steps or self.steps
        self.start_time = time.time()

    def execute(self):
        if not self.running():
            return
        if self.elapsed() >= self.total_time() and not self.loop:
            self.start_time = None
            return 0.0, 0.0  # Sequence finished

        idx = self.get_current_index()
        if idx != -1:
            x, theta, t = self.unpack_step(self.steps[idx])
            # Calculate throttle and yaw
            throttle = x * self.linear_factor / t
            yaw = theta * self.angular_factor / t
            return throttle, yaw

# SHARED VISION PIPELINE
def adaptive_color_thresh(frame, drawing_frame=None,
                          target_hue=0,
                          hue_tol=10,
                          sat_thresh=60,
                          block_size=255,
                          c_value=5):
    """
    Adaptive + absolute hue threshold.
      • target_hue in [0–179]
      • hue_tol = max absolute hue difference (band half-width)
      • sat_thresh = min saturation
      • block_size, c_value = adaptiveThreshold-style params, but applied manually
    
    Returns mask (uint8) and overwrites drawing_frame with the BGR mask.
    """
    # 1) Prepare drawing_frame
    drawing_frame = drawing_frame if drawing_frame is not None else frame.copy()

    # 2) HSV split
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 3) Circular hue diff
    diff = cv2.absdiff(h, np.full_like(h, target_hue))
    diff = cv2.min(diff, 180 - diff).astype(np.float32)

    # 4) Local mean of diff
    #    (box filter approximates a sliding-window average)
    mean_diff = cv2.blur(diff, (block_size, block_size))

    # 5) Build mask: both adaptive AND absolute conditions
    #    - adaptive: diff ≤ mean_diff − c_value
    #    - absolute: diff ≤ hue_tol
    #    - saturation: s ≥ sat_thresh
    mask = np.zeros_like(h, dtype=np.uint8)
    cond = (diff <= (mean_diff - c_value)) & (diff <= hue_tol) & (s >= sat_thresh)
    mask[cond] = 255

    # 6) Overwrite drawing_frame with BGR mask colored by target_hue
    # Create a color version of the mask using the target hue
    color_mask = np.zeros_like(frame)
    # Create an HSV image where H=target_hue, S=255, V=255
    hsv_color = np.zeros_like(frame)
    hsv_color[..., 0] = target_hue
    hsv_color[..., 1] = 255
    hsv_color[..., 2] = 255
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
    # Apply the mask: where mask==255, use the color; else, keep black
    color_mask[mask == 255] = bgr_color[mask == 255]
    drawing_frame[:] = color_mask

    return mask

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

def undistort_fisheye(frame, drawing_frame=None, zoom=True):
    """
    Undistort a fisheye image.
    If zoom=False, also returns a mask (uint8) where
    valid pixels==1 and border fill==0.
    """
    h, w = frame.shape[:2]
    
    # choose balance & borderMode
    if zoom:
        balance, borderMode = 0.0, cv2.BORDER_CONSTANT
    else:
        balance, borderMode = 1.0, cv2.BORDER_REPLICATE

    # compute new camera matrix & remap
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=balance
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
    )
    undistorted = cv2.remap(
        frame, map1, map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=borderMode
    )

    # Overwrite the drawing frame with the undistorted image
    if drawing_frame is not None:
        drawing_frame[:] = undistorted

    # if zoom=False, build & return a valid-pixel mask
    if not zoom:
        # start with a plane of ones
        ones = np.ones((h, w), dtype=np.uint8)
        # remap with constant=0 → zeros at any out-of-bounds
        mask = cv2.remap(
            ones, map1, map2,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        return undistorted, mask

    # otherwise just return the image
    return undistorted, None

# HELPERS
def get_contour_line_info(c, fix_vert=True):
    # Fit a line to the contour
    vx, vy, cx, cy = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    
    # Project contour points onto the line's direction vector.
    projections = [((int(pt[0][0]) - int(cx)) * vx + (int(pt[0][1]) - int(cy)) * vy) for pt in c]
    min_proj = min(projections)
    max_proj = max(projections)
    
    # Compute endpoints from the extreme projection values.
    pt1 = (int(round(cx + vx * min_proj)), int(round(cy + vy * min_proj)))
    pt2 = (int(round(cx + vx * max_proj)), int(round(cy + vy * max_proj)))
    
    # Calculate the line angle in degrees.
    angle = math.degrees(math.atan2(vy, vx))
    if fix_vert:
        angle = angle - 90 * np.sign(angle)
    
    # Calculate the line length given pt1 and pt2.
    length = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
    
    # Ensure cx, cy are ints as well
    cx_int = int(round(cx))
    cy_int = int(round(cy))
    center = (cx_int, cy_int)
    
    return pt1, pt2, center, angle, length

def group_dotted_lines_simple(points,
                              min_inliers=4,
                              dist_threshold=3.0,
                              distance_ratio=2.5):
    """
    points: list of (x, y) int tuples
    Returns: list of lists of (x, y) int tuples,
             each a contiguous dotted‐line segment
    """
    pts_arr = np.array(points, dtype=float)
    # 1) Collect every inlier‐set for each line defined by a point pair
    candidate_sets = {}
    for i, j in combinations(range(len(pts_arr)), 2):
        p1, p2 = pts_arr[i], pts_arr[j]
        v = p2 - p1
        norm = np.linalg.norm(v)
        if norm < 1e-6:
            continue
        u = v / norm
        # normal to line
        n = np.array([-u[1], u[0]])
        # distance of every point to this line
        dists = np.abs((pts_arr - p1) @ n)
        inliers = np.where(dists <= dist_threshold)[0]
        if len(inliers) >= min_inliers:
            # sort the inliers by their original tuple to make a unique key
            key = tuple(sorted((points[k] for k in inliers)))
            candidate_sets[key] = inliers

    # 2) For each unique inlier‐set, split it into contiguous segments by gap
    lines = []
    for key, idxs in candidate_sets.items():
        arr = np.array(key, dtype=float)
        # principal direction via PCA (largest eigenvector of covariance)
        cov = np.cov(arr, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        dir_vec = eigvecs[:, np.argmax(eigvals)]

        # project & sort
        proj = arr @ dir_vec
        order = np.argsort(proj)
        sorted_pts = arr[order]

        # compute gaps and find minimum gap
        deltas = np.linalg.norm(np.diff(sorted_pts, axis=0), axis=1)
        if len(deltas) == 0:
            continue
        d_min = deltas.min()

        # split on any jump > distance_ratio * d_min
        segments = []
        current = [sorted_pts[0]]
        for pt, gap in zip(sorted_pts[1:], deltas):
            if gap > distance_ratio * d_min:
                if len(current) >= min_inliers:
                    segments.append(current)
                current = [pt]
            else:
                current.append(pt)
        if len(current) >= min_inliers:
            segments.append(current)

        # convert back to int tuples
        for seg in segments:
            seg_pts = [(int(x), int(y)) for x, y in seg]
            lines.append(seg_pts)

    return lines

def find_corresponding_point(new_point, old_points, threshold):
    """
    Returns the first old point that is within the absolute pixel distance 'threshold'
    from new_point. If none is found, returns None.
    
    Parameters:
        new_point: A sequence (x, y) representing the new point.
        old_points: An iterable of points (each as a sequence (x, y)) to search through.
        threshold: Absolute pixel distance threshold (float or int).
    
    Returns:
        A point from old_points that is within the threshold distance of new_point or
        None if no such point exists.
    """
    # Compute distances from new_point to each old point
    corresponding = sorted(
        ((pt, ((new_point[0]-pt[0])**2 + (new_point[1]-pt[1])**2)**0.5) for pt in old_points),
        key=lambda item: item[1]
    )
    
    for pt, dist in corresponding:
        if dist < threshold:
            return pt
    return None

def assign_tracked_ids(new_objs, tracked_objs, id_gen, get_id, set_id, get_pos, upd_obj, threshold_px=100, persist=False):
    """
    Updates a collection of tracked objects by matching them with a new set of detected objects.
    This function is agnostic to the underlying object structure and works with any type, as long
    as the appropriate accessor, mutator, and updater callables are provided.

    For each object in new_objs, the function:
      - Extracts a representative position using get_pos.
      - Compares this position against each object in tracked_objs (via get_pos) using the 
        find_corresponding_point utility and a threshold (threshold_px) to determine if the 
        object has been seen before.
      - If a match is found:
          • The existing object's ID (obtained via get_id) is assigned to the new object using set_id.
          • The existing object's data is updated with that from the new object using upd_obj.
          • The matched object is then removed from further consideration.
      - If no match is found:
          • A new unique ID is generated (using id_gen), assigned to the new object using set_id,
            and the new object is added to tracked_objs.
    
    Optionally, if persist is False, objects in tracked_objs that are not present in new_objs are 
    considered "lost" and removed from tracked_objs; regardless, they are returned in a separate lost_objs list.

    Parameters:
      new_objs (list): Collection of newly detected objects.
      tracked_objs (list): Collection of objects currently being tracked.
      id_gen (callable): Function to generate new unique IDs.
      get_id (callable): Function to extract an object's unique identifier.
      set_id (callable): Function to set an object's unique identifier.
      get_pos (callable): Function that returns a representative position (e.g., a point) for matching.
      upd_obj (callable): Function that updates an existing object's data with that of a new detection.
      threshold_px (int, optional): Maximum pixel distance to consider two objects as matching (default is 100).
      persist (bool, optional): If False, objects not matched in new_objs will be removed from tracked_objs (default is False).

    Returns:
      tuple: (tracked_objs, lost_objs)
             tracked_objs: The updated collection of tracked objects.
             lost_objs: Objects that were not matched in new_objs (i.e. "lost" objects).
    """
    # Iterate over the new objects and assign existing or new IDs
    prev_objs = tracked_objs.copy()
    for new_obj in new_objs:
        # Attempt to find a corresponding point in the previous list
        new_pos = get_pos(new_obj)
        old_poss = [get_pos(o) for o in prev_objs]
        corresponding_point = find_corresponding_point(new_pos, old_poss, threshold=threshold_px)

        # It's a previously seen object
        if corresponding_point is not None:
            # Use the corresponding point's index to find the corresponding object
            old_idx = old_poss.index(corresponding_point)
            corresponding_obj = prev_objs[old_idx]

            # Assign the ID from the old object to the new object
            set_id(new_obj, get_id(corresponding_obj))

            # Update the old object with the new object's data
            upd_obj(corresponding_obj, new_obj)

            # Remove the matched object from the previous list so it won't be matched again
            del prev_objs[old_idx]
        else:  # It's a new object
            # Assign a new ID to the new object
            set_id(new_obj, id_gen())
            # Add the new object to the list of tracked objects
            tracked_objs.append(new_obj)
    
    # Optionally, remove objects that have left the field of view
    lost_objs = [o for o in tracked_objs if get_id(o) not in (get_id(o) for o in new_objs)]
    if not persist:
        tracked_objs = [o for o in tracked_objs if o not in lost_objs]
    return tracked_objs, lost_objs

def clear_lost_objects(tracked_objs, lost_objs, lost_timeout, is_lost, get_lost_time, set_lost_time, refind, get_id=None): # get_lost_time, set_lost_time, refind_obj
    # Remove the lost key from objects that are no longer lost
    refound_objs = [o for o in tracked_objs if o not in lost_objs and is_lost(o)]
    for obj in refound_objs:
        if get_id is not None:
            print(f"Object {get_id(obj)} refound after being lost for {time.time() - get_lost_time(obj)} seconds")
        refind(obj)

    # For each lost object, set a lost_time key if it doesn't have one
    newly_lost_objs = [o for o in lost_objs if not is_lost(o)]
    for obj in newly_lost_objs:
        set_lost_time(obj, time.time())

    # For each lost object, if it has a lost_time key, check if it's been lost for more than n seconds
    for obj in lost_objs:
        if is_lost(obj):
            if time.time() - get_lost_time(obj) > lost_timeout:
                tracked_objs.remove(obj)
                if get_id is not None:
                    print(f"Object {get_id(obj)} removed after being lost for {lost_timeout} seconds")

# GLOBAL CAMERA PARAMETERS
K = np.array([
    [394.32766428,   0.,         343.71433623],
    [  0.,         524.94987967, 274.24900983],
    [  0.,           0.,           1.        ]
], dtype=np.float64)
D = np.array([-0.02983132, -0.02312677, 0.03447185, -0.02105932], dtype=np.float64)
