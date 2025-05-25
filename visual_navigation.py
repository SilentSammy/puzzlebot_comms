import threading
from simple_pid import PID
import numpy as np
import math
import cv2
import time
from collections import deque
from itertools import combinations

# GLOBAL CAMERA PARAMETERS
K = np.array([
    [394.32766428,   0.,         343.71433623],
    [  0.,         524.94987967, 274.24900983],
    [  0.,           0.,           1.        ]
], dtype=np.float64)
D = np.array([-0.02983132, -0.02312677, 0.03447185, -0.02105932], dtype=np.float64)

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
    
    return pt1, pt2, angle, cx_int, cy_int, length

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

# SHARED VISION STAGES
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

# LINE FOLLOWING EXCLUSIVE VISION STAGES
def get_line_mask(frame, drawing_frame=None,
    v_fov = 0.4,  # Bottom field of view (0.4 = 40% of the frame height)
    morph_kernel = np.ones((5, 5), np.uint8),  # Kernel for morphological operations
    erode_iterations = 4,  # Number of iterations for erosion
    dilate_iterations = 6,  # Number of iterations for dilation
):
    # Get mask
    mask = adaptive_thres(frame)

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

# STOPLIGHT EXCLUSIVE VISION STAGES
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

def ellipse_mask(mask, drawing_frame=None,
    morph_kernel = np.ones((3, 3), np.uint8),       # Kernel for morphological operations
    erode_iterations = 3,                           # Number of iterations for erosion
    dilate_iterations = 2,                          # Number of iterations for dilation
    min_fill_ratio = 0.85,                          # Minimum fill ratio for ellipses
    max_outside_ratio=0.15,                          # Maximum allowed outside-area ratio
    max_tilt = 0.5,                                 # Maximum tilt ratio for ellipses
    max_norm_major = 0.8,                           # Maximum normalized major axis for ellipses
    min_norm_major = 0.02,                          # Minimum normalized major axis for ellipses
    ):

    # Erode and dilate to remove noise and fill gaps.
    mask = cv2.erode(mask, kernel=morph_kernel, iterations=erode_iterations)
    mask = cv2.dilate(mask, kernel=morph_kernel, iterations=dilate_iterations)

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
        if min(axis1, axis2) < max_tilt * max(axis1, axis2):
            if drawing_frame is not None:
                cv2.drawContours(drawing_frame, [cnt], -1, (255, 0, 255), 1)  # Draw contour in magenta
            continue

        # 4. Filter out ellipses too small or too large.
        norm_major = max(axis1, axis2) / mask.shape[1]
        if norm_major < min_norm_major or norm_major > max_norm_major:
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
        if fill_ratio < min_fill_ratio:
            if drawing_frame is not None:
                cv2.ellipse(drawing_frame, ellipse, (0, 255, 255), 2)  # Draw ellipse in yellow
            continue

        # 6. Outside-area ratio filter
        outside_pixels = ellipse_pixels - filled   # Pixels inside ellipse but not in the mask
        outside_ratio = outside_pixels / ellipse_pixels
        if outside_ratio > max_outside_ratio:
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

def stoplight_mask(frame, drawing_frame=None):
    red_mask = adaptive_color_thresh( frame )
    yellow_mask = adaptive_color_thresh( frame, target_hue=30, hue_tol=12, sat_thresh=80 )
    green_mask = adaptive_color_thresh( frame, target_hue=65, hue_tol=20, sat_thresh=30 )

    red_mask, red_ellipses = ellipse_mask( red_mask )
    green_mask, green_ellipses = ellipse_mask( green_mask )
    yellow_mask, yellow_ellipses = ellipse_mask( yellow_mask )

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

def identify_stoplight(frame, drawing_frame=None,
    edge_thres_x=0.3, # Fraction of the frame width
    edge_thres_y=0.0, # Fraction of the frame height
    chain_length=4,   # Consecutive frames to consider a color as seen
    max_chain_gap=1,  # Maximum gap in frames to consider a color as seen
):
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
        near_x = abs(x_norm) + normalized_radius_x >= (1 - edge_thres_x)
        near_y = abs(y_norm) + normalized_radius_y >= (1 - edge_thres_y)
        
        return near_x or near_y
    
    def ellipse_key(color_coded_ellipse):
        color_code, ((center_x, center_y), (axis1, axis2), angle) = color_coded_ellipse
        return axis1

    def check_latest_colors():
        if len(identify_stoplight.latest_colors) < chain_length:
            return None

        latest = list(identify_stoplight.latest_colors)
        count_valid = 0     # count of valid (non-None) frames of the candidate color
        gap_count = 0       # count of consecutive None frames
        candidate = None

        # Process from the newest to older frames
        for color in reversed(latest):
            if color is None:
                gap_count += 1
                # If gaps exceed allowed, break out.
                if gap_count > max_chain_gap:
                    break
            else:
                if candidate is None:
                    candidate = color  # first valid color establishes our candidate
                    count_valid = 1
                elif color == candidate:
                    count_valid += 1
                else:
                    # a different valid color disrupts the chain
                    break

            # Check if we've seen at least the required total frames
            if (count_valid + gap_count) >= chain_length:
                # Only accept candidate if valid frames are sufficient
                if count_valid >= (chain_length - max_chain_gap):
                    return candidate

        return None

    # Static variables
    identify_stoplight.latest_colors = deque(maxlen=30) if not hasattr(identify_stoplight, "latest_colors") else identify_stoplight.latest_colors

    # Process the frame
    mask, red_ellipses, yellow_ellipses, green_ellipses = stoplight_mask(frame)
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
    if len(red_ellipses_fltrd) != len(red_ellipses):
        # print("Some red ellipses are near the edge of the frame. Ignoring them.")
        pass

    ellipses = [(0, e) for e in red_ellipses_fltrd]
    ellipses.extend([(1, e) for e in yellow_ellipses])
    ellipses.extend([(2, e) for e in green_ellipses])

    # Get the largest if any
    ellipses = sorted(ellipses, key=ellipse_key, reverse=True)
    largest = next(iter(ellipses), None)

    seen_color = largest[0] if largest is not None else None
    identify_stoplight.latest_colors.append(seen_color)
    confirmed_color = check_latest_colors()
    # print(f"Seen: {seen_color}, Confirmed: {confirmed_color}")

    return confirmed_color

# INTERSECTION VISION STAGES
def get_dark_mask(frame, drawing_frame=None,
    undistort=True,
    v_fov = 0.6,  # Bottom field of view (0.6 = 60% of the frame height)
    morph_kernel = np.ones((3, 3), np.uint8),  # Kernel for morphological operations
    erode_iterations = 3,  # Number of iterations for erosion
    dilate_iterations = 2,  # Number of iterations for dilation
):
    # Undistort the frame if needed
    valid_mask = None
    if undistort:
        frame, valid_mask = undistort_fisheye(frame, zoom=False)
    
    # Find dark areas using adaptive thresholding
    mask = adaptive_thres(frame)

    # Crop out the upper part of the mask to keep only the lower part of the frame.
    mask[:int(frame.shape[:2][0] * (1-v_fov)), :] = 0

    # Crop out invalid areas due to undistortion
    if valid_mask is not None:
        mask = cv2.bitwise_and(mask, mask, mask=valid_mask)

    # Erode and dilate to remove noise and fill gaps.
    mask = cv2.erode(mask, kernel=morph_kernel, iterations=erode_iterations)
    mask = cv2.dilate(mask, kernel=morph_kernel, iterations=dilate_iterations)
    
    # Overwrite the drawing frame with the mask for debugging.
    if drawing_frame is not None:
        drawing_frame[:] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    return mask

def find_dots(frame, drawing_frame=None,
    max_aspect_ratio = 10.0,
    min_area = 20,
    ep = 0.035, # Approximation factor for contour approximation
    undistort=True,
):
    mask = get_dark_mask(frame, undistort=undistort)
    
    if drawing_frame is not None:
        drawing_frame[:] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Find quadrilateral contours in the mask with sufficient area
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > min_area]
    dots = []
    
    # Maximum allowed aspect ratio (long side divided by short side)
    for cnt in contours:
        # Approximate the contour to a polygon
        epsilon = ep * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Check if the approximated contour has 4 points (quadrilateral) and is convex.
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)
            # Filter out quadrilaterals that are too elongated
            if min(w, h) == 0 or max(w, h) / min(w, h) > max_aspect_ratio:
                continue
            pt1, pt2, angle, cx, cy, length = get_contour_line_info(approx, fix_vert=False)
            center = (cx, cy)
            dots.append(center)
            center = (int(cx), int(cy))
            line = ((pt1[0], pt1[1]), (pt2[0], pt2[1]))
            # Optionally, draw the detected dot on the image
            cv2.circle(drawing_frame, center, 5, (0, 0, 255), -1)
            cv2.polylines(drawing_frame, [approx], True, (0, 255, 0), 2)
        else:
            cv2.polylines(drawing_frame, [approx], True, (255, 0, 0), 2)

    return dots

def find_dotted_lines(frame, drawing_frame=None,
    min_points=5,  # Minimum number of points to consider a line
    undistort=True,
):
    dots = find_dots(frame, drawing_frame=drawing_frame, undistort=undistort)
    groups = group_dotted_lines_simple(dots, min_inliers=min_points)
    dotted_lines = [(group[0], group[-1]) for group in groups if len(group) >= 2]
    line_centers = [((line[0][0] + line[1][0]) // 2, (line[0][1] + line[1][1]) // 2) for line in dotted_lines]
    angles = [((math.degrees(math.atan2(l[1][1] - l[0][1], l[1][0] - l[0][0])) + 90) % 180) - 90 for l in dotted_lines]

    
    # Optionally draw the lines and their centers on the image
    for i, line in enumerate(dotted_lines):
        cv2.line(drawing_frame, line[0], line[1], (255, 0, 0), 2)
        cv2.circle(drawing_frame, line_centers[i], 8, (0, 255, 0), -1)
    return dotted_lines, line_centers, angles

def find_intersection(frame, drawing_frame=None,
    undistort=True,
):
    dotted_lines, centers, angles = find_dotted_lines(frame, drawing_frame=None, undistort=undistort)
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

# FLAG VISION STAGES
def get_flag_distance(frame, drawing_frame=None,
    pattern_size=(4, 3),
    square_size=0.025,
):
    """
    Estima Z (m) usando solo la altura en píxeles del patrón.
    Devuelve (Z, h_pix).
    """
    ret, corners = cv2.findChessboardCorners(frame, pattern_size, None)
    if not ret:
        return None
    f_y = K[1,1]
    ys = corners[:,:,1].flatten()
    h_pix = ys.max() - ys.min()
    # Altura real entre la primer y última fila de esquinas internas
    H_real = square_size * (pattern_size[1] - 1)
    dist = (f_y * H_real) / h_pix
    if drawing_frame is not None:
        # Draw text above the topmost chessboard corner
        cv2.drawChessboardCorners(drawing_frame, pattern_size, corners, ret)
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

def get_flag_distance_nb(frame, drawing_frame=None,
    pattern_size=(4, 3),
    square_size=0.025,
):
    # Static worker thread
    if not hasattr(get_flag_distance_nb, "worker"):
        get_flag_distance_nb.worker = None
        get_flag_distance_nb.lock = threading.Lock()
        get_flag_distance_nb.last_result = None
        get_flag_distance_nb.last_drawing_frame = None
    
    with get_flag_distance_nb.lock:
        result = get_flag_distance_nb.last_result
        annotated_frame = get_flag_distance_nb.last_drawing_frame
        
    
    # If processing is not ongoing, process in background
    if get_flag_distance_nb.worker is None or not get_flag_distance_nb.worker.is_alive():
        def worker_func(frame_copy, drawing_frame, pattern_size, square_size):
            dist = get_flag_distance( frame_copy, drawing_frame=drawing_frame, pattern_size=pattern_size, square_size=square_size )
            with get_flag_distance_nb.lock:
                get_flag_distance_nb.last_result = dist
                get_flag_distance_nb.last_drawing_frame = drawing_frame
    
        # Start the worker thread
        t = threading.Thread(
            target=worker_func,
            args=(frame.copy(), np.zeros_like(frame), pattern_size, square_size),
        )
        t.daemon = True
        t.start()
        get_flag_distance_nb.worker = t
    
    if drawing_frame is not None and annotated_frame is not None:
        # Overwrite the drawing_frame pixels with annotated_frame pixels wherever the mask is True.
        non_black_mask = np.any(annotated_frame != 0, axis=2)
        drawing_frame[non_black_mask] = annotated_frame[non_black_mask]

    return result

# END NAVIGATION ALGORITHMS (THESE RETURN THROTTLE AND YAW) (NON-BLOCKING, MUST BE CALLED IN A LOOP)
def sequence(actions=None, when_done=None, speed_factor=1):
    """ Execute a sequence of actions. Each action is a tuple (v, w, t) """

    # Static variables
    sequence.last_time = sequence.last_time if hasattr(sequence, "last_time") else time.time()
    sequence.elapsed_time = sequence.elapsed_time if hasattr(sequence, "elapsed_time") else 0
    
    # Define the sequence of actions or use default
    actions = actions or [ # v, w, t
        (0.15, 0, 2), # Move 30cm forward
        (0, -math.radians(30), 3), # 90° turn
        (0.15, 0, 2), # Move 30cm forward
    ]
    actions = [(v*speed_factor, w*speed_factor, t) for v, w, t in actions]
    total_time = sum([t for _, _, t in actions])

    # Get the elapsed time
    since_last = time.time() - sequence.last_time
    if since_last > 0.5: # If the time since last call is too long, reset the timer.
        sequence.elapsed_time = 0
        sequence.last_time = time.time()
        since_last = 0
    sequence.elapsed_time += since_last * speed_factor
    sequence.last_time = time.time()

    elapsed = sequence.elapsed_time
    if elapsed > total_time: # done
        if when_done:
            when_done()
    elapsed =  elapsed % total_time

    action = None
    ac_time = 0
    for i, act in enumerate(actions):
        ac_time += act[2]
        if elapsed < ac_time:
            action = act
            break
    
    throttle, yaw = 0, 0
    throttle, yaw, _ = action
    return throttle, yaw

def follow_line(frame, drawing_frame=None,
    Kp=0.6, Ki=0, Kd=0.1,       # PID parameters
    max_yaw=math.radians(60),   # Maximum yaw in radians
    max_thr=0.2,                # Maximum throttle
    align_thres = 0.3           # Throttle will be max_thr when aligned, 0 at the threshold, and negative below the threshold.
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
    return 0.0, 0.0

def follow_line_w_intersection(frame, drawing_frame=None,
    undistort=False,
):
    # Attempt to find intersection
    intersection = find_intersection(frame, drawing_frame=drawing_frame, undistort=undistort)

    # If intersection is found, stop at it
    if intersection is not None:
        throttle, yaw = stop_at_intersection(frame, drawing_frame=drawing_frame, intersection=intersection)
    else:
        # If no intersection is found, follow the line
        throttle, yaw = follow_line(frame, drawing_frame=drawing_frame)
    return throttle, yaw

def follow_line_w_signs(frame, drawing_frame=None, end_action=None):
    # Static variables
    follow_line_w_signs.tmr = follow_line_w_signs.tmr if hasattr(follow_line_w_signs, "tmr") else 0
    follow_line_w_signs.action_index = follow_line_w_signs.action_index if hasattr(follow_line_w_signs, "action_index") else -1
    follow_line_w_signs.stoplight = follow_line_w_signs.stoplight if hasattr(follow_line_w_signs, "stoplight") else 2
    follow_line_w_signs.end_reached = follow_line_w_signs.end_reached if hasattr(follow_line_w_signs, "end_reached") else False

    if not follow_line_w_signs.end_reached:
        # Determine the speed factor based on the stoplight.
        stoplight = identify_stoplight(frame, drawing_frame=drawing_frame)
        if stoplight is not None and stoplight != 1: # If red or green, remember it
            follow_line_w_signs.stoplight = stoplight
        speed_factor = (stoplight or follow_line_w_signs.stoplight) * 0.5
        
        # Check if the flag is close
        dist = get_flag_distance_nb(frame, drawing_frame=drawing_frame)
        flag_is_close = dist is not None and dist < 0.5
        follow_line_w_signs.end_reached = flag_is_close

        if not follow_line_w_signs.end_reached:
            thr, yaw = follow_line(frame, drawing_frame=drawing_frame)
            thr *= speed_factor
            yaw *= speed_factor
        else:
            if end_action:
                end_action()
            thr = 0
            yaw = 0

        return thr, yaw
    else:
        if drawing_frame is not None:
            cv2.putText( drawing_frame, "Flag reached", (drawing_frame.shape[1] // 2 - 80, drawing_frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA )
        return 0, 0

def stop_at_intersection(frame, drawing_frame=None, intersection=None):
    # Static variables
    max_yaw = math.radians(30)
    max_thr = 0.15
    yaw_threshold = 5.0  # The robot will start translation when the target is this many degrees from the center
    if not hasattr(stop_at_intersection, "pids"):
        stop_at_intersection.pids = {
            "w_pid": PID(2.0, 0, 0.1, setpoint=0, output_limits=(-max_yaw, max_yaw)),
            "v_pid": PID(0.5, 0, 0.1, setpoint=0.7, output_limits=(-max_thr, max_thr)),
        }
    w_pid = stop_at_intersection.pids["w_pid"]
    v_pid = stop_at_intersection.pids["v_pid"]
    throttle, yaw = 0, 0

    # Get the intersection
    intersection = find_intersection(frame, drawing_frame=drawing_frame) if intersection is None else intersection

    # Align the robot with the intersection
    if intersection is not None:
        line, center, angle = intersection
        error = math.radians(angle)
        yaw = w_pid(error)
        alpha = 1 - (abs(error) / yaw_threshold) if abs(error) < yaw_threshold else 0
        norm_y = center[1] / frame.shape[0]
        measured_distance = (1 - alpha) * v_pid.setpoint + alpha * norm_y
        throttle = v_pid(measured_distance)

    return throttle, yaw

def reset():
    follow_line.__dict__.clear()
    identify_stoplight.__dict__.clear()
    follow_line_w_signs.__dict__.clear()
    get_flag_distance_nb.__dict__.clear()
