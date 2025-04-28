import numpy as np
import math
import cv2

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

# SQUARES
def find_quadrilaterals(frame, lower_hsv, upper_hsv):
    """
    Finds quadrilaterals in the given frame based on the specified HSV color range.

    Parameters:
        frame: The input image frame.
        lower_hsv: Lower bound of the HSV color range.
        upper_hsv: Upper bound of the HSV color range.

    Returns:
        A list of quadrilaterals, where each quadrilateral is represented as a list of 4 points.
    """

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    quadrilaterals = []
    for cnt in contours:
        # 1. Filter out small contours by area
        if cv2.contourArea(cnt) < 100:
            continue

        # 2. Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 3. Check if the polygon has 4 vertices and is convex
        if len(approx) == 4 and cv2.isContourConvex(approx):
            approx = order_points(approx)
            quadrilaterals.append(approx)

    return quadrilaterals

def order_points(pts):
    """
    Orders a set of 4 points in the following order:
    top-left, top-right, bottom-right, bottom-left.

    Parameters:
      pts: A numpy array of shape (4,1,2) (or (4,2)) that you want to reorder.
    
    Returns:
      A numpy array with the same shape as the input, with the points reordered.
    """
    # Flatten into (4,2) regardless of input shape.
    pts = pts.reshape(4, 2)
    ordered = np.empty_like(pts)
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    
    # Reshape to match the input shape; if input was (4,1,2) return that.
    return ordered.reshape(4, 1, 2)

def estimate_square_pose(square_corners, frame, camera_matrix, dist_coeffs, square_length):
    """
    Estimates the pose of a square given its detected corners. This function assumes that
    square_corners is of shape (1, 4, 2) (e.g. from find_quadrilaterals) and that the square's
    side length (in meters) is known.
    
    It returns a tuple:
       (yaw, pitch, roll, estimated_distance, cam_pitch, cam_yaw)
    where yaw, pitch, roll are in degrees, estimated_distance is the distance from the camera,
    and cam_pitch and cam_yaw are angular offsets computed from the marker's center.

    This version uses cv2.solvePnP and skips the custom (ref_size) approach.
    """
    
    def order_points2(pts):
        """
        Orders a set of 4 points in the following order:
        top-left, top-right, bottom-right, bottom-left.
        
        Parameters:
        pts: A numpy array of shape (4,1,2) (as typically returned by cv2.findContours).
        
        Returns:
        A numpy array of shape (4,1,2) with the points ordered.
        """
        # Convert the shape from (4,1,2) to (4,2) for easier processing.
        pts = pts.reshape(4, 2)
        ordered = np.zeros((4, 2), dtype=pts.dtype)
        
        # The top-left point has the smallest sum; bottom-right has the largest sum.
        s = pts.sum(axis=1)
        ordered[0] = pts[np.argmin(s)]
        ordered[2] = pts[np.argmax(s)]
        
        # The top-right point has the smallest difference; bottom-left has the largest difference.
        diff = np.diff(pts, axis=1)
        ordered[1] = pts[np.argmin(diff)]
        ordered[3] = pts[np.argmax(diff)]
        
        # Reshape back to (4,1,2) for consistency.
        return ordered.reshape(4, 1, 2)

    def rotationMatrixToEulerAngles(R):
        """
        Converts a rotation matrix to Euler angles (roll, pitch, yaw) using the Tait–Bryan angles convention.
        Returns a numpy array [roll, pitch, yaw] in radians.
        """
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            roll  = math.atan2(R[2, 1], R[2, 2])
            pitch = math.atan2(-R[2, 0], sy)
            yaw   = math.atan2(R[1, 0], R[0, 0])
        else:
            roll  = math.atan2(-R[1, 2], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            yaw   = 0
        return np.array([roll, pitch, yaw])

    # Get image dimensions.
    h, w, _ = frame.shape

    # Define 3D object points for the square.
    # Here we assume the square lies in the z=0 plane and its corners are at:
    # (0, 0, 0); (square_length, 0, 0); (square_length, square_length, 0); (0, square_length, 0)
    # You may adjust this ordering based on your desired coordinate system.
    obj_points = np.array([
        [0, 0, 0],
        [square_length, 0, 0],
        [square_length, square_length, 0],
        [0, square_length, 0]
    ], dtype=np.float32)

    # Ensure points are in the correct format for solvePnP.
    img_points = square_corners
    # img_points = order_points(img_points)  # Ensure points are ordered correctly
    img_points = img_points.reshape(4, 2).astype(np.float32)

    # Solve the PnP problem.
    retval, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
    if not retval:
        return None

    # Convert rvec to rotation matrix and then to Euler angles.
    R, _ = cv2.Rodrigues(rvec)
    euler_rad = rotationMatrixToEulerAngles(R)
    euler_deg = np.degrees(euler_rad)
    # Map the angles (this mapping can be adjusted to your preferred convention)
    yaw = euler_deg[1]
    pitch = euler_deg[0]
    roll = euler_deg[2]

    # Estimated distance is the norm of the translation vector.
    estimated_distance = np.linalg.norm(tvec)

    # Compute the square center in image coordinates.
    marker_center = np.mean(img_points, axis=0)
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    offset_x = marker_center[0] - cx
    offset_y = marker_center[1] - cy

    # Compute FOV angles.
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    fov_x = math.degrees(2 * math.atan(w / (2 * fx)))
    fov_y = math.degrees(2 * math.atan(h / (2 * fy)))
    cam_yaw = (offset_x / (w / 2)) * (fov_x / 2)
    cam_pitch = (offset_y / (h / 2)) * (fov_y / 2)

    return yaw, pitch, roll, estimated_distance, cam_pitch, cam_yaw

def draw_quad(frame, quad, drawOutline=True):
    # Convert quad to the proper type.
    quad_int = quad.astype(np.int32)
    if drawOutline:
        cv2.polylines(frame, [quad_int], isClosed=True, color=(255, 255, 255), thickness=10)
    
    # Compute center of the quadrilateral.
    corners = quad_int.reshape(4, 2)
    center = np.mean(corners, axis=0).astype(np.int32)
    center = np.array(center, dtype=np.float32)
    
    # get distance between first two corners
    line_length = np.linalg.norm(corners[0] - corners[1]) * 0.25
    
    # For each corner, compute direction toward the center
    # then draw a line from the corner toward the center with the computed length.
    for idx, corner in enumerate(corners):
        corner_f = np.array(corner, dtype=np.float32)
        direction = center - corner_f
        norm = np.linalg.norm(direction)
        if norm != 0:
            direction = direction / norm  # normalize
        
        endpoint = (corner_f + direction * line_length).astype(np.int32)
        # The first corner's line will be red; others will be orange.
        color = (0, 0, 255) if idx == 0 else (0, 165, 255)
        cv2.line(frame, tuple(corner), tuple(endpoint), color, 10)

# CIRCLES
def find_ellipses(frame, lower_hsv, upper_hsv):
    # Find contours that match the specified color
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    ellipses = []
    for cnt in contours:
        # 1. Filter out polygons with fewer than 5 points.
        if len(cnt) < 5:
            continue

        # 2. Fit an ellipse to the contour.
        ellipse = cv2.fitEllipse(cnt)
        
        # 3. Check if the ellipse has valid dimensions to avoid raising an (-215:Assertion failed) error.
        (center_x, center_y), (axis1, axis2), angle = ellipse
        if axis1 <= 0 or axis2 <= 0:  # Skip invalid ellipses
            print("Invalid ellipse dimensions:", axis1, axis2)
            continue

        # 4. Filter out small areas
        ellipse_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.ellipse(ellipse_mask, ellipse, 255, -1)
        occluded_area = cv2.countNonZero(cv2.bitwise_and(mask, mask, mask=ellipse_mask))
        if occluded_area < 50:
            continue

        # 5. Filter out non-elliptical shapes
        contour_area = cv2.contourArea(cnt)
        (center_x, center_y), (axis1, axis2), angle = ellipse
        ellipse_area = math.pi * (axis1/2) * (axis2/2)
        area_ratio = contour_area / ellipse_area
        if area_ratio < 0.8 or area_ratio > 1.2:
            continue
        
        # 6. Add the ellipse to the list.
        ellipses.append(ellipse)
    
    return ellipses

def estimate_circle_pose(ellipse, frame, camera_matrix=None, real_diameter=None, ref_size=None, fov_x=None, fov_y=None):
    """
    Estimates the distance, yaw, and pitch of the camera relative to a circle
    (detected as an ellipse). Two approaches are available:
    
      Traditional approach:
         If calibration parameters (i.e. a camera_matrix) and real_diameter are provided,
         this method uses the pinhole camera model:
             estimated_distance = (real_diameter * fx) / (major_axis * corrective_factor)
         (Optionally, you may apply a corrective_factor to the measured ellipse axes.)
      
      Custom (ref_size) approach:
         If calibration parameters are not provided but a ref_size is supplied,
         then default calibration parameters are assumed to compute the normalized major axis.
         The distance is then computed as:
             estimated_distance = ref_size / normalized_major
         (where normalized_major = (corrected_major_axis) / image_width)
    
    In either case, the function also computes angular offsets (cam_pitch and cam_yaw)
    based on the ellipse center, the assumed (or provided) camera matrix, and the frame dimensions.
    
    Parameters:
      ellipse: Tuple ((center_x, center_y), (axis1, axis2), angle)
      frame: The current image frame.
      camera_matrix: (Optional) The camera calibration matrix.
      real_diameter: (Optional) The true diameter of the circle (meters) for the traditional approach.
      ref_size: (Optional) The normalized apparent size of a reference object (as used in marker pose)
                for the custom approach.
      
    Returns:
      Tuple (estimated_distance, cam_pitch, cam_yaw)
    """

    def estimate_circle_orientation(ellipse):
        """
        Given an ellipse (from cv2.fitEllipse), compute the circle's orientation.
        It first computes the tilt based on the axis ratio (minor/major = cos(tilt)).
        Then, using the ellipse's angle, it distributes this tilt into yaw and pitch components.
        Returns a tuple (yaw, pitch) in degrees (with roll assumed to be zero).
        """
        # Unpack ellipse parameters.
        # ellipse = ((center_x, center_y), (axis1, axis2), angle)
        (center_x, center_y), (axis1, axis2), angle = ellipse
        major = max(axis1, axis2)
        minor = min(axis1, axis2)
        # Avoid division by zero.
        if major == 0:
            tilt_deg = 0
        else:
            # Compute the tilt angle from the ratio.
            tilt_rad = math.acos(minor / major)
            tilt_deg = math.degrees(tilt_rad)
        
        # Convert the ellipse's angle to radians.
        angle_rad = math.radians(angle)
        
        # Distribute the tilt: if angle=0, all tilt goes to yaw; if angle=90, all goes to pitch.
        yaw   = tilt_deg * math.cos(angle_rad)
        pitch = tilt_deg * math.sin(angle_rad)
        
        return yaw, pitch, 0

    h, w, _ = frame.shape

    # Unpack ellipse. We'll use the larger axis as the apparent diameter.
    (center_x, center_y), axes, angle = ellipse
    measured_major = max(axes)
    
    # Optionally, apply a corrective factor to compensate for ellipse overestimation.
    # corrective_factor = 0.84
    corrective_factor = 1.0
    corrected_major = measured_major * corrective_factor

    # Determine which approach to use.
    # Traditional required both a camera_matrix and a real_diameter.
    use_traditional = (camera_matrix is not None and real_diameter is not None)
    use_custom = (not use_traditional and ref_size is not None)
    
    if not use_traditional and not use_custom:
        raise ValueError("Insufficient parameters. Provide either (camera_matrix and real_diameter) or ref_size.")

    if not use_traditional:
        # Custom approach: assume default calibration parameters.
        # These defaults should match those used when creating the ref_size.
        fx_default = fy_default = 800.0
        cx_default, cy_default = w / 2.0, h / 2.0
        camera_matrix = np.array([[fx_default, 0, cx_default],
                                  [0, fy_default, cy_default],
                                  [0, 0, 1]], dtype=np.float32)
        # Calculate normalized major axis: ratio of corrected_major to image width.
        normalized_major = corrected_major / float(w)
        estimated_distance = ref_size / normalized_major
    else:
        # Traditional approach using the pinhole camera model.
        fx = camera_matrix[0, 0]
        # Use the corrected major axis for distance estimation.
        estimated_distance = (real_diameter * fx) / corrected_major if corrected_major > 0 else 0.0

    # In either case, use the (assumed or provided) camera_matrix to compute angular offsets.
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    # Use externally provided fov values if they exist; otherwise compute.
    if fov_x is None:
        fov_x = math.degrees(2 * math.atan(w / (2 * fx)))
    if fov_y is None:
        fov_y = math.degrees(2 * math.atan(h / (2 * fy)))
    
    # Compute pixel offsets from the principal point.
    offset_x = center_x - cx
    offset_y = center_y - cy
    
    # Convert these pixel offsets to angular offsets.
    cam_yaw = (offset_x / (w / 2)) * (fov_x / 2)
    cam_pitch = (offset_y / (h / 2)) * (fov_y / 2)
    
    yaw, pitch, roll = estimate_circle_orientation(ellipse)

    # return estimated_distance, cam_pitch, cam_yaw
    return yaw, pitch, roll, estimated_distance, cam_pitch, cam_yaw

def find_corresponding_ellipse(new_ellipse, old_ellipses, threshold=2.5, relative_to_ellipse=True):
    new_center = new_ellipse[0]
    major_axis = max(new_ellipse[1])

    # Determine the base threshold
    base_threshold = major_axis if relative_to_ellipse else 1.0

    # Sort old ellipses by distance to the new ellipse
    sorted_ellipses = sorted(
        ((old_ellipse, math.hypot(new_center[0] - old_ellipse[0][0], new_center[1] - old_ellipse[0][1]))
         for old_ellipse in old_ellipses),
        key=lambda item: item[1]
    )

    # Find the first old ellipse within the threshold
    for old_ellipse, dist in sorted_ellipses:
        if dist < threshold * base_threshold:
            return old_ellipse
    return None

def draw_ellipse(frame, ellipse):
    cv2.ellipse(frame, ellipse, (0, 255, 0), 2)
    (center_x, center_y), (major, minor), angle = ellipse
    center = (int(center_x), int(center_y))
    theta = math.radians(angle)
    major_dx = (major / 2) * math.cos(theta)
    major_dy = (major / 2) * math.sin(theta)
    pt1 = (int(center_x - major_dx), int(center_y - major_dy))
    pt2 = (int(center_x + major_dx), int(center_y + major_dy))
    cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
    theta_minor = theta + math.pi/2  
    minor_dx = (minor / 2) * math.cos(theta_minor)
    minor_dy = (minor / 2) * math.sin(theta_minor)
    pt3 = (int(center_x - minor_dx), int(center_y - minor_dy))
    pt4 = (int(center_x + minor_dx), int(center_y + minor_dy))
    cv2.line(frame, pt3, pt4, (0, 0, 255), 2)

# ARUCO MARKERS
def find_arucos(frame):
    # Detect markers
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    if hasattr(cv2.aruco, 'DetectorParameters_create'):
        parameters = cv2.aruco.DetectorParameters_create()
    else:
        parameters = cv2.aruco.DetectorParameters()
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # print(f"Detected {len(corners)} markers")

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    
    return corners, ids

def estimate_marker_pose(marker_corners, frame, camera_matrix=None, dist_coeffs=None, marker_length=None, ref_size=None, fov_x=None, fov_y=None):
    """
    Estimates pose and extracts Euler angles (yaw, pitch, roll) and distance.
    
    Two modes are available:
      1. Traditional approach:
         If calibration parameters (camera_matrix, dist_coeffs, and marker_length) are provided,
         this method uses cv2.aruco.estimatePoseSingleMarkers.
         
      2. Custom (ref_size) approach:
         If calibration parameters are not provided but a ref_size is, then default calibration 
         parameters are assumed and the distance is computed based on the marker's apparent size:
             estimated_distance = ref_size / normalized_apparent_size
         (with normalized_apparent_size = apparent_size_pixels / image_width).
         
      In addition, the function calculates the camera's pitch and yaw relative to the marker 
      based on its position in the image.
      
      Returns a tuple: (yaw, pitch, roll, distance, cam_pitch, cam_yaw)
    """

    def rotationMatrixToEulerAngles(R):
        """
        Converts a rotation matrix to Euler angles (roll, pitch, yaw) using the 
        Tait–Bryan angles convention. Returns a numpy array [roll, pitch, yaw] in radians.
        """
        sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6
        if not singular:
            roll = math.atan2(R[2, 1], R[2, 2])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = math.atan2(R[1, 0], R[0, 0])
        else:
            roll = math.atan2(-R[1, 2], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = 0
        return np.array([roll, pitch, yaw])

    h, w, _ = frame.shape

    # Determine which approach to use.
    use_traditional = (camera_matrix is not None and dist_coeffs is not None and marker_length is not None)
    use_custom = (not use_traditional and ref_size is not None)
    
    if not use_traditional and not use_custom:
        raise ValueError("Insufficient parameters. Provide calibration parameters (camera_matrix, dist_coeffs, marker_length) or a ref_size.")
    
    # If calibration parameters are missing, assume defaults.
    if not use_traditional:
        marker_length = 0.05  # default marker size in meters
        fx_default = fy_default = 800.0
        cx_default, cy_default = w / 2.0, h / 2.0
        camera_matrix = np.array([[fx_default, 0, cx_default],
                                  [0, fy_default, cy_default],
                                  [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    # Compute pose using the traditional method.
    ret_vals = cv2.aruco.estimatePoseSingleMarkers(marker_corners, marker_length, camera_matrix, dist_coeffs)
    if ret_vals is None or len(ret_vals[0]) == 0:
        return None

    # Extract rotation vector and convert it to Euler angles.
    rvec = ret_vals[0][0][0]  # rotation vector for the first detected marker
    R, _ = cv2.Rodrigues(rvec)
    euler_rad = rotationMatrixToEulerAngles(R)
    euler_deg = np.degrees(euler_rad)
    # Mapping convention:
    yaw = euler_deg[1]
    new_pitch = ((180 - euler_deg[0] + 180) % 360) - 180  # adjusted pitch
    new_pitch = -new_pitch  # Invert pitch for camera perspective
    roll = euler_deg[2]

    # Determine distance.
    if use_custom:
        # Compute apparent marker size.
        corners_array = marker_corners.reshape((4, 2))
        distances = []
        for i in range(4):
            j = (i + 1) % 4
            dx = corners_array[j][0] - corners_array[i][0]
            dy = corners_array[j][1] - corners_array[i][1]
            distances.append(math.hypot(dx, dy))
        apparent_size_pixels = max(distances)
        normalized_apparent_size = apparent_size_pixels / float(w)
        estimated_distance = ref_size / normalized_apparent_size
    else:
        # Use translation vector from the pose.
        tvec = ret_vals[1][0][0]
        estimated_distance = np.linalg.norm(tvec)

    # Calculate the camera's pitch and yaw relative to the marker using both FOV axes.
    # Extract focal lengths from the camera matrix.
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    
    # Use externally provided fov values if they exist; otherwise compute.
    if fov_x is None:
        fov_x = math.degrees(2 * math.atan(w / (2 * fx)))
    if fov_y is None:
        fov_y = math.degrees(2 * math.atan(h / (2 * fy)))
    
    # Compute the marker center.
    marker_center = np.mean(marker_corners.reshape((4, 2)), axis=0)
    # Get principal point from the camera matrix.
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    # Pixel offsets from the principal point.
    offset_x = marker_center[0] - cx
    offset_y = marker_center[1] - cy
    # Convert pixel offsets to angular offsets: half image width corresponds to half fov_x, etc.
    cam_yaw = (offset_x / (w / 2)) * (fov_x / 2)
    cam_pitch = (offset_y / (h / 2)) * (fov_y / 2)

    return (yaw, new_pitch, roll, estimated_distance, cam_pitch, cam_yaw)

# CAMERA
def estimate_camera_pose(yaw, pitch, roll, cam_dist, cam_pitch, cam_yaw):
    def unroll_angle_offsets(yaw_deg, pitch_deg, roll_deg):
        """
        Given angular offsets (e.g. cam_yaw, cam_pitch) in degrees and a roll angle, 
        rotates the (yaw, pitch) vector by -roll so that the roll’s influence is removed.
        
        Returns:
            A tuple (unrolled_yaw, unrolled_pitch) in degrees.
        """
        # Convert to radians.
        yaw_rad = math.radians(yaw_deg)
        pitch_rad = math.radians(pitch_deg)
        roll_rad = math.radians(roll_deg)
        
        # Apply 2D rotation: we rotate the vector (yaw_rad, pitch_rad) by -roll_rad.
        unrolled_yaw_rad = math.cos(-roll_rad)*yaw_rad - math.sin(-roll_rad)*pitch_rad
        unrolled_pitch_rad = math.sin(-roll_rad)*yaw_rad + math.cos(-roll_rad)*pitch_rad
        
        # Convert back to degrees.
        unrolled_yaw = math.degrees(unrolled_yaw_rad)
        unrolled_pitch = math.degrees(unrolled_pitch_rad)
        return unrolled_yaw, unrolled_pitch

    cam_yaw, cam_pitch = unroll_angle_offsets(cam_yaw, cam_pitch, roll)
    yaw_rad = math.radians(cam_yaw-yaw)
    pitch_rad = math.radians(cam_pitch+pitch)
    y = cam_dist * math.cos(pitch_rad) * math.sin(yaw_rad)  # rightward displacement
    x = cam_dist * math.sin(pitch_rad)                      # forward displacement
    z = cam_dist * math.cos(pitch_rad) * math.cos(yaw_rad)  # upward displacement
    alpha = math.radians(yaw)
    beta = math.radians(pitch)
    gamma = 0

    return x, y, z, alpha, beta, gamma

def get_camera_matrix(x_res, y_res, fov_x_deg):
    """Compute the camera matrix based on the resolution and field of view."""
    focal_length = (x_res / 2) / math.tan(math.radians(fov_x_deg) / 2)
    return np.array([[focal_length,      0.0, x_res / 2],
                     [     0.0, focal_length, y_res / 2],
                     [     0.0,      0.0,      1.0]], dtype=np.float32)
