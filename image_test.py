import cv2
import numpy as np


# ——— Camera specs ———
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
    D = np.array([-0.14, -0.1, -0.1, -0.1], dtype=np.float64)

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

def adaptive_threshold(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 5)
    
    return thresh

if __name__ == "__main__":
    # Display
    image_path = "./resources/screenshots/irl_intersection.png"
    img = cv2.imread(image_path)
    img = undistort_fisheye(img)
    img = adaptive_threshold(img)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
