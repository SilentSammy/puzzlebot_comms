import cv2

# — CONFIGURATION —
STREAM_URL   = "http://192.168.137.10:5000/car_cam"
# STREAM_URL   = "http://192.168.137.25:4747/video"
PATTERN_SIZE = (8, 5)   # inner corners per row, column
FLAGS = (cv2.CALIB_CB_ADAPTIVE_THRESH
       | cv2.CALIB_CB_NORMALIZE_IMAGE
       | cv2.CALIB_CB_FAST_CHECK)

# — SET UP VIDEO STREAM —
cap = cv2.VideoCapture(STREAM_URL)

print("Press 'q' or ESC to exit.")
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # 1) Grayscale conversion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2) Chessboard detection
    found, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, flags=FLAGS)

    if found:
        # 3) Sub-pixel refinement
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), term)

        # 4) Draw detected corners
        cv2.drawChessboardCorners(frame, PATTERN_SIZE, corners, found)
        status_text, color = "Detected", (0,255,0)
    else:
        status_text, color = "Not detected", (0,0,255)

    # 5) Overlay status and display
    cv2.putText(frame, f"Chessboard: {status_text}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.imshow("Chessboard Test", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        break

cap.release()
cv2.destroyAllWindows()
