import web
import time
import cv2

gst_pipeline = (
    'nvarguscamerasrc ! '
    'video/x-raw(memory:NVMM),width=640,height=480,framerate=15/1 ! '
    'nvvidconv flip-method=0 ! '
    'video/x-raw,format=BGRx ! '
    'videoconvert ! '
    'video/x-raw,format=BGR ! '
    # appsink caps to drop late frames and keep latency low
    'appsink drop=true max-buffers=1'
)

# cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
cap = cv2.VideoCapture(0)  # Use the default camera (0) for testing

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame")
#         break
#     cv2.imshow('Camera', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

def video_source():
    ret, frame = cap.read()
    return frame if ret else None

if __name__ == '__main__':
    web.port = 5001
    web.video_endpoints["car_cam"] = video_source
    web.start_webserver(threaded=True)
    while True:
        print("Main thread is running...")
        time.sleep(1)
