import cv2
import time
import traceback
from local_sign_detector import LocalSignDetector
import requests

BASE_URL = "http://127.0.0.1:5002/"
STREAM_URL = BASE_URL + "video_feed"
DATA_URL = BASE_URL + "frame_data"
DISPLAY = False

sg_det = LocalSignDetector("gtsrb_cnn_98.h5")

def process_frame(frame):
    sign = sg_det.get_best_sign_nb(frame, drawing_frame=frame)
    if sign is not None:
        # Assemble JSON object
        sign_json = {
            "id": sign.id,
            "label": sign.label,
            "detection": list(sign.detection)  # Convert tuple to list for JSON serialization
        }

        # Send a POST request with the sign data
        try:
            response = requests.post(DATA_URL, json=sign_json)
            if response.status_code == 200:
                print("Sign data sent successfully.")
            else:
                print(f"Failed to send sign data: {response.status_code}")
        except requests.RequestException as e:
            print(f"Error sending sign data: {e}")
    else:
        # Post an empty body if no sign is detected
        try:
            response = requests.post(DATA_URL)
            if response.status_code == 200:
                print("Empty sign data sent successfully.")
            else:
                print(f"Failed to send empty sign data: {response.status_code}")
        except requests.RequestException as e:
            print(f"Error sending empty sign data: {e}")
    return frame

def get_stream(url):
    while True:
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            print("Connected to stream.")
            return cap
        print("Waiting for stream...")
        cap.release()
        time.sleep(2)

def main():
    cap = None
    while True:
        if cap is None or not cap.isOpened():
            cap = get_stream(STREAM_URL)
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Stream hiccup or ended. Reconnecting...")
            cap.release()
            cap = None
            time.sleep(1)
            continue
        try:
            processed = process_frame(frame)
        except Exception as e:
            print(f"Error in process_frame: {e}")
            traceback.print_exc()
            processed = frame  # fallback to original frame
        if DISPLAY:
            cv2.imshow("Stream", processed)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break
    if cap:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()