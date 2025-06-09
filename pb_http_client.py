import math
import cv2
import requests
import time
import threading

class PuzzlebotHttpClient:
    def __init__(self, base_url="http://192.168.137.139:5000", safe_mode=True, id = 0):
        self.id = id
        self.base_url = base_url
        self.cap = None
        self.safe_mode = safe_mode
        self.prev_v = None
        self.prev_w = None
        self.latest_frame = None
        self._frame_thread = None
        self._frame_thread_running = False
        self._buzzer_running = False  # Add this flag

    def play_buzzer(self, melody):
        if self._buzzer_running:
            print("Buzzer request ignored: already running.")
            return
        self._buzzer_running = True
        def _buzzer_thread():
            try:
                json_data = {"melody": melody}
                response = requests.post(f"{self.base_url}/buzzer", json=json_data)
                if response.status_code == 200:
                    print("Buzzer played successfully.")
                else:
                    print("Failed to send melody:", response.status_code, response.text)
            except Exception as ex:
                print("Error sending melody:", ex)
            finally:
                self._buzzer_running = False
        threading.Thread(target=_buzzer_thread, daemon=True).start()

    def _send_vel(self, v=None, w=None):
        params = {}
        if v is not None:
            params["v"] = v
        if w is not None:
            params["w"] = w
        try:
            endpoint = "/cmd_vel_safe" if self.safe_mode else "/cmd_vel"
            response = requests.get(f"{self.base_url}{endpoint}", params=params)
            return response.json()
        except Exception as ex:
            print("Error sending velocity:", ex)
            
    def _start_stream(self):
        self.cap = cv2.VideoCapture(f"{self.base_url}/car_cam")
        if not self.cap.isOpened():
            return False
        self._frame_thread_running = True
        self._frame_thread = threading.Thread(target=self._frame_grabber, daemon=True)
        self._frame_thread.start()
        return True

    def _stop_stream(self):
        self._frame_thread_running = False
        if self._frame_thread is not None:
            self._frame_thread.join(timeout=1)
            self._frame_thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _frame_grabber(self):
        while self._frame_thread_running and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                self.latest_frame = frame
            else:
                # Try to recover if stream fails
                self._stop_stream()
                self._start_stream()
            time.sleep(0.001)  # Small sleep to avoid busy-waiting

    def send_vel(self, v=None, w=None, wait_for_completion=False, force=False):
        if force or self.prev_v != v or self.prev_w != w:
            self.prev_v = v
            self.prev_w = w
            if wait_for_completion:
                return self._send_vel(v, w)
            threading.Thread(target=self._send_vel, args=(v, w), daemon=True).start()

    def get_state(self):
        response = requests.get(f"{self.base_url}/state")
        if response.status_code == 200:
            return response.json()
        else:
            return None

    def get_frame(self):
        if self.cap is None or not self._frame_thread_running:
            # Automatically start the stream if it's not running
            if not self._start_stream():
                return None
        # Wait for the first frame if needed
        wait_count = 0
        while self.latest_frame is None and wait_count < 100:
            time.sleep(0.01)
            wait_count += 1
        return self.latest_frame