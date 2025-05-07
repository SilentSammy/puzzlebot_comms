import math
import cv2
import requests
import time
import threading

class PuzzlebotHttpClient:
    def __init__(self, base_url="http://192.168.137.139:5000", safe_mode=True):
        self.base_url = base_url
        self.cap = None
        self.safe_mode = safe_mode
        self.prev_v = None
        self.prev_w = None

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
        return self.cap.isOpened()

    def _stop_stream(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

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
        if self.cap is None:
            # Automatically start the stream if it's not running
            if not self._start_stream():
                return None
        ret, frame = self.cap.read()
        if not ret:
            # Stream might have failed, try restarting
            self._stop_stream()
            if not self._start_stream():
                return None
            ret, frame = self.cap.read()
            if not ret:
                return None
        return frame
