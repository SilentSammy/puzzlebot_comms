import math
import cv2
import requests
import time
import threading

class PuzzlebotHttpClient:
    def __init__(self, base_url="http://192.168.137.139:5000"):
        self.base_url = base_url
        self.cap = None
    
    def send_vel(self, linear=None, angular=None):
        params = {}
        if linear is not None:
            params["v"] = linear
        if angular is not None:
            params["w"] = angular
        try:
            response = requests.get(f"{self.base_url}/cmd_vel", params=params)
        except Exception as ex:
            print("Error sending velocity:", ex)

    def send_vel_async(self, linear, angular):
        threading.Thread(target=self.send_vel, args=(linear, angular), daemon=True).start()
        
    def start_stream(self):
        self.cap = cv2.VideoCapture(f"{self.base_url}/car_cam")
        return self.cap.isOpened()

    def stop_stream(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def get_frame(self):
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        if not ret:
            self.stop_stream()
            self.start_stream()
            return None
        return frame

    def get_state(self):
        response = requests.get(f"{self.base_url}/state")
        if response.status_code == 200:
            return response.json()
        else:
            return None