import math
import cv2
import requests
import time
import threading

class PuzzlebotClient:
    def __init__(self, base_url="http://192.168.137.139:5000"):
        self.base_url = base_url
    
def send_vel(self, linear=None, angular=None):
    """Send velocity command to the simulation webserver."""
    try:
        params = {"v": linear, "w": angular}
        response = requests.get(f"{self.base_url}/cmd_vel", params=params)
    except Exception as ex:
        print("Error sending velocity:", ex)

def send_vel_async(linear, angular):
    """Dispatch send_vel asynchronously."""
    threading.Thread(target=send_vel, args=(linear, angular), daemon=True).start()
