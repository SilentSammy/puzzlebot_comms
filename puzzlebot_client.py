import cv2
import requests
import time
import threading
import math

class PuzzlebotClient:
    def __init__(self, base_url="http://192.168.137.139:5000", trim=0.0):
        self.base_url = base_url
        self.trim = trim  # Motor balance compensation: negative slows left, positive slows right
        self.cap = None
        self.prev_cmd = {}
        self.latest_frame = None
        self._frame_thread = None
        self._frame_thread_running = False

    def _apply_trim(self, cmd):
        """Apply trim compensation to command dict.
        
        Adds corrective steering proportional to forward speed:
        w_corrected = w + (x * trim)
        
        Positive trim steers right, negative trim steers left.
        """
        if self.trim == 0.0 or not cmd:
            return cmd
        
        x = cmd.get('x', 0.0)
        w = cmd.get('w', 0.0)
        
        # Add corrective steering proportional to forward speed
        w_corrected = w + (x * self.trim)
        
        return {'x': x, 'w': w_corrected}

    def _send_vel(self, cmd):
        """Send velocity command as dict {'x': ..., 'w': ...}"""
        # Apply trim compensation
        cmd = self._apply_trim(cmd)
        
        params = {}
        
        if 'x' in cmd and cmd['x'] is not None:
            params['x'] = cmd['x']
            
        if 'w' in cmd and cmd['w'] is not None:
            params['w'] = cmd['w']
        
        if not params:
            return
            
        try:
            response = requests.get(f"{self.base_url}/cmd_vel", params=params)
            return response.json()
        except Exception as ex:
            print(f"Error sending velocity: {ex}")
            
    def _start_stream(self):
        """Start MJPEG stream from /stream/car_cam"""
        self.cap = cv2.VideoCapture(f"{self.base_url}/stream/car_cam")
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
            time.sleep(0.001)

    def send_vel(self, cmd, wait_for_completion=False, force=False):
        """Send velocity command as dict, e.g., {'x': 0.5, 'w': 0.2}"""
        if force or self.prev_cmd != cmd:
            self.prev_cmd = cmd.copy()
            if wait_for_completion:
                return self._send_vel(cmd)
            threading.Thread(target=self._send_vel, args=(cmd,), daemon=True).start()

    def get_frame(self):
        """Get latest camera frame. Auto-starts stream if not running."""
        if self.cap is None or not self._frame_thread_running:
            if not self._start_stream():
                return None
        # Wait for first frame
        wait_count = 0
        while self.latest_frame is None and wait_count < 100:
            time.sleep(0.01)
            wait_count += 1
        return self.latest_frame


def merge_proportional(cmd_primary, cmd_secondary):
    """Merge two command dicts where primary proportionally overrides secondary.
    
    Primary command overrides secondary based on its magnitude. Small primary inputs
    (<0.05) pass through secondary values. Larger inputs interpolate toward full ±1.0.
    
    Args:
        cmd_primary: Primary command dict (e.g., manual control)
        cmd_secondary: Secondary command dict (e.g., autonomous control)
    
    Returns:
        Merged command dict
    """
    cmd_final = {}
    
    # Handle all axes from both commands
    all_axes = set(cmd_primary.keys()) | set(cmd_secondary.keys())
    
    for axis in all_axes:
        primary_input = cmd_primary.get(axis, 0.0)
        secondary_input = cmd_secondary.get(axis, 0.0)
        
        if abs(primary_input) < 0.05:  # No primary input
            cmd_final[axis] = secondary_input
        else:
            # Primary input interpolates between secondary and desired value
            override_strength = abs(primary_input)
            desired_value = 1.0 if primary_input > 0 else -1.0
            cmd_final[axis] = (1 - override_strength) * secondary_input + override_strength * desired_value
    
    return cmd_final
