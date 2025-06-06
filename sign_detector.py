import cv2
from dataclasses import dataclass

@dataclass
class Sign:
    id: int
    label: str
    detection: tuple  # (x, y, w, h, class_id, score)

# Abstract class for sign detection
class SignDetector:
    def get_best_sign(self, frame, drawing_frame=None) -> Sign:
        """
        Returns the Sign dataclass instance of
        the most robustly detected sign in the current frame.
        If no sign is detected, returns None.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_best_sign_nb(self, frame, drawing_frame=None) -> Sign:
        """
        Non-blocking version of get_best_sign.
        Returns the Sign dataclass instance of the most robustly detected sign in the current frame.
        If no sign is detected, returns None.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def draw_detections(self, frame, detections):
        # This method expects a detection to be a tuple (x, y, w, h, class_id, score)
        color=(0, 255, 0)
        for det in detections:
            x, y, w, h, cls, score = det
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{cls}: {score:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
    def draw_signs(self, frame, signs, color=(0, 255, 0)):
        """
        Draws the detected signs on the frame.
        Expects signs to be a list of Sign dataclass instances.
        """
        for sign in signs:
            x, y, w, h, cls, score = sign.detection
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{sign.label}: {score:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
