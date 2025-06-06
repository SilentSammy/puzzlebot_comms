import collections
import cv2
import numpy as np
import os
from backg_poller import BackgroundPoller
from dataclasses import dataclass

@dataclass
class Sign:
    id: int
    box: np.ndarray
    confidence: float
    class_name: str

class SignDetector:
    def __init__(self,
        get_signs_func,
        history_len=4,
        chain_length=6,
        max_chain_gap=2,
    ):
        # _get_signs_func(frame, drawing_frame) -> boxes, sign_ids, confidences, class_names
        self._get_signs_func = get_signs_func

        # Define the signs and their associated class IDs
        self._signs = {
            0: "back",
            1: "left",
            2: "right",
            3: "forw",
            4: "stop",
            5: "yield",
            6: "road_work",
        }

        # Create a deque history for each sign
        self.chain_length = chain_length  # Consecutive frames to consider a sign as seen
        self.max_chain_gap = max_chain_gap  # Maximum gap in frames to consider a sign as seen
        history_length = max(history_len, chain_length + max_chain_gap)
        sign_ids = self._signs.keys()
        self._sign_histories = {sign_id: collections.deque(maxlen=history_length) for sign_id in sign_ids}

        # Background poller for asynchronous processing
        self._bg_poll = BackgroundPoller()

    def get_signs(self, frame, drawing_frame=None) -> list[Sign]:
        result = self._get_signs_func(frame, drawing_frame)
        boxes, sign_ids, confidences, class_names = result if result is not None else ([], [], [], [])
        signs = []
        for box, sign_id, confidence, class_name in zip(boxes, sign_ids, confidences, class_names):
            if sign_id in self._signs:
                signs.append(Sign(id=sign_id, box=box, confidence=confidence, class_name=class_name))
        return signs
    
    def get_best_sign(self, frame, drawing_frame=None) -> Sign:
        """
        Returns the Sign dataclass instance of the most robustly detected sign in the current frame,
        using temporal filtering with self._sign_histories.
        """
        # Get the signs in the frame (Sign dataclass instances)
        signs = self.get_signs(frame)

        # Update histories: for each sign_id, append True if seen, else False
        detected_sign_ids = {sign.id for sign in signs}
        for sign_id in self._sign_histories:
            self._sign_histories[sign_id].append(sign_id in detected_sign_ids)

        # Helper to check if a sign is confirmed
        def is_confirmed(history):
            return sum(history) >= (self.chain_length - self.max_chain_gap)

        # Find all confirmed signs and their counts
        confirmed_signs = []
        for sign_id, history in self._sign_histories.items():
            if is_confirmed(history):
                count = sum(history)
                confirmed_signs.append((sign_id, count))

        best_sign = None
        if confirmed_signs:
            confirmed_signs.sort(key=lambda x: x[1], reverse=True)
            best_sign_id = confirmed_signs[0][0]
            # Find the first matching Sign instance in this frame
            for sign in signs:
                if sign.id == best_sign_id:
                    best_sign = sign
                    break

        if drawing_frame is not None:
            # Draw the confirmed sign in green
            self.draw_signs( drawing_frame, [sign for sign in signs if best_sign and sign.id == best_sign.id], color=(0, 255, 0) )

            # Draw all other signs in red
            self.draw_signs( drawing_frame, [sign for sign in signs if best_sign and sign.id != best_sign.id], color=(0, 0, 255) )

        return best_sign

    def draw_signs(self, frame, signs: list[Sign] , color=(0, 255, 0)):
        """
        Draws bounding boxes and labels for the given signs on the frame.
        """
        for sign in signs:
            box = sign.box
            if box is not None and len(box) == 4:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{sign.class_name} ({sign.confidence:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def get_best_sign_nb(self, frame, drawing_frame=None):
        return self._bg_poll.poll_with_annotated( frame, drawing_frame, lambda af: self.get_best_sign(frame, af) )
