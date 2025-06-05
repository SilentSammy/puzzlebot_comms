import collections
import cv2
import numpy as np
import argparse
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import tkinter as tk
from PIL import Image, ImageTk
import threading
import ml_vision as mv

class SignDetector:
    def __init__(self,
        model_path,
        cls_thres=None,
        signs=None,
        history_len=4,
        chain_length=3,
        max_chain_gap=1,
    ):
        self.model = mv.load_classifier(model_path)

        # Overwrite some constants in ml_vision.py
        mv.CLASS_THRESHOLDS = cls_thres or {
            13: 0.90,  # Yield
            14: 0.90,  # Stop
            25: 0.60,  # Road work: umbral reducido a 60%
            33: 0.90,  # Turn right ahead
            34: 0.90,  # Turn left ahead
            35: 0.90,  # Ahead only
            38: 0.90,  # Keep right
            39: 0.90,  # Keep left
        }
        mv.ALLOWED_CLASS_IDS = set(mv.CLASS_THRESHOLDS.keys())
        
        # Multithreading setup
        self._worker = None
        self._lock = threading.Lock()
        self._last_result = None
        self._last_drawing_frame = None

        # Define the signs and their associated class IDs
        self._signs = signs or {
            0: ("back", []),
            1: ("left", [ 34, 39 ]),
            2: ("right", [ 33, 38 ]),
            3: ("forw", [ 35 ]),
            4: ("stop", [ 14 ]),
            5: ("yield", [ 13 ]),
            6: ("road_work", [ 25 ]),
        }

        # Dictionary of just the sign ids to their labels
        self._sign_lbls = {v: k[0] for v, k in self._signs.items()}

        # Generate _cls_to_sign_map from self._signs
        self._cls_to_sign_map = {}
        for sign_id, (label, class_ids) in self._signs.items():
            for cls_id in class_ids:
                self._cls_to_sign_map[cls_id] = sign_id
                
        # Create a deque history for each sign
        self.chain_length = chain_length  # Consecutive frames to consider a sign as seen
        self.max_chain_gap = max_chain_gap  # Maximum gap in frames to consider a sign as seen
        history_length = max(history_len, chain_length + max_chain_gap)
        sign_ids = self._signs.keys()
        self._sign_histories = {sign_id: collections.deque(maxlen=history_length) for sign_id in sign_ids}
        
    def process_frame(self, frame, drawing_frame=None):
        rois = mv.extract_rois(frame)
        boxes, scores, clases = [], [], []
        for x, y, w, h in rois:
            class_id, score = mv.classify_roi(self.model, frame[y:y+h, x:x+w])
            if class_id not in mv.ALLOWED_CLASS_IDS or score < mv.CLASS_THRESHOLDS[class_id]:
                continue
            boxes.append([x, y, w, h])
            scores.append(score)
            clases.append(class_id)
        dets = mv.non_max_suppression(boxes, scores, clases)
        
        if drawing_frame is not None:
            self.draw_detections(drawing_frame, dets)

        return dets

    def process_frame_nb(self, frame, drawing_frame=None):
        """
        Non-blocking version of process_frame.
        Returns the last detections and optionally overlays the last annotated frame.
        """
        with self._lock:
            result = self._last_result
            annotated_frame = self._last_drawing_frame

        # Start a new worker if none is running
        if self._worker is None or not self._worker.is_alive():
            def worker_func(frame_copy, drawing_frame):
                dets = self.process_frame(frame_copy, drawing_frame=drawing_frame)
                with self._lock:
                    self._last_result = dets
                    self._last_drawing_frame = drawing_frame

            t = threading.Thread(
                target=worker_func,
                args=(frame.copy(), np.zeros_like(frame)),
            )
            t.daemon = True
            t.start()
            self._worker = t

        if drawing_frame is not None and annotated_frame is not None:
            # Overwrite the drawing_frame pixels with annotated_frame pixels wherever the mask is True.
            non_black_mask = np.any(annotated_frame != 0, axis=2)
            drawing_frame[non_black_mask] = annotated_frame[non_black_mask]

        return result

    def get_signs(self, frame, drawing_frame=None):
        # Get tensorflow detections (x, y, w, h, class_id, score)
        dets = self.process_frame(frame)

        # Append our custom sign_ids to each detection based on class_id (x, y, w, h, class_id, score, +sign_id, +sign_label)
        dets = [det for det in dets if det[4] in self._cls_to_sign_map]  # Filter out detections not in cls_to_sign_map
        for i, det in enumerate(dets):
            x, y, w, h, cls, score = det
            dets[i] = (x, y, w, h, cls, score, self._cls_to_sign_map[cls], self._sign_lbls[self._cls_to_sign_map[cls]])

        if drawing_frame is not None:
            self.draw_detections(drawing_frame, dets, lbl_get=lambda det: f"{det[7]}")

        return dets
    
    def get_best_sign(self, frame, drawing_frame=None):
        """
        Returns the sign_id of the most robustly detected sign in the current frame,
        using temporal filtering with self._sign_histories.
        """
        # Get the signs in the frame (x, y, w, h, class_id, score, sign_id, sign_label)
        signs = self.get_signs(frame)

        # Update histories: for each sign_id, append True if seen, else False
        detected_sign_ids = {sign[6] for sign in signs}  # sign_id is 6th element in the tuple
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

        # Return the sign_id with the most appearances in its history
        confirmed_sign = None
        if confirmed_signs:
            confirmed_signs.sort(key=lambda x: x[1], reverse=True)
            confirmed_sign = confirmed_signs[0][0]

        if drawing_frame is not None:
            # Draw the confirmed sign in green
            self.draw_detections(
                drawing_frame,
                [sign for sign in signs if sign[6] == confirmed_sign],
                color=(0, 255, 0),
                lbl_get=lambda det: f"{det[7]} (confirmed)"
            )

            # Draw all other signs in red
            self.draw_detections(
                drawing_frame,
                [sign for sign in signs if sign[6] != confirmed_sign],
                color=(0, 0, 255),
                lbl_get=lambda det: f"{det[7]} (not confirmed)"
            )

        return confirmed_sign

    def draw_detections(self, frame, detections, color=(0, 255, 0), lbl_get=None):
        for det in detections:
            # Draw the rectangle
            x, y, w, h = det[:4]
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            
            # Write the label
            label = lbl_get(det) if lbl_get is not None else f"{mv.CLASS_NAMES[det[4]]}"
            label += f": {det[5]:.2f}"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)