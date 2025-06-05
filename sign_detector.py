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
from backg_poller import BackgroundPoller


class SignDetector:
    def __init__(self,
        model_path,
        cls_thres=None,
        signs=None,
        history_len=4,
        chain_length=6,
        max_chain_gap=2,
    ):
        self._bg_poll = BackgroundPoller(max_workers=5)

        self.feature_extractor = None  # Will be set if using ResNet50
        self.model = self.load_classifier(model_path)

        # Overwrite some constants in ml_vision.py
        self.cls_thres = cls_thres or {
            13: 0.90,  # Yield
            14: 0.90,  # Stop
            25: 0.60,  # Road work: umbral reducido a 60%
            33: 0.90,  # Turn right ahead
            34: 0.90,  # Turn left ahead
            35: 0.90,  # Ahead only
            38: 0.90,  # Keep right
            39: 0.90,  # Keep left
        }
        self.cls_ids = list(set(self.cls_thres.keys()))
        
        # Multithreading setup
        self._worker = None
        self._lock = threading.Lock()
        self._last_result = None
        self._last_drawing_frame = None

        # Define the signs and their associated class IDs
        self._signs = signs or {
            0: ("back",  []),
            1: ("left",  [ 34, 39 ]),
            2: ("right", [ 33, 38 ]),
            3: ("forw",  [ 35 ]),
            4: ("stop",  [ 14 ]),
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

    def load_classifier(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        model = load_model(model_path)
        num_output = model.output_shape[-1]
        if len(CLASS_NAMES) != num_output:
            raise ValueError(f"Mismatch: {len(CLASS_NAMES)} etiquetas vs. {num_output} salidas del modelo.")
        input_shape = model.input_shape
        global feature_extractor
        if len(input_shape) == 2 and input_shape[1] == 2048:
            self.feature_extractor = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224,224,3))
        return model

    def batch_classify_rois(self, rois):
        """
        Classifies a list of ROIs using the loaded CNN model.
        Returns a list of (class_id, score) tuples.
        """
        if not rois:
            return []
        # Determine input size
        if self.feature_extractor:
            size = (224, 224)
            processed = []
            for roi in rois:
                img = cv2.resize(roi, size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                x = img.astype('float32')
                processed.append(x)
            input_tensor = preprocess_input(np.stack(processed, axis=0))
            feats = self.feature_extractor.predict(input_tensor)
            preds = self.model.predict(feats)
        else:
            h, w = self.model.input_shape[1:3]
            processed = []
            for roi in rois:
                img = cv2.resize(roi, (w, h))
                x = img.astype('float32') / 255.0
                processed.append(x)
            input_tensor = np.stack(processed, axis=0)
            preds = self.model.predict(input_tensor)
        class_ids = np.argmax(preds, axis=1)
        scores = np.max(preds, axis=1)
        return list(zip(class_ids, scores))

    def process_frame(self, frame, drawing_frame=None):
        rois = extract_rois(frame)
        boxes, scores, clases = [], [], []
        # Batch classify all ROIs for speed
        roi_imgs = [frame[y:y+h, x:x+w] for x, y, w, h in rois]
        results = self.batch_classify_rois(roi_imgs)
        for (x, y, w, h), (class_id, score) in zip(rois, results):
            if class_id not in self.cls_ids or score < self.cls_thres[class_id]:
                continue
            boxes.append([x, y, w, h])
            scores.append(score)
            clases.append(class_id)
        dets = self.non_max_suppression(boxes, scores, clases)
        
        if drawing_frame is not None:
            self.draw_detections(drawing_frame, dets)

        return dets
    
    def non_max_suppression(self, boxes, scores, classes):
        min_thresh = min(self.cls_thres.values())
        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=min_thresh, nms_threshold=IOU_THRESHOLD)
        if not len(indices):
            return []
        idxs = indices.flatten()
        return [(boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], classes[i], scores[i]) for i in idxs]
    
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

                # Return the sign tuple with the most appearances in its history
        best_sign = None
        if confirmed_signs:
            confirmed_signs.sort(key=lambda x: x[1], reverse=True)
            best_sign_id = confirmed_signs[0][0]
            # Find the first matching sign tuple in this frame
            for sign in signs:
                if sign[6] == best_sign_id:
                    best_sign = sign
                    break

        if drawing_frame is not None:
            # Draw the confirmed sign in green
            self.draw_detections(
                drawing_frame,
                [sign for sign in signs if best_sign and sign[6] == best_sign[6]],
                color=(0, 255, 0),
                lbl_get=lambda det: f"{det[7]} (confirmed)"
            )

            # Draw all other signs in red
            self.draw_detections(
                drawing_frame,
                [sign for sign in signs if not best_sign or sign[6] != best_sign[6]],
                color=(0, 0, 255),
                lbl_get=lambda det: f"{det[7]} (not confirmed)"
            )

        return best_sign

    def get_best_sign_nb(self, frame, drawing_frame=None):
        return self._bg_poll.poll_with_annotated( frame, drawing_frame, lambda af: self.get_best_sign(frame, af) )

    def draw_detections(self, frame, detections, color=(0, 255, 0), lbl_get=None):
        for det in detections:
            # Draw the rectangle
            x, y, w, h = det[:4]
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            
            # Write the label
            label = lbl_get(det) if lbl_get is not None else f"{CLASS_NAMES[det[4]]}"
            label += f": {det[5]:.2f}"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def segment_by_color(hsv):
    masks = {}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    for color, ranges in COLOR_RANGES.items():
        mask = None
        for lower, upper in ranges:
            m = cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        masks[color] = mask
    return masks

def extract_rois(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    masks = segment_by_color(hsv)
    rois = []
    for mask in masks.values():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_AREA:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) >= 3:
                x, y, w, h = cv2.boundingRect(approx)
                rois.append((x, y, w, h))
    return rois

# CONSTANTS FOR SIGN DETECTION
COLOR_RANGES = {
    'red': [((0, 100, 100), (10, 255, 255)), ((160, 100, 100), (179, 255, 255))],
    'orange': [((5, 100, 100), (15, 255, 255)), ((15, 100, 100), (25, 255, 255))],
    'blue': [((90, 100, 50), (140, 255, 255))],
    'white': [((0, 0, 200), (180, 30, 255))]
}
CLASS_NAMES = [
    'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)',
    'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)',
    'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)',
    'No passing', 'No passing veh over 3.5 tons', 'Right-of-way at intersection',
    'Priority road', 'Yield', 'Stop', 'No vehicles', 'Veh > 3.5 tons prohibited',
    'No entry', 'General caution', 'Dangerous curve left', 'Dangerous curve right',
    'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
    'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing',
    'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
    'End speed + passing limits', 'Turn right ahead', 'Turn left ahead',
    'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right',
    'Keep left', 'Roundabout mandatory', 'End of no passing', 'End no passing veh > 3.5 tons'
]
IOU_THRESHOLD = 0.3
MIN_AREA = 500