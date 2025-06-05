import cv2
import numpy as np
import argparse
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import tkinter as tk
from PIL import Image, ImageTk

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


ALLOWED_CLASS_IDS = {13, 14, 24, 25, 33, 34, 35}
IOU_THRESHOLD = 0.3
MIN_AREA = 500

CLASS_THRESHOLDS = {
    13: 0.90,  # Yield
    14: 0.90,  # Stop
    24: 0.60,
    25: 0.60,  # Road work: umbral reducido a 60%
    33: 0.90,  # Turn right ahead
    34: 0.90,  # Turn left ahead
    35: 0.90   # Ahead only
}

feature_extractor = None

def load_classifier(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    model = load_model(model_path)
    num_output = model.output_shape[-1]
    if len(CLASS_NAMES) != num_output:
        raise ValueError(f"Mismatch: {len(CLASS_NAMES)} etiquetas vs. {num_output} salidas del modelo.")
    input_shape = model.input_shape
    global feature_extractor
    if len(input_shape) == 2 and input_shape[1] == 2048:
        feature_extractor = ResNet50(weights='imagenet', include_top=False,
                                     pooling='avg', input_shape=(224,224,3))
    return model


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


def classify_roi(model, roi):
    if feature_extractor:
        img = cv2.resize(roi, (224,224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = preprocess_input(np.expand_dims(img.astype('float32'), axis=0))
        feats = feature_extractor.predict(x)
        preds = model.predict(feats)
    else:
        h, w = model.input_shape[1:3]
        img = cv2.resize(roi, (w, h))
        x = img.astype('float32') / 255.0
        x = np.expand_dims(x, axis=0)
        preds = model.predict(x)
    class_id = int(np.argmax(preds[0]))
    score = float(preds[0][class_id])
    return class_id, score


def non_max_suppression(boxes, scores, classes):
    min_thresh = min(CLASS_THRESHOLDS.values())
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=min_thresh, nms_threshold=IOU_THRESHOLD)
    if not len(indices):
        return []
    idxs = indices.flatten()
    return [(boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], classes[i], scores[i]) for i in idxs]


def draw_detections(frame, detections):
    for x, y, w, h, cls, score in detections:
        if score < CLASS_THRESHOLDS.get(cls, 1.0):
            continue
        label = f"{CLASS_NAMES[cls]}: {score:.2f}"
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)


def process_frame(frame, model):
    rois = extract_rois(frame)
    boxes, scores, clases = [], [], []
    for x, y, w, h in rois:
        class_id, score = classify_roi(model, frame[y:y+h, x:x+w])
        if class_id not in ALLOWED_CLASS_IDS or score < CLASS_THRESHOLDS[class_id]:
            continue
        boxes.append([x, y, w, h])
        scores.append(score)
        clases.append(class_id)
    dets = non_max_suppression(boxes, scores, clases)
    draw_detections(frame, dets)
    return frame


def main(args):
    model = load_classifier(args.model)
    cap = cv2.VideoCapture(0) if args.live else cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir fuente de video")
    if args.live:
        root = tk.Tk()
        root.title("Live Detection")
        label = tk.Label(root)
        label.pack()
        def update_frame():
            ret, frame = cap.read()
            if not ret:
                root.after(10, update_frame)
                return
            frame = cv2.resize(frame, (args.width, args.height))
            output = process_frame(frame, model)
            img = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))
            label.imgtk = imgtk
            label.configure(image=imgtk)
            root.after(10, update_frame)
        update_frame()
        root.mainloop()
        cap.release()
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = args.output or f"processed_{os.path.basename(args.video)}"
        out = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                              (args.width, args.height))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (args.width, args.height))
            out.write(process_frame(frame, model))
        cap.release()
        out.release()
        print(f"Video procesado guardado en '{out_path}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detección de señalamientos')
    parser.add_argument('--model', required=True, help='Ruta al modelo .h5')
    parser.add_argument('--video', help='Ruta al video de entrada')
    parser.add_argument('--live', action='store_true', help='Usar cámara web')
    parser.add_argument('--output', nargs='?', const=None, help='Video de salida (opcional)')
    parser.add_argument('--width', type=int, default=640, help='Ancho del frame')
    parser.add_argument('--height', type=int, default=480, help='Alto del frame')
    parser.add_argument('--src_points', nargs=8, type=float, help='Puntos perspectiva')
    args = parser.parse_args()
    if not args.live and not args.video:
        parser.error("Se requiere --video o --live")
    main(args)