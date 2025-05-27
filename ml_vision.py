import threading
import numpy as np
import math
import cv2
import time
from collections import deque
from itertools import combinations, count
import os
from tensorflow.keras.models import load_model

# SIGN DETECTION HELPERS
def grid_split(frame, grid_size=2):
    """
    Yields (patch, (x, y)) tuples for all grid patches (main, secondary, tertiary, quaternary).
    """
    h, w = frame.shape[:2]
    min_dim = min(h, w)
    square_size = min_dim // grid_size

    # Main grid (centered)
    x0 = (w - square_size * grid_size) // 2
    y0 = (h - square_size * grid_size) // 2
    for row in range(grid_size):
        for col in range(grid_size):
            x = x0 + col * square_size
            y = y0 + row * square_size
            patch = frame[y:y+square_size, x:x+square_size].copy()
            yield (patch, (x, y))

    # Secondary grid (centered, grid_size-1 x grid_size-1)
    if grid_size > 1:
        sec_size = grid_size - 1
        sec_square_size = square_size
        sec_x0 = (w - sec_square_size * sec_size) // 2
        sec_y0 = (h - sec_square_size * sec_size) // 2
        for row in range(sec_size):
            for col in range(sec_size):
                x = sec_x0 + col * sec_square_size
                y = sec_y0 + row * sec_square_size
                patch = frame[y:y+sec_square_size, x:x+sec_square_size].copy()
                yield (patch, (x, y))

        # Tertiary grid: grid_size x (grid_size-1)
        tert_x0 = (w - square_size * grid_size) // 2
        tert_y0 = (h - square_size * sec_size) // 2
        for row in range(sec_size):
            for col in range(grid_size):
                x = tert_x0 + col * square_size
                y = tert_y0 + row * square_size
                patch = frame[y:y+square_size, x:x+square_size].copy()
                yield (patch, (x, y))

        # Tertiary grid: (grid_size-1) x grid_size
        tert2_x0 = (w - square_size * sec_size) // 2
        tert2_y0 = (h - square_size * grid_size) // 2
        for row in range(grid_size):
            for col in range(sec_size):
                x = tert2_x0 + col * square_size
                y = tert2_y0 + row * square_size
                patch = frame[y:y+square_size, x:x+square_size].copy()
                yield (patch, (x, y))

    # Quaternary grids: two (grid_size x grid_size) grids, flush with edges (if not square)
    if w != h:
        if w > h:
            # Landscape: flush left and right
            for edge_x in [0, w - square_size * grid_size]:
                for row in range(grid_size):
                    for col in range(grid_size):
                        x = edge_x + col * square_size
                        y = (h - square_size * grid_size) // 2 + row * square_size
                        patch = frame[y:y+square_size, x:x+square_size].copy()
                        yield (patch, (x, y))
        else:
            # Portrait: flush top and bottom
            for edge_y in [0, h - square_size * grid_size]:
                for row in range(grid_size):
                    for col in range(grid_size):
                        x = (w - square_size * grid_size) // 2 + col * square_size
                        y = edge_y + row * square_size
                        patch = frame[y:y+square_size, x:x+square_size].copy()
                        yield (patch, (x, y))

def grid_splits(frame, *grid_sizes):
    """
    Yields (patch, (x, y)) tuples for all patches from all specified grid sizes.
    Calls grid_split for each grid_size provided.
    """
    for grid_size in grid_sizes:
        yield from grid_split(frame, grid_size)

def get_class_cnn(patches):
    """
    Classifies a list of patches using the loaded CNN model.
    Returns a list of (class, confidence) tuples.
    """
    get_class_cnn.model = get_class_cnn.model if hasattr(get_class_cnn, "model") else load_model('gtsrb_cnn_98.h5')
    # Preprocess all patches
    processed = []
    for patch in patches:
        if patch.ndim == 2 or patch.shape[2] == 1:
            patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
        patch_resized = cv2.resize(patch, (30, 30))
        patch_norm = patch_resized.astype('float32') / 255.0
        processed.append(patch_norm)
    input_tensor = np.stack(processed, axis=0)
    preds = get_class_cnn.model.predict(input_tensor)
    class_ids = np.argmax(preds, axis=1)
    confidences = np.max(preds, axis=1)
    return list(zip(class_ids, confidences))

def find_feature(frame, feature_func, patches, whitelist=None, top_n=1):
    """
    Returns a list of the top n (patch, coord, class, confidence) tuples.
    If whitelist is provided, only considers patches whose class is in whitelist.
    """
    patch_imgs = [patch for patch, _ in patches]
    results = feature_func(patch_imgs)
    indexed = [
        (i, clss, conf)
        for i, (clss, conf) in enumerate(results)
        if whitelist is None or clss in whitelist
    ]
    if not indexed:
        return []
    # Sort by confidence descending and take top_n
    top = sorted(indexed, key=lambda x: x[2], reverse=True)[:top_n]
    out = []
    for idx, clss, conf in top:
        patch, coord = patches[idx]
        out.append((patch, coord, clss, conf))
    return out

# SIGN DETECTION STAGES
def identify_signs(frame, drawing_frame=None,
    confidence_threshold=0.999,  # Minimum confidence to consider a patch valid
):
    # Signs and descriptions
    signs = {
        0: "back",
        1: "left",
        2: "right",
        3: "forw",
    }

    # Class mapping
    cls_map = {
        33: 2,
        38: 2,
        34: 1,
        39: 1,
        35: 3,
    }
    
    # Split into patches
    patches = list(grid_splits(frame, 2, 4, 6))
    
    # Find the top patches using a CNN classifier
    top_results = find_feature(frame, get_class_cnn, patches, whitelist=list(cls_map.keys()), top_n=1)
    top_results = [
        (patch, coord, clss, conf, cls_map[clss]) for patch, coord, clss, conf in top_results
        if conf >= confidence_threshold
    ]

    # Show the top patches on the frame
    if drawing_frame is not None:
        colors = [(0, 255, 0), (0, 200, 100), (0, 255, 255), (0, 165, 255), (0, 100, 255), (0, 0, 255)]
        for i, (patch, coord, clss, conf, sign_id) in enumerate(top_results):
            x, y = coord
            h, w = patch.shape[:2]
            color = colors[i % len(colors)]
            cv2.rectangle(drawing_frame, (x, y), (x + w, y + h), color, 2)
            label = f"Class: {signs[sign_id]}, Conf: {conf:.2f}"
            # Calculate the position for the text inside the rectangle (bottom-left corner, with padding)
            text_x = x + 5
            text_y = y + h - 10
            cv2.putText(drawing_frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            print(f"Class: {clss}, Confidence: {conf}, Coord: {coord}")
    
    top_results = [sign_id for _, _, _, _, sign_id in top_results]
    return top_results

def identify_signs_nb(frame, drawing_frame=None,
    confidence_threshold=0.997,
):
    """
    Non-blocking version of identify_signs.
    Returns the latest detected sign_ids (list), and overlays the last annotated frame if drawing_frame is provided.
    """
    # Static variables
    if not hasattr(identify_signs_nb, "worker"):
        identify_signs_nb.worker = None
        identify_signs_nb.lock = threading.Lock()
        identify_signs_nb.last_result = []
        identify_signs_nb.last_drawing_frame = None

    with identify_signs_nb.lock:
        result = identify_signs_nb.last_result
        annotated_frame = identify_signs_nb.last_drawing_frame

    # If no worker is running, start one
    if identify_signs_nb.worker is None or not identify_signs_nb.worker.is_alive():
        def worker_func(frame_copy):
            # Create a blank drawing frame for annotation
            annotated = np.zeros_like(frame_copy)
            sign_ids = identify_signs(frame_copy, drawing_frame=annotated, confidence_threshold=confidence_threshold)
            with identify_signs_nb.lock:
                identify_signs_nb.last_result = sign_ids
                identify_signs_nb.last_drawing_frame = annotated

        t = threading.Thread(target=worker_func, args=(frame.copy(),))
        t.daemon = True
        t.start()
        identify_signs_nb.worker = t

    # Overlay the last annotated frame if available
    if drawing_frame is not None and annotated_frame is not None:
        non_black_mask = np.any(annotated_frame != 0, axis=2)
        drawing_frame[non_black_mask] = annotated_frame[non_black_mask]

    return result
