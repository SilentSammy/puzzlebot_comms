import os
from simple_pid import PID
import numpy as np
import math
import cv2
import time
import keybrd
from collections import deque
import visual_navigation as vn

class VideoPlayer:
    def __init__(self, frame_source):
        self.frame_source = frame_source
        self.frame_count = 0
        self._frame_idx = 0.0
        self.fps = 30  # Default FPS
        self._get_frame = None
        self.last_time = None
        self.dt = 0.0
        self.setup_video_source()
        self.first_time = True

    def show_frame(self, img, name, scale=1):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)
        if self.first_time:
            self.first_time = False
            cv2.resizeWindow(name, int(img.shape[1]*scale), int(img.shape[0]*scale))
            self.first_time = False
        cv2.imshow(name, img)
        if cv2.waitKey(1) & 0xFF == 27:
            raise KeyboardInterrupt

    def get_frame(self, idx=None):
        if idx is None:
            idx = self.frame_idx
        return self._get_frame(idx)

    def step(self, step_size=1):
        self._frame_idx += step_size
        self._frame_idx = self._frame_idx % self.frame_count
    
    def time_step(self):
        self.dt = time.time() - self.last_time if self.last_time is not None else 0.0
        self.last_time = time.time()
        return self.dt

    def move(self, speed=1):
        self._frame_idx += speed * self.dt * self.fps
        self._frame_idx = self._frame_idx % self.frame_count

    @property
    def frame_idx(self):
        return int(self._frame_idx)

    def setup_video_source(self):
        # If frame_source is a folder, load images
        if os.path.isdir(self.frame_source):
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
            image_files = sorted([
                os.path.join(self.frame_source, f) 
                for f in os.listdir(self.frame_source) 
                if f.lower().endswith(image_extensions)
            ])
            self.frame_count = len(image_files)
            print("Total frames (images):", self.frame_count)
            
            def get_frame(idx):
                idx = int(idx)
                if idx < 0 or idx >= len(image_files):
                    print("Index out of bounds:", idx)
                    return None
                frame = cv2.imread(image_files[idx])
                if frame is None:
                    print("Failed to load image", image_files[idx])
                return frame
            
            self._get_frame = get_frame
        else:
            # Assume frame_source is a video file.
            cap = cv2.VideoCapture(self.frame_source)
            if not cap.isOpened():
                print("Error opening video file:", self.frame_source)
                exit(1)
            
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print("Total frames:", self.frame_count)
            
            def get_frame(idx):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    print("Failed to get frame", idx)
                    return None
                return frame
            
            self._get_frame = get_frame

line_detection_pipeline = [
    ("adaptive_thres", lambda: vn.adaptive_thres(frame, drawing_frame)),
    # ("gray_mask", lambda: get_gray_mask(frame, drawing_frame)),
    # ("refined_mask", lambda: refined_mask(frame, drawing_frame)),
    ("line_mask", lambda: vn.get_line_mask(frame, drawing_frame)),
    ("line_candidates", lambda: vn.get_line_candidates(frame, drawing_frame)),
    ("middle_line", lambda: vn.get_middle_line(frame, drawing_frame)),
    ("follow_line", lambda: vn.follow_line(frame, drawing_frame)),
]

stoplight_pipeline = [
    ("red", lambda: vn.adaptive_color_thresh( frame, drawing_frame )),
    ("yellow", lambda: vn.adaptive_color_thresh( frame, drawing_frame=drawing_frame, target_hue=30, hue_tol=12, sat_thresh=80 )),
    ("green", lambda: vn.adaptive_color_thresh( frame, drawing_frame=drawing_frame, target_hue=65, hue_tol=20, sat_thresh=30 )),

    ("red_ellipses", lambda: vn.ellipse_mask( vn.adaptive_color_thresh( frame ), drawing_frame=drawing_frame )),
    ("yellow_ellipses", lambda: vn.ellipse_mask( vn.adaptive_color_thresh( frame, target_hue=30, hue_tol=12, sat_thresh=80 ), drawing_frame=drawing_frame,  )),
    ("green_ellipses", lambda: vn.ellipse_mask( vn.adaptive_color_thresh( frame, target_hue=65, hue_tol=20, sat_thresh=30 ), drawing_frame=drawing_frame )),

    ("stoplight_mask", lambda: vn.stoplight_mask( frame, drawing_frame=drawing_frame )),
    ("identify_stoplight", lambda: print(vn.identify_stoplight( frame, drawing_frame=drawing_frame ))),
]

intersection_pipeline = [
    ("undistort", lambda: vn.undistort_fisheye(frame, drawing_frame=drawing_frame, zoom=False)),
    ("dark_mask", lambda: vn.get_dark_mask(frame, drawing_frame=drawing_frame)),
    ("mask_intersection", lambda: vn.find_dots(frame, drawing_frame=drawing_frame)),
]

checkerboard_pipeline = [
    ("estimate_distance_from_height", lambda: vn.estimate_distance_from_flag(frame, drawing_frame=drawing_frame)),
]

if __name__ == "__main__":
    import keybrd
    # vp = VideoPlayer(r"resources/videos/track7.mp4")  # Path to the video file
    vp = VideoPlayer(r"resources\screenshots\calibration")  # Path to the image folder
    # vp = VideoPlayer(r"resources/videos/stoplight_test.mp4")  # Path to the video file
    # vp = VideoPlayer(r"resources\videos\output_2025-05-21_16-14-54.mp4")  # Path to the video file
    re = keybrd.rising_edge # Function to check if a key is pressed once
    pr = keybrd.is_pressed  # Function to check if a key is held down
    tg = keybrd.is_toggled  # Function to check if a key is toggled
    layers = checkerboard_pipeline
    layer = 1
    
    while True:
        # Get current frame
        vp.time_step()
        vp.move(1 if pr('d') else -1 if pr('a') else 0)  # Move forward/backward
        vp.move((1 if pr('e') else -1 if pr('q') else 0) * 10)  # Fast forward/backward
        vp.step(1 if re('w') else -1 if re('s') else 0)  # Step forward/backward
        mask = None
        frame = vp.get_frame()
        drawing_frame = frame.copy()

        # Print the current frame
        print(f"Frame {vp.frame_idx}/{vp.frame_count} ", end='')

        # Choose layer to show
        for i in range(1, 10):
            if re(str(i)):
                layer = i
                break

        # Choose the layer to show. Layer 1 is do nothing. Layer 2 is index 0 in the pipeline, etc.
        if layer >= 2 and layer <= len(layers) + 1:
            name, func = layers[layer - 2]
            print(name, end=', ')
            func()

        print()
    
        if re('p'): # Save the current frame as an image.
            output_file = f"frame_{vp.frame_idx}_layer_{layer}.png"
            cv2.imwrite(output_file, drawing_frame)
            print(f"Saved frame {vp.frame_idx} as {output_file}")

        # Show
        vp.show_frame(drawing_frame, "Frame")
