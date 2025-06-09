import os
import cv2
import time
import numpy as np


class VideoRecorder:
    def __init__(self, dir_path="./resources/videos", fps=30):
        self.dir_path = dir_path
        self.fps = fps
        self.vw = None
        self.filename = None

    def start(self, frame):
        if self.vw is not None:
            self.stop()
        os.makedirs(self.dir_path, exist_ok=True)
        height, width = frame.shape[:2]
        self.filename = os.path.join(self.dir_path, time.strftime("output_%Y-%m-%d_%H-%M-%S.mp4"))
        fourcc = cv2.VideoWriter.fourcc(*'avc1')  # Use 'mp4v' for better compatibility
        self.vw = cv2.VideoWriter(self.filename, fourcc, self.fps, (width, height))
        if not self.vw.isOpened():
            print(f"Failed to open video writer for {self.filename}")
        else:
            print(f"Video recording started: {self.filename}")

    def write(self, frame):
        if self.vw is not None and frame is not None:
            self.vw.write(frame)

    def stop(self):
        if self.vw is not None:
            self.vw.release()
            print(f"Video recording closed: {self.filename}")
            self.vw = None
            self.filename = None

    def is_recording(self):
        return self.vw is not None

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
        return self._get_frame(idx) # type: ignore

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
        # If frame_source is a cv2.VideoCapture object, use it directly
        if isinstance(self.frame_source, cv2.VideoCapture):
            cap = self.frame_source
            if not cap.isOpened():
                print("Error opening video file")
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
        # If frame_source is a folder, load images
        elif os.path.isdir(self.frame_source):
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

if __name__ == "__main__":
    import visual_navigation as vn
    from input_manager import keybrd
    # from yolo import get_signs

    line_foll = vn.LineFollower()
    sl_det = vn.StoplightDetector(hsv_val_range=(120, 255))
    fl_det = vn.FlagDetector(pattern_size=(4, 3), square_size=0.025)
    in_det = vn.IntersectionDetector()
    sl_nav = vn.StoplightNavigator(
        line_follower=line_foll, 
        stoplight_detector=sl_det, 
        flag_detector=fl_det
    )
    int_nav = vn.IntersectionNavigator(
        line_follower=line_foll,
        intersection_detector=in_det,
    )
    sg_det = vn.SignDetector(
        # get_signs_func=lambda f: get_signs(f) # Uncomment this line to use YOLO for sign detection
    )

    line_detection_pipeline = [
        ("adaptive_thres", lambda: line_foll.adaptive_thres(frame, drawing_frame)),
        ("line_mask", lambda: line_foll.get_line_mask(frame, drawing_frame)),
        ("line_candidates", lambda: line_foll.get_line_candidates(frame, drawing_frame)),
        ("id_lines", lambda: line_foll.id_line_candidates(frame, drawing_frame)),
        ("middle_line", lambda: line_foll.get_middle_line(frame, drawing_frame)),
        ("persistent_line", lambda: line_foll.get_persistent_line(frame, drawing_frame)),
        ("follow_line", lambda: line_foll.follow_line(frame, drawing_frame)),
    ]

    stoplight_pipeline = [
        ("find_solid_blobs", lambda: sl_det.find_solid_blobs(frame, drawing_frame=drawing_frame)),
        ("find_elliptical_blobs", lambda: sl_det.find_elliptical_blobs(frame, drawing_frame=drawing_frame)),

        ("canny_edges", lambda: sl_det.canny_edges(frame, drawing_frame=drawing_frame)),
        ("ellipses", lambda: sl_det.detect_elliptical_edges(frame, drawing_frame=drawing_frame)),

        ("get_all_ellipses", lambda: sl_det.find_all_elliptical_candidates(frame, drawing_frame=drawing_frame)),

        ("solid_ellipses", lambda: sl_det.filter_solid_color_ellipses(frame, drawing_frame=drawing_frame)),
        ("filtered_ellipses", lambda: sl_det.filter_hsv_ellipses(frame, drawing_frame=drawing_frame)),
        ("classify_ellipses", lambda: sl_det.classify_stoplight_ellipses(frame, drawing_frame=drawing_frame)),
        ("confirm_stoplight", lambda: sl_det.identify_stoplight(frame, drawing_frame=drawing_frame)),
    ]

    chessboard = [
        ("get_flag_distance", lambda: print(fl_det.get_flag_distance(frame, drawing_frame=drawing_frame))),
        ("get_flag_distance_nb", lambda: print(fl_det.get_flag_distance_nb(frame, drawing_frame=drawing_frame))),
    ]

    intersection_pipeline = [
        ("dark_mask", lambda: in_det.get_dark_mask(frame, drawing_frame=drawing_frame)),
        ("find_dots", lambda: in_det.find_dots(frame, drawing_frame=drawing_frame)),
        ("find_dotted_lines", lambda: in_det.find_dotted_lines(frame, drawing_frame=drawing_frame)),
        ("find_intersection", lambda: in_det.find_intersection(frame, drawing_frame=drawing_frame)),
    ]

    algorithms = [
        ("follow_line", lambda: line_foll.follow_line(frame, drawing_frame)),
        ("follow_line_w_signs", lambda: sl_nav.navigate(frame, drawing_frame)),
        ("navigate_track", lambda: int_nav.navigate(frame, drawing_frame))
    ]

    signs_pipeline = [
        ("get_signs", lambda: sg_det.get_signs(frame, drawing_frame)),
        ("set_sign_distances", lambda: sg_det.set_sign_distances(frame, drawing_frame)),
        ("get_confirmed_signs_nb", lambda: sg_det.get_confirmed_signs(frame, drawing_frame)),
    ]
    
    vp = VideoPlayer(r"resources\videos\stoplight_monday.mp4")  # Path to the video file
    # vp = VideoPlayer(r"http://192.168.137.165:5000/car_cam")  # Path to the video file
    re = keybrd.rising_edge # Function to check if a key is pressed once
    pr = keybrd.is_pressed  # Function to check if a key is held down
    tg = keybrd.is_toggled  # Function to check if a key is toggled
    layers = stoplight_pipeline
    layer = 1
    
    while True:
        # Get current frame
        vp.time_step()
        vp.move(1 if pr('d') else -1 if pr('a') else 0)  # Move forward/backward
        vp.move((1 if pr('e') else -1 if pr('q') else 0) * 10)  # Fast forward/backward
        vp.step(1 if re('w') else -1 if re('s') else 0)  # Step forward/backward
        mask = None
        frame = vp.get_frame()
        drawing_frame = frame.copy() # type: ignore

        # Print the current frame
        print(f"Frame {vp.frame_idx}/{vp.frame_count} ", end='')

        # Choose layer to show
        for i in range(1, 10):
            if re(str(i)):
                layer = i
                break
        if re('0'):
            layer = 10

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
