import threading
from time import sleep
import numpy as np

class BackgroundPoller:
    def __init__(self):
        self._result = None
        self._lock = threading.Lock()
        self._worker = None

    def poll(self, func):
        with self._lock:
            result = self._result

        if self._worker is None or not self._worker.is_alive():
            def worker_func():
                value = func()
                with self._lock:
                    self._result = value
            t = threading.Thread(target=worker_func)
            t.daemon = True
            t.start()
            self._worker = t

        return result
    
    def poll_with_annotated(self, frame, drawing_frame, func):
        """
        Helper for non-blocking vision functions that use a drawing frame.
        - frame: input image
        - drawing_frame: output image to overlay results on
        - func: function of (annotated_frame) -> result
        """
        def worker_func():
            annot = np.zeros_like(frame)
            result = func(annot)
            return result, annot

        result = self.poll(worker_func)
        if result is not None:
            value, annotated_frame = result
            if drawing_frame is not None and annotated_frame is not None:
                non_black_mask = np.any(annotated_frame != 0, axis=2)
                drawing_frame[non_black_mask] = annotated_frame[non_black_mask]
            return value
        return None