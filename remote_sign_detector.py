import cv2
from vision_offloader import VisionOffloader
from sign_detector import Sign, SignDetector

class RemoteSignDetector(SignDetector):
    def __init__(self):
        super().__init__()
        self.vo = VisionOffloader(video_endpoint="video_feed", reception_endpoint="frame_data")
    
    def get_best_sign(self, frame, drawing_frame=None):
        self.vo.offload_frame(frame)
        if self.vo.received_data is None:
            return None
        
        # Otherwise, assume self.vo.received_data contains a JSON as such:
        # {
        #     "id": 1,
        #     "label": "stop",
        #     "detection": [x, y, w, h, class_id, score]
        # }
        data = self.vo.received_data
        best_sign = Sign(
            id=data["id"],
            label=data["label"],
            detection=tuple(data["detection"])
        )

        # Draw the best sign on the drawing frame if provided
        if drawing_frame is not None:
            x, y, w, h, cls, score = best_sign.detection
            cv2.rectangle(drawing_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{best_sign.label}: {score:.2f}"
            cv2.putText(drawing_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return best_sign

    def get_best_sign_nb(self, frame, drawing_frame=None):
        return self.get_best_sign(frame, drawing_frame) # No need for background polling in this case, as the VisionOffloader handles it.
