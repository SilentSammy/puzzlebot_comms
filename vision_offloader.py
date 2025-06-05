import web
import cv2
import numpy as np

class VisionOffloader:
    def __init__(self, video_endpoint, reception_endpoint):
        web.video_endpoints[video_endpoint] = lambda: self._last_frame
        web.http_endpoints[reception_endpoint] = self._receive_data

        # Last input frame and output data
        self._last_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Placeholder for a blank frame
        cv2.putText(self._last_frame, "Waiting for data...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        self.received_data = None

        if not web.STARTED:
            web.port = 5002
            web.start_webserver(threaded=True)

    def offload_frame(self, frame):
        """Offload a frame to the web server."""
        self._last_frame = frame
    
    def _receive_data(self, request):
        # Receive JSON data from the request body
        self.received_data = request.get_json(silent=True)
        if self.received_data is None:
            return "No data received", 400
        return "Data received successfully", 200
