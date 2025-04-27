from flask import Flask, request, Response
import threading
import time
import cv2

app = Flask(__name__)
http_endpoints = {}
video_endpoints = {}
default_cap = None

# Catch-all route to process every request from the base
@app.route('/', defaults={'path': ''}, methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
def catch_all(path):
    # Debug prints
    print("Received Request:")
    print("URL Path:", path)
    print("Method:", request.method)
    print("URL Parameters:", request.args)
    print("Form Data:", request.form)
    print("JSON Data:", request.get_json(silent=True))
    print("Raw Data:", request.data)
    
    # First, check if the path corresponds to an HTTP endpoint.
    callback = http_endpoints.get(path)
    if callback:
        return callback(request)

    # Next, check if the path corresponds to a video feed.
    video_callback = video_endpoints.get(path)
    if video_callback:
        def generate():
            while True:
                frame = video_callback()
                if frame is None:
                    continue
                ret, jpeg = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    return "Endpoint not defined", 404

def start_webserver(threaded=True):
    if threaded:
        server_thread = threading.Thread(
            target=lambda: app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
        )
        server_thread.daemon = True
        server_thread.start()
    else:
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

def default_video_source():
    global default_cap
    default_cap = default_cap or cv2.VideoCapture(0)
    ret, frame = default_cap.read()
    return frame if ret else None

# Example usage (for testing purposes)
if __name__ == '__main__':
    v, w = 0.0, 0.0  # Default velocities
    def receive_vel(request):
        global v, w  # or use global if you prefer
        v = request.args.get('v', type=float) or v
        w = request.args.get('w', type=float) or w
        return f"Linear Velocity: {v}, Angular Velocity: {w}", 200

    http_endpoints["cmd_vel"] = receive_vel
    video_endpoints["default"] = default_video_source

    start_webserver(threaded=True)

    # Main thread continues non-blocking work
    while True:
        print("Main thread is running...")
        time.sleep(1)