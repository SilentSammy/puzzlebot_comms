from flask import Flask, request, Response, stream_with_context
import threading
import time
import cv2

app = Flask(__name__)
http_endpoints = {}
video_endpoints = {}

def start_webserver(threaded=True):
    if threaded:
        server_thread = threading.Thread(
            target=lambda: app.run(
                host='0.0.0.0',
                port=5000,
                debug=True,
                use_reloader=False,
                threaded=True
            )
        )
        server_thread.daemon = True
        server_thread.start()
    else:
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            use_reloader=False,
            threaded=True
        )

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
    
    # HTTP endpoints
    callback = http_endpoints.get(path)
    if callback:
        return callback(request)

    # Video streaming endpoints
    video_callback = video_endpoints.get(path)
    if video_callback:
        def generate():
            boundary = b'--frame\r\n'
            header = b'Content-Type: image/jpeg\r\n\r\n'
            while True:
                try:
                    frame = video_callback()
                    if frame is None:
                        time.sleep(0.01)
                        continue
                    ret, jpeg = cv2.imencode('.jpg', frame)
                    if not ret:
                        continue
                    yield boundary + header + jpeg.tobytes() + b'\r\n'
                except Exception:
                    app.logger.exception("Error in video stream generator, restarting loop")
                    time.sleep(0.1)
        return Response(
            stream_with_context(generate()),
            mimetype='multipart/x-mixed-replace; boundary=frame',
            headers={'Connection': 'keep-alive'}
        )

    return "Endpoint not defined", 404

# Default video source (for testing or fallback)
def default_video_source():
    global default_cap
    default_cap = globals().get('default_cap') or cv2.VideoCapture(0)
    ret, frame = default_cap.read()
    return frame if ret else None

if __name__ == '__main__':
    # Example usage/demo
    v, w = 0.0, 0.0
    def receive_vel(request):
        global v, w
        v = request.args.get('v', type=float) or v
        w = request.args.get('w', type=float) or w
        return f"Linear Velocity: {v}, Angular Velocity: {w}", 200

    http_endpoints['cmd_vel'] = receive_vel
    video_endpoints['default'] = default_video_source

    start_webserver(threaded=True)

    while True:
        print("Main thread is running...")
        time.sleep(1)
