import flask
import cv2
import numpy as np

app = flask.Flask(__name__)

@app.route("/get_sign", methods=["POST"])
def get_sign():
    # Expect image as form-data file with key 'image'
    if 'image' not in flask.request.files:
        return flask.jsonify({"error": "No image uploaded"}), 400
    file = flask.request.files['image']
    # Convert to OpenCV frame
    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        return flask.jsonify({"error": "Invalid image"}), 400
    # Save latest frame for debugging
    cv2.imwrite("latest_sign.jpg", frame)
    # Return dummy result
    return flask.jsonify({"sign_id": 0})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
