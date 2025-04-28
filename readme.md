# Puzzlebot Communications

This repository demonstrates HTTP communication with the Puzzlebot differential robot car via three endpoints exposed by the server.

## Endpoints

- **cmd_vel**  
  Sets the robot's linear (`v`) and angular (`w`) velocities.  
  **Example:**  
  `http://192.168.137.139:5000/cmd_vel?v=0.2&w=0.5`

- **state**  
  Returns the current state as JSON, including `v`, `w`, `x`, `y`, and `theta`.  
  **Example:**  
  `http://192.168.137.139:5000/state`

- **car_cam**  
  Provides an MJPEG stream from the onboard camera.  
  **Example:**  
  `http://192.168.137.139:5000/car_cam`

> **Note:** When testing, ensure you are on the same network as the Puzzlebot's Jetson Nano. Use the Jetson’s IP address with port `5000`.

## Quick Start Guide

You can test the system in two ways:

### 1. Simulation
- Launch CoppeliaSim and open the `sim_demo/vision_test.ttt` scene.
- Run `sim_demo/sim_server.py` in its own terminal.
- Execute `rc_control.py` Make sure to change the IP to `http://127.0.0.1:5000` (at the start of the `rc_control.py` script).

### 2. Real-life Robot
- On the Jetson Nano, after boot, execute:
  - `ros2 launch puzzlebot_ros micro_ros_agent.launch.py`
  - `sudo service nvargus-daemon restart`
  - `ros2 launch pb_http_server launch_http_server.py`
- Run `rc_control.py`, setting the connection URL (at the start of the `rc_control.py` script) to the Jetson’s IP (e.g., `http://192.168.137.139:5000`).

## RC Control Instructions

- **Mode Toggle:**  
  Switch between manual and auto mode by pressing `M` (manual is the default).
  
- **Manual Mode:**  
  Use the keyboard keys:
  - `W` to move forward
  - `S` to move backward
  - `A` to turn left
  - `D` to turn right  
  *Tip:* Ensure the OpenCV window is unfocused to prevent input lag.

- **Auto Mode:**  
  The robot will automatically drive towards an Aruco Marker if detected.

## Using the Python Client

The primary module, `pb_http_client.py`, provides a single class: `PuzzlebotHttpClient`, which simplifies communication with the Puzzlebot. Below is a detailed description of its public methods:

### Class: PuzzlebotHttpClient

#### `send_vel(v, w, wait_for_completion=False, force=False)`
- **Description:**  
  Sends velocity commands to the Puzzlebot.
- **Parameters:**
  - `v` (float): The linear velocity.
  - `w` (float): The angular velocity.
  - `wait_for_completion` (bool):  
    If set to `True`, this method will block until the HTTP request completes and return the server response.  
    Otherwise, the command is sent asynchronously.
  - `force` (bool):  
    If `True`, the command is sent even if it is identical to the previous values (overriding the automatic request suppression).
- **Returns:**  
  When `wait_for_completion` is `True`, returns a dictionary containing the JSON response from the server. Otherwise, returns `None`.

#### `get_state()`
- **Description:**  
  Retrieves the current state of the Puzzlebot.
- **Operation:**  
  Sends a GET request to the `/state` endpoint and parses the JSON response.
- **Returns:**  
  A dictionary containing the state (`v`, `w`, `x`, `y`, `theta`) if the request is successful, or `None` otherwise.

#### `get_frame()`
- **Description:**  
  Captures a frame from the Puzzlebot's camera stream.
- **Operation:**  
  If the camera stream is not active, it automatically starts the stream.  
  If a frame cannot be captured, it attempts to restart the stream.
- **Returns:**  
  A NumPy array representing the captured frame (as returned by OpenCV), or `None` if no frame can be retrieved.
