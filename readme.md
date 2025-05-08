# Puzzlebot Communications

This repository demonstrates HTTP communication with the Puzzlebot differential robot car via several endpoints exposed by the server. (So you can control it from Windows!)

## Endpoints

- **cmd_vel**  
  Sets the robot's linear (`v`) and angular (`w`) velocities.
  **Example:**  
  `http://192.168.137.139:5000/cmd_vel?v=0.2&w=0.5`

- **cmd_vel_safe**  
  Allows setting higher linear (`v`) and angular (`w`) velocities, by limiting acceleration to prevent overcurrent.
  **Example:**  
  `http://192.168.137.139:5000/cmd_vel_safe?v=0.5&w=1.0`

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

- Launch CoppeliaSim and open the [`./sim_demo/navigation_demo.ttt`](./sim_demo/navigation_demo.ttt) scene.
- Run `sim_demo/sim_server.py` in its own terminal.
- Execute `rc_control.py`  
  Make sure to change the IP to `http://127.0.0.1:5000` (at the start of the `rc_control.py` script).

### 2. Real-life Robot
- On the Jetson Nano, after boot, execute:
  - `ros2 launch puzzlebot_ros micro_ros_agent.launch.py`
  - `sudo service nvargus-daemon restart`
  - `ros2 run accel_limiter accel_limiter`
  - `ros2 launch pb_http_server launch_http_server.py`
- Run `rc_control.py`, setting the connection URL (at the start of the [`rc_control.py`](./rc_control.py) script) to the Jetson’s IP (e.g., `http://192.168.137.139:5000`).

## RC Control Instructions

- **Navigation Mode Selection via Radio Buttons:**  
  The keys `1`, `2`, and `3` are configured as radio buttons:
  - Pressing **1** disables any auto-navigation algorithm (allowing pure manual control).
  - Pressing **2** enables auto-navigation using ArUco marker detection.
  - Pressing **3** enables auto-navigation using line following.
  
- **Safe Mode Toggle:**  
  Additionally, you can toggle safe mode by (for example) pressing a designated key (such as `Z` in the code), which sets the client into safe mode. When enabled, velocity commands use the `/cmd_vel_safe` endpoint where acceleration is smoothly ramped to safe levels.

- **Manual Control:**  
  Use the keyboard keys:
  - `W` to move forward
  - `S` to move backward
  - `A` to turn left
  - `D` to turn right  
  *Tip:* Ensure the OpenCV window is unfocused (e.g: by clicking on the taskbar) to prevent the video from lagging.

- **Auto Mode:**  
  When an auto-navigation key is active (either `2` for marker tracking or `3` for line following), the robot automatically drives toward the target. When no auto algorithm is active (pressing `1`), only manual control is applied.

## Using the Python Client

The primary module, `pb_http_client.py`, provides a single class: `PuzzlebotHttpClient`, which simplifies communication with the Puzzlebot. Key details:

### Class: PuzzlebotHttpClient

- The client now uses the safe mode flag to choose the appropriate endpoint:
  - When safe mode is **OFF**, it sends commands to **`/cmd_vel`**.
  - When safe mode is **ON**, it sends commands to **`/cmd_vel_safe`** so that the motor controller can smoothly ramp velocities using internal acceleration limits.

#### `send_vel(v, w, wait_for_completion=False, force=False)`
- **Description:**  
  Sends velocity commands to the Puzzlebot.
- **Parameters:**
  - `v` (float): The linear velocity.
  - `w` (float): The angular velocity.
  - `wait_for_completion` (bool):  
    If set to `True`, this method will block until the HTTP request completes and return the server response; otherwise, the command is sent asynchronously.
  - `force` (bool):  
    When `True`, the command is sent even if it is identical to the previous values.
- **Behavior:**  
  If safe mode is active, the velocities are sent to `/cmd_vel_safe` where a separate control loop in the server gently updates the robot's speeds.

#### `get_state()`
- **Description:**  
  Retrieves the current state of the Puzzlebot.
- **Returns:**  
  A dictionary containing `v`, `w`, `x`, `y`, and `theta` if the request is successful, or `None` otherwise.

#### `get_frame()`
- **Description:**  
  Captures a frame from the Puzzlebot's camera stream.
- **Returns:**  
  A NumPy array (as returned by OpenCV) representing the captured frame, or `None` if no frame is available.

Happy testing and tweaking!
