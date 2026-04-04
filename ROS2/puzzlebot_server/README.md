# Puzzlebot Server

HTTP/WebSocket server for controlling the Puzzlebot and streaming camera feed.

## Features

- **HTTP Endpoint**: `/cmd_vel` - Send velocity commands via query parameters `x` (linear m/s) and `w` (angular rad/s)
- **MJPEG Stream**: `/stream/car_cam` - Camera feed for browsers
- **WebSocket Stream**: `/ws/car_cam` - Camera feed for programmatic access

## Dependencies

### System Packages
```bash
sudo apt-get update
sudo apt-get install -y python3-opencv
```

### Python Packages
The FastAPI server and related dependencies are bundled with the package. No additional pip installations required if using the provided `io_server` module.

## Camera Setup (Waveshare IMX219-160)

The Puzzlebot uses a Waveshare IMX219-160 CSI camera. Proper camera configuration is required for the GStreamer pipeline to work.

### Camera Configuration

1. **Download and extract camera overrides**:
   ```bash
   cd ~/Downloads
   wget https://files.waveshare.com/upload/e/eb/Camera_overrides.tar.gz
   tar -xvf Camera_overrides.tar.gz
   ```

2. **Install camera configuration**:
   ```bash
   sudo cp camera_overrides.isp /var/nvidia/nvcam/settings/
   sudo chmod 664 /var/nvidia/nvcam/settings/camera_overrides.isp
   sudo chown root:root /var/nvidia/nvcam/settings/camera_overrides.isp
   ```

3. **Reboot the Jetson**:
   ```bash
   sudo reboot
   ```

### Test Camera

Download and test the simple camera script from Waveshare:
```bash
cd ~/Downloads
wget https://www.waveshare.com/w/upload/e/e0/Simple_camera.zip
unzip Simple_camera.zip
python3 simple_camera.py
```

If the camera opens successfully, the GStreamer pipeline is working correctly.

**Reference**: [Waveshare IMX219-160 Camera Wiki](https://www.waveshare.com/wiki/IMX219-160_Camera)

## Building

```bash
cd ~/ros2_ws
colcon build --packages-select puzzlebot_server
source install/setup.bash
```

## Running

```bash
ros2 run puzzlebot_server puzzlebot_server
```

The server will start on port 5000. You should see output like:
```
[INFO] [puzzlebot_server]: Camera pipeline opened successfully
[INFO] [puzzlebot_server]: Puzzlebot Server node started
[INFO] [puzzlebot_server]: Web server running on port 5000
[INFO] [puzzlebot_server]: Camera stream: http://localhost:5000/stream/car_cam
[INFO] [puzzlebot_server]: Velocity control: http://localhost:5000/cmd_vel?x=0.5&w=0.2
```

## Usage

### Send velocity commands
```bash
# Move forward at 0.5 m/s
curl "http://localhost:5000/cmd_vel?x=0.5&w=0"

# Rotate in place
curl "http://localhost:5000/cmd_vel?x=0&w=1.0"

# Stop
curl "http://localhost:5000/cmd_vel?x=0&w=0"
```

### View camera stream
Open in browser: `http://<robot-ip>:5000/stream/car_cam`

## Adjusting Camera Settings

Edit camera resolution and framerate in `puzzlebot_server.py`, `_open_camera()` method:

```python
pipeline = self.gstreamer_pipeline(
    capture_width=1280,   # Sensor capture width
    capture_height=720,   # Sensor capture height
    display_width=1280,   # Output/stream width
    display_height=720,   # Output/stream height
    framerate=15,         # FPS
    flip_method=0         # Image rotation (0, 2 most common)
)
```

Supported sensor modes (IMX219):
- 3264 x 2464 @ 21 fps
- 3264 x 1848 @ 28 fps
- 1920 x 1080 @ 30 fps (16:9)
- 1640 x 1232 @ 30 fps
- 1280 x 720 @ 60 fps (16:9)
- 1280 x 720 @ 120 fps (16:9)

**Note**: Maintain 16:9 aspect ratio for CV algorithms (e.g., 1280x720, 640x360).

## Troubleshooting

### GStreamer TLS Error
If you see errors about `libGLdispatch.so.0: cannot allocate memory in static TLS block`, the ctypes preload fix should handle this automatically. This is already implemented in the code.

### Camera Not Found
- Verify camera connection with `ls /dev/video*`
- Check camera overrides are installed: `ls /var/nvidia/nvcam/settings/camera_overrides.isp`
- Restart nvargus daemon: `sudo service nvargus-daemon restart`
- Test with `simple_camera.py` script

### Port Already in Use
If port 5000 is in use, edit `io_server.py` and change the `port` variable.

## ROS Topics

- **Publishes**: `/cmd_vel_safe` (geometry_msgs/Twist) - Velocity commands to robot
- **Note**: Works with `accel_limiter` node for smooth acceleration
