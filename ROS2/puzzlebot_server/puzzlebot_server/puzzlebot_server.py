#!/usr/bin/env python3
# Fix for GStreamer TLS issue on Jetson - must be FIRST, before cv2 import
import ctypes
import sys

# Preload the problematic library with RTLD_GLOBAL flag
try:
    ctypes.CDLL('/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0', mode=ctypes.RTLD_GLOBAL)
except Exception as e:
    print(f"Warning: Could not preload libGLdispatch.so.0: {e}", file=sys.stderr)

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from fastapi import Request
import cv2
import threading

# Import the io_server components
from . import io_server


class PuzzlebotServer(Node):
    def __init__(self):
        super().__init__('puzzlebot_server')
        
        # State variables
        self.current_x = 0.0
        self.current_w = 0.0
        
        # Camera setup using working JetsonHacks pipeline format
        self.cap = None
        self.cap_lock = threading.Lock()
        self._open_camera()
        
        # Publisher for velocity commands
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel_safe', 10)
        
        # Register HTTP endpoints
        io_server.http_endpoints['cmd_vel'] = self.handle_cmd_vel
        
        # Register stream endpoints
        io_server.stream_endpoints['car_cam'] = self.get_camera_frame
        
        # Start web server in background thread
        io_server.start_webserver(threaded=True)
        
        self.get_logger().info('Puzzlebot Server node started')
        self.get_logger().info(f'Web server running on port {io_server.port}')
        self.get_logger().info(f'Camera stream: http://localhost:{io_server.port}/stream/car_cam')
        self.get_logger().info(f'Velocity control: http://localhost:{io_server.port}/cmd_vel?x=0.5&w=0.2')
    
    def gstreamer_pipeline(
        self,
        sensor_id=0,
        capture_width=1920,
        capture_height=1080,
        display_width=960,
        display_height=540,
        framerate=30,
        flip_method=0,
    ):
        """GStreamer pipeline for CSI camera - JetsonHacks format"""
        return (
            "nvarguscamerasrc sensor-id=%d !"
            "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                sensor_id,
                capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,
            )
        )
    
    def _open_camera(self):
        """Open camera using GStreamer pipeline"""
        try:
            pipeline = self.gstreamer_pipeline(
                capture_width=640,
                capture_height=360,
                display_width=640,
                display_height=360,
                framerate=15,
                flip_method=0
            )
            self.get_logger().info(f'Opening camera with pipeline: {pipeline}')
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if not cap.isOpened():
                raise RuntimeError('GStreamer pipeline failed to open')
            self.cap = cap
            self.get_logger().info('Camera pipeline opened successfully')
        except Exception as e:
            self.cap = None
            self.get_logger().error(f'Could not open camera pipeline: {e}')
    
    def get_camera_frame(self):
        """Stream endpoint: returns latest camera frame"""
        with self.cap_lock:
            if self.cap is None or not self.cap.isOpened():
                self.get_logger().warning('Re-opening camera stream')
                self._open_camera()
            try:
                ret, frame = self.cap.read()
                if not ret:
                    raise RuntimeError('Failed to grab frame')
                return frame
            except Exception as e:
                self.get_logger().error(f'Camera read error: {e}')
                # Release and nullify cap so next call reopens
                try:
                    self.cap.release()
                except:
                    pass
                self.cap = None
                return None
    
    def handle_cmd_vel(self, request: Request):
        """HTTP endpoint: receive velocity commands and publish to ROS"""
        params = dict(request.query_params)
        
        # Update velocities from query parameters
        self.current_x = float(params.get('x', self.current_x))
        self.current_w = float(params.get('w', self.current_w))
        
        # Publish to ROS
        msg = Twist()
        msg.linear.x = self.current_x
        msg.angular.z = self.current_w
        self.vel_pub.publish(msg)
        
        self.get_logger().info(f'Velocity command: x={self.current_x}, w={self.current_w}')
        
        # Return current velocities as JSON
        return {"x": self.current_x, "w": self.current_w}


def main(args=None):
    rclpy.init(args=args)
    node = PuzzlebotServer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up camera
        with node.cap_lock:
            if node.cap:
                node.cap.release()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
