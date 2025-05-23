#!/usr/bin/env python3

import threading
import cv2
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from cam_navigation import visual_navigation as vn

def navigate_stub(frame, drawing_frame=None):
    # Example: draw the frame’s average brightness as text
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()
    if drawing_frame is not None:    
        cv2.putText(drawing_frame, f"Bright: {brightness:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    print(f"[navigate] Frame brightness: {brightness:.2f}")
    return 0.0, 0.0

class CamNavigationNode(Node):
    def __init__(self):
        super().__init__('cam_navigation')
        self.get_logger().info("Starting CamNavigationNode")

        # Publishers
        self.cmd_pub   = self.create_publisher(Twist, '/cmd_vel_safe',     10)
        self.image_pub = self.create_publisher(Image, '/processed_frame', 10)

        # CvBridge for converting OpenCV → ROS Image
        self.bridge = CvBridge()

        # GStreamer pipeline…
        self._gst_pipeline = (
            'nvarguscamerasrc ! '
            'video/x-raw(memory:NVMM),width=640,height=480,framerate=15/1 ! '
            'nvvidconv flip-method=0 ! '
            'video/x-raw,format=BGRx ! '
            'videoconvert ! '
            'video/x-raw,format=BGR ! '
            'appsink drop=true max-buffers=1'
        )
        self._cap = None
        self._cap_lock = threading.Lock()
        self._open_camera()

        # 15 Hz timer
        self.create_timer(1.0 / 15.0, self._on_timer)

    def _open_camera(self):
        try:
            cap = cv2.VideoCapture(self._gst_pipeline, cv2.CAP_GSTREAMER)
            if not cap.isOpened():
                raise RuntimeError("Pipeline failed to open")
            self._cap = cap
            self.get_logger().info("Camera opened successfully")
        except Exception as e:
            self._cap = None
            self.get_logger().error(f"Could not open camera: {e}")

    def _read_frame(self):
        with self._cap_lock:
            if self._cap is None or not self._cap.isOpened():
                self.get_logger().warning("Camera not open")
                return None
            ret, frame = self._cap.read()
        if not ret or frame is None:
            self.get_logger().warning("Failed to grab frame")
            return None
        return frame

    def _on_timer(self):
        # 1) grab raw frame
        frame = self._read_frame()
        if frame is None:
            return

        # 2) make a drawing copy
        drawing_frame = frame.copy()

        # 3) your nav logic now takes BOTH
        throttle, yaw = self.navigate(frame, drawing_frame)

        # 4) publish the annotated frame
        img_msg = self.bridge.cv2_to_imgmsg(drawing_frame, encoding="bgr8")
        img_msg.header.stamp = self.get_clock().now().to_msg()
        self.image_pub.publish(img_msg)

        # 5) publish motion command
        twist = Twist()
        twist.linear.x  = float(throttle)
        twist.angular.z = float(yaw)
        self.cmd_pub.publish(twist)

    def navigate(self, frame, drawing_frame=None):
        """
        Placeholder navigation function.

        Args:
            frame         (numpy.ndarray): original BGR image
            drawing_frame (numpy.ndarray): copy you can draw on

        Returns:
            throttle (float): forward speed [m/s]
            yaw      (float): turning rate [rad/s]
        """

        # thr, yaw = navigate_stub(frame, drawing_frame)
        thr, yaw = vn.follow_line(frame, drawing_frame=drawing_frame)
        
        return thr, yaw

    def destroy_node(self):
        # Stop the robot by sending zero velocity
        stop_twist = Twist()
        stop_twist.linear.x = 0.0
        stop_twist.angular.z = 0.0
        self.cmd_pub.publish(stop_twist)
        self.get_logger().info("Sent stop command")

        # Release the camera
        with self._cap_lock:
            if self._cap is not None and self._cap.isOpened():
                self._cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CamNavigationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
