import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, TwistStamped
import cv2
import math
import json
import threading
import time
from rclpy.qos import qos_profile_sensor_data
from pb_http_server import web

class RCServerNode(Node):
    def __init__(self):
        super().__init__('rc_server_node')

        # CMD_VEL HTTP ENDPOINT SETUP
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.v = 0.0
        self.w = 0.0
        def receive_vel(request):
            new_v = request.args.get('v', type=float)
            new_w = request.args.get('w', type=float)
            self.v = new_v if new_v is not None else self.v
            self.w = new_w if new_w is not None else self.w
            msg = Twist()
            msg.linear.x = self.v
            msg.angular.z = self.w
            self.publisher_.publish(msg)
            self.get_logger().info(
                f'HTTP update -> published Twist(linear.x={self.v}, angular.z={self.w})'
            )
            return json.dumps({'v': self.v, 'w': self.w}), 200, {'Content-Type': 'application/json'}
        web.http_endpoints['cmd_vel'] = receive_vel


        # --- cmd_vel_safe HTTP endpoint ---
        self.safe_pub = self.create_publisher(Twist, '/cmd_vel_safe', 10)
        self.v_safe = 0.0
        self.w_safe = 0.0
        def receive_safe(request):
            # Parse query params, preserving old values if missing
            new_v = request.args.get('v', type=float)
            new_w = request.args.get('w', type=float)
            self.v_safe = new_v if new_v is not None else self.v_safe
            self.w_safe = new_w if new_w is not None else self.w_safe
            # Publish safe command
            msg = Twist()
            msg.linear.x = self.v_safe
            msg.angular.z = self.w_safe
            self.safe_pub.publish(msg)
            self.get_logger().info(
                f'HTTP safe update -> published safe Twist(linear.x={self.v_safe}, angular.z={self.w_safe})'
            )
            # Return JSON response
            return (json.dumps({'v': self.v_safe, 'w': self.w_safe}),
                    200, {'Content-Type': 'application/json'})

        web.http_endpoints['cmd_vel_safe'] = receive_safe

        # STATE HTTP ENDPOINT SETUP
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.robot_v = 0.0
        self.robot_w = 0.0
        def pose_callback(msg: PoseStamped):
            self.x = msg.pose.position.x
            self.y = msg.pose.position.y
            z = msg.pose.orientation.z
            w = msg.pose.orientation.w
            self.theta = 2 * math.atan2(z, w)
        self.create_subscription(PoseStamped, '/estimated_pose', pose_callback, 10)
        def robot_vel_callback(msg: TwistStamped):
            self.robot_v = msg.twist.linear.x
            self.robot_w = msg.twist.angular.z
        self.create_subscription(TwistStamped,'/robot_vel',robot_vel_callback,qos_profile_sensor_data)
        def get_state(request):
            data = { 'v': self.robot_v, 'w': self.robot_w }
            if self.get_publishers_info_by_topic('/estimated_pose'): # Append pose data if topic exists
                data['x'] = self.x
                data['y'] = self.y
                data['theta'] = self.theta
            return (json.dumps(data), 200, {'Content-Type': 'application/json'})
        web.http_endpoints['state'] = get_state

        # CAR_CAM MJPEG STREAM SETUP
        self.gst_pipeline = (
            'nvarguscamerasrc ! '
            'video/x-raw(memory:NVMM),width=640,height=480,framerate=15/1 ! '
            'nvvidconv flip-method=0 ! '
            'video/x-raw,format=BGRx ! '
            'videoconvert ! '
            'video/x-raw,format=BGR ! '
            # appsink caps to drop late frames and keep latency low
            'appsink drop=true max-buffers=1'
        )
        self.cap = None
        self.cap_lock = threading.Lock()
        self._open_camera()
        def video_source():
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
        web.video_endpoints['car_cam'] = video_source

        # Start the HTTP server in background
        web.start_webserver(threaded=True)
        self.get_logger().info('HTTP server started on port 5000')

    def _open_camera(self):
        # Attempt to open the camera pipeline
        try:
            cap = cv2.VideoCapture(self.gst_pipeline, cv2.CAP_GSTREAMER)
            if not cap.isOpened():
                raise RuntimeError('GStreamer pipeline failed to open')
            self.cap = cap
            self.get_logger().info('Camera pipeline opened successfully')
        except Exception as e:
            self.cap = None
            self.get_logger().error(f'Could not open camera pipeline: {e}')

    def pose_callback(self, msg: PoseStamped):
        self.x = msg.pose.position.x
        self.y = msg.pose.position.y
        z = msg.pose.orientation.z
        w = msg.pose.orientation.w
        self.theta = 2 * math.atan2(z, w)
        self.pose_received = True

    def robot_vel_callback(self, msg: TwistStamped):
        self.robot_v = msg.twist.linear.x
        self.robot_w = msg.twist.angular.z

def main(args=None):
    rclpy.init(args=args)
    node = RCServerNode()
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
