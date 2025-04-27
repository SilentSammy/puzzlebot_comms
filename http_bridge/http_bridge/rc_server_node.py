import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
import cv2
import math
import json

# import the web framework
from http_bridge import web

class RCServerNode(Node):
    def __init__(self):
        super().__init__('rc_server_node')

        # Publisher for forwarding to ROS /cmd_vel
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)

        # Store the last received velocities
        self.v = 0.0
        self.w = 0.0

        # HTTP endpoint callback for velocity commands
        def receive_vel(request):
            # Parse query params, preserving old values if missing
            new_v = request.args.get('v', type=float)
            new_w = request.args.get('w', type=float)
            self.v = new_v if new_v is not None else self.v
            self.w = new_w if new_w is not None else self.w

            # Publish to ROS topic
            msg = Twist()
            msg.linear.x = self.v
            msg.angular.z = self.w
            self.publisher_.publish(msg)
            self.get_logger().info(
                f'HTTP update -> published Twist(linear.x={self.v}, angular.z={self.w})'
            )

            # Echo back
            return f"Linear Velocity: {self.v}, Angular Velocity: {self.w}", 200

        web.http_endpoints['cmd_vel'] = receive_vel

        # -- Subscribe to estimated pose --
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.create_subscription(
            PoseStamped,
            '/estimated_pose',
            self.pose_callback,
            10
        )

        # HTTP endpoint to return latest pose as JSON
        def get_pose(request):
            publishers = self.get_publishers_info_by_topic('/estimated_pose')
            if not publishers:
                # No pose available or source node is down
                return (json.dumps({}), 200, {'Content-Type': 'application/json'})
            data = {'x': self.x, 'y': self.y, 'theta': self.theta}
            return (json.dumps(data), 200, {'Content-Type': 'application/json'})

        web.http_endpoints['pose'] = get_pose

        # --- Camera setup using GStreamer for CSI camera on Jetson ---
        gst_pipeline = (
            'nvarguscamerasrc ! '
            'video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! '
            'nvvidconv flip-method=0 ! '
            'video/x-raw,format=BGRx ! '
            'videoconvert ! '
            'video/x-raw,format=BGR ! '
            'appsink'
        )
        self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            self.get_logger().error("Could not open CSI camera via GStreamer pipeline!")

        def video_source():
            ret, frame = self.cap.read()
            return frame if ret else None

        web.video_endpoints['car_cam'] = video_source

        # Start the HTTP server in background thread
        web.start_webserver(threaded=True)
        self.get_logger().info('HTTP server started on port 5000')

    def pose_callback(self, msg: PoseStamped):
        # Update stored pose values from the PoseStamped message
        self.x = msg.pose.position.x
        self.y = msg.pose.position.y
        z = msg.pose.orientation.z
        w = msg.pose.orientation.w
        self.theta = 2 * math.atan2(z, w)

def main(args=None):
    rclpy.init(args=args)
    node = RCServerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Release camera resource and shutdown
        node.cap.release()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
