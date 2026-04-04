import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped
import time

class AccelLimiterNode(Node):
    def __init__(self):
        super().__init__('accel_limiter')

        # Parameters for maximum velocity jumps
        self.declare_parameter('max_lin_jump', 0.1)
        self.declare_parameter('max_ang_jump', 0.5)
        self.max_lin_jump = self.get_parameter('max_lin_jump').value
        self.max_ang_jump = self.get_parameter('max_ang_jump').value
        self.add_on_set_parameters_callback(self._on_parameter_change)

        # State variables
        self.v_safe = 0.0   # desired linear velocity
        self.w_safe = 0.0   # desired angular velocity
        self.v_act = 0.0    # actual linear velocity
        self.w_act = 0.0    # actual angular velocity

        # Override logic
        self.override_active = False
        self._last_out_msg = Twist()
        self._last_out_time = time.monotonic()

        # Publisher to Hackerboard's cmd_vel
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Observe any /cmd_vel publications (to detect external publishers)
        self.create_subscription(
            Twist,
            '/cmd_vel',
            self._cmd_vel_observer,
            10
        )

        # Subscribe to safe commands from user
        self.create_subscription(
            Twist,
            '/cmd_vel_safe',
            self.safe_cmd_callback,
            10
        )
        # Subscribe to actual robot velocities
        self.create_subscription(
            TwistStamped,
            '/robot_vel',
            self.robot_vel_callback,
            qos_profile_sensor_data
        )

        # Timer for applying the slew-rate limiter at 20 Hz
        timer_period = 0.05  # seconds
        self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info(
            f'AccelLimiter started: max_lin_jump={self.max_lin_jump}, '
            f'max_ang_jump={self.max_ang_jump}'
        )

    def _on_parameter_change(self, params):
        for param in params:
            if param.name == 'max_lin_jump' and param.type_ == param.PARAMETER_DOUBLE:
                self.max_lin_jump = param.value
                self.get_logger().info(f'Updated max_lin_jump to {self.max_lin_jump}')
            if param.name == 'max_ang_jump' and param.type_ == param.PARAMETER_DOUBLE:
                self.max_ang_jump = param.value
                self.get_logger().info(f'Updated max_ang_jump to {self.max_ang_jump}')
        return SetParametersResult(successful=True)

    def _cmd_vel_observer(self, msg: Twist):
        """
        Detect external /cmd_vel publications. If a Twist arrives that
        is not our own recent publication, enter override mode.
        """
        now = time.monotonic()
        if self.override_active:
            return
        same = (
            msg.linear.x == self._last_out_msg.linear.x and
            msg.angular.z == self._last_out_msg.angular.z and
            (now - self._last_out_time) < 0.02
        )
        if same:
            return
        self.override_active = True
        self.get_logger().warn('External /cmd_vel detected → override mode')

    def safe_cmd_callback(self, msg: Twist):
        # Only reset override if safe command has changed
        old_v_safe = self.v_safe
        old_w_safe = self.w_safe
        self.v_safe = msg.linear.x
        self.w_safe = msg.angular.z
        if self.override_active and (self.v_safe != old_v_safe or self.w_safe != old_w_safe):
            self.override_active = False
            self.get_logger().info('New /cmd_vel_safe received → exiting override mode')

    def robot_vel_callback(self, msg: TwistStamped):
        self.v_act = msg.twist.linear.x
        self.w_act = msg.twist.angular.z

    def timer_callback(self):
        if self.override_active:
            return

        delta_v = self.v_safe - self.v_act
        delta_w = self.w_safe - self.w_act
        delta_v = max(-self.max_lin_jump, min(delta_v, self.max_lin_jump))
        delta_w = max(-self.max_ang_jump, min(delta_w, self.max_ang_jump))

        v_out = self.v_act + delta_v
        w_out = self.w_act + delta_w

        out_msg = Twist()
        out_msg.linear.x = v_out
        out_msg.angular.z = w_out
        self.cmd_pub.publish(out_msg)
        self._last_out_msg = out_msg
        self._last_out_time = time.monotonic()
        self.get_logger().debug(
            f'Limiter output → v_out: {v_out:.2f}, w_out: {w_out:.2f}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = AccelLimiterNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
