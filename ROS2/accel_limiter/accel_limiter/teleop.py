#!/usr/bin/env python3
import curses
import threading
import time
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import Twist

class TeleopNode(Node):
    def __init__(self):
        super().__init__('teleop_keyboard')
        self.pub = self.create_publisher(Twist, '/cmd_vel_safe', 10)
        self.linear = 0.0
        self.angular = 0.0

        # Declare parameters
        self.declare_parameter('linear_speed_normal', 0.2)
        self.declare_parameter('linear_speed_fast', 0.5)
        self.declare_parameter('angular_speed_normal', 1.0)
        self.declare_parameter('angular_speed_fast', 2.0)

        # Initialize speed values from parameters
        self.linear_speed_normal = self.get_parameter('linear_speed_normal').value
        self.linear_speed_fast = self.get_parameter('linear_speed_fast').value
        self.angular_speed_normal = self.get_parameter('angular_speed_normal').value
        self.angular_speed_fast = self.get_parameter('angular_speed_fast').value

        # Add parameter callback to update speeds at runtime.
        self.add_on_set_parameters_callback(self._param_callback)

        # Start the keyboard loop in a separate thread
        thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        thread.start()
        self.get_logger().info('Teleop keyboard node started. Use WASD to drive and IJKL to drive faster, X to stop.')

    def _param_callback(self, params):
        result = SetParametersResult()
        result.successful = True
        for param in params:
            if param.name == "linear_speed_normal":
                self.linear_speed_normal = param.value
                self.get_logger().info(f"Updated linear_speed_normal to {param.value}")
            elif param.name == "linear_speed_fast":
                self.linear_speed_fast = param.value
                self.get_logger().info(f"Updated linear_speed_fast to {param.value}")
            elif param.name == "angular_speed_normal":
                self.angular_speed_normal = param.value
                self.get_logger().info(f"Updated angular_speed_normal to {param.value}")
            elif param.name == "angular_speed_fast":
                self.angular_speed_fast = param.value
                self.get_logger().info(f"Updated angular_speed_fast to {param.value}")
        return result

    def _keyboard_loop(self):
        stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        stdscr.nodelay(True)
        try:
            while rclpy.ok():
                key = stdscr.getch()
                # Update linear speed based on parameters
                if key == ord('w'):
                    self.linear = self.linear_speed_normal
                elif key == ord('s'):
                    self.linear = -self.linear_speed_normal
                elif key == ord('i'):
                    self.linear = self.linear_speed_fast
                elif key == ord('k'):
                    self.linear = -self.linear_speed_fast
                # Update angular speed based on parameters
                if key == ord('a'):
                    self.angular = self.angular_speed_normal
                elif key == ord('d'):
                    self.angular = -self.angular_speed_normal
                elif key == ord('j'):
                    self.angular = self.angular_speed_fast
                elif key == ord('l'):
                    self.angular = -self.angular_speed_fast
                # Stop both on 'x'
                if key == ord('x'):
                    self.linear = 0.0
                    self.angular = 0.0

                msg = Twist()
                msg.linear.x = self.linear
                msg.angular.z = self.angular
                self.pub.publish(msg)

                time.sleep(0.05)  # 20 Hz
        finally:
            curses.nocbreak()
            curses.echo()
            curses.endwin()


def main(args=None):
    rclpy.init(args=args)
    node = TeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()