from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='pb_http_server',
            executable='pose_estimator',
            name='pose_estimator',
            output='screen'
        ),
        Node(
            package='pb_http_server',
            executable='pb_http_server',
            name='pb_http_server',
            output='screen'
        ),
    ])
