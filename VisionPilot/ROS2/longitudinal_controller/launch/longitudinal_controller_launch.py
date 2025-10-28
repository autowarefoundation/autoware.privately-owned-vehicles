from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='longitudinal_controller',  
            executable='longitudinal_controller_node',
            name='longitudinal_controller_node',
            output='screen'
        ),
        
        Node(
            package='odom_publisher',
            executable='pub_odom_node',
            name='pub_odom_node',
            output='screen'
        )

    ])