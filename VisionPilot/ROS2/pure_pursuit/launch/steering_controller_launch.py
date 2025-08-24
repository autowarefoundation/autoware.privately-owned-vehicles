from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='pure_pursuit',  
            executable='pure_pursuit_node',
            name='pure_pursuit_node',
            output='screen'
        ),
        
        Node(
            package='road_shape_publisher',
            executable='road_shape_publisher_node',
            name='road_shape_publisher_node',
            output='screen'
        ),
        
        Node(
            package='PATHFINDER',
            executable='pathfinder_node',
            name='pathfinder_node',
            output='screen'
        ),

    ])