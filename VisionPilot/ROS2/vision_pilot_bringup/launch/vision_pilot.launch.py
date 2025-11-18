from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    steering_controller_node_h = Node(
        package='steering_controller',
        executable='steering_controller_node',
        name='steering_controller_node',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    longitudinal_controller_node_h = Node(
        package='longitudinal_controller',
        executable='longitudinal_controller_node',
        name='longitudinal_controller_node',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    pathfinder_node_h = Node(
        package='PATHFINDER',
        executable='pathfinder_node',
        name='pathfinder_node',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    tf2_ros_node_h = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='front_to_base_broadcaster',
        arguments=[
            "1.425", "0", "0",  # translation: x y z
            "0", "0", "0",  # rotation in rpy (roll pitch yaw in radians)
            "hero",  # parent frame
            "hero_front"  # child frame
        ]
    )

    return LaunchDescription([
        steering_controller_node_h,
        longitudinal_controller_node_h,
        pathfinder_node_h,
        tf2_ros_node_h,
    ])
