import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    pkg_dir = get_package_share_directory("autoware_pov_scene3d")
    
    # --- Declare Launch Arguments ---
    declare_model_path_arg = DeclareLaunchArgument(
        "model_path",
        default_value=os.path.join(pkg_dir, "models", "scene3d.onnx"),
        description="Path to the ONNX model file for depth estimation."
    )
    
    declare_backend_arg = DeclareLaunchArgument(
        "backend",
        default_value="onnxruntime",
        description="The backend to use for inference (e.g., 'onnxruntime', 'tensorrt')."
    )
    
    declare_precision_arg = DeclareLaunchArgument(
        "precision",
        default_value="cpu",
        description="The execution provider/precision to use for inference (e.g., 'cpu', 'cuda', 'fp32', 'fp16')."
    )

    declare_gpu_id_arg = DeclareLaunchArgument(
        "gpu_id",
        default_value="0",
        description="The GPU device ID to use for inference."
    )

    declare_measure_latency_arg = DeclareLaunchArgument(
        "measure_latency",
        default_value="false",
        description="Whether to measure and log inference latency."
    )

    # Add a new launch argument to make the input topic configurable
    declare_input_topic_arg = DeclareLaunchArgument(
        "input_image_topic",
        default_value="/image_raw",
        description="The topic for the input images."
    )

    # --- Define the Composable Node ---
    scene3d_node = ComposableNode(
        package="autoware_pov_scene3d",
        plugin="autoware_pov::AutoSeg::Scene3D::Scene3DNode",
        name="scene3d_node",
        parameters=[{
            "model_path": LaunchConfiguration("model_path"),
            "backend": LaunchConfiguration("backend"),
            "precision": LaunchConfiguration("precision"),
            "gpu_id": LaunchConfiguration("gpu_id"),
            "measure_latency": LaunchConfiguration("measure_latency"),
        }],
        # --- Remap input/output topics using the launch argument ---
        remappings=[
            ("~/in/image", LaunchConfiguration("input_image_topic")),
        ]
    )

    # --- Define the Node Container ---
    # This is the modern and efficient way to run nodes in ROS2
    container = ComposableNodeContainer(
        name="scene3d_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[scene3d_node],
        output="screen",
    )

    return LaunchDescription([
        declare_model_path_arg,
        declare_backend_arg,
        declare_precision_arg,
        declare_gpu_id_arg,
        declare_measure_latency_arg,
        declare_input_topic_arg,
        container
    ]) 