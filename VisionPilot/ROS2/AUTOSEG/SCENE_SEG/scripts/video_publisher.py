#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import sys

class VideoPublisherNode(Node):
    def __init__(self, video_path, topic_name, frame_rate):
        super().__init__('video_publisher_node')
        self.publisher_ = self.create_publisher(Image, topic_name, 10)
        self.bridge = CvBridge()
        self.video_capture = cv2.VideoCapture(video_path)
        
        if not self.video_capture.isOpened():
            self.get_logger().error(f"Failed to open video file: {video_path}")
            sys.exit(1)
            
        self.timer = self.create_timer(1.0 / frame_rate, self.timer_callback)
        self.get_logger().info(f"Publishing video from '{video_path}' to '{topic_name}' at {frame_rate} FPS.")

    def timer_callback(self):
        ret, frame = self.video_capture.read()
        if ret:
            # Convert OpenCV image to ROS Image message and publish
            ros_image_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            ros_image_msg.header.stamp = self.get_clock().now().to_msg()
            self.publisher_.publish(ros_image_msg)
        else:
            self.get_logger().info("End of video file reached. Looping...")
            # Loop the video
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

def main(args=None):
    rclpy.init(args=args)
    
    # --- Get parameters from command line or defaults ---
    # This is a simple way to get args for a standalone script.
    # A more robust solution would use declare_parameter if this were a Component.
    video_path_arg = sys.argv[1] if len(sys.argv) > 1 else 'input.mp4'
    topic_name_arg = sys.argv[2] if len(sys.argv) > 2 else '/in/image'
    frame_rate_arg = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0

    video_publisher_node = VideoPublisherNode(video_path_arg, topic_name_arg, frame_rate_arg)
    
    try:
        rclpy.spin(video_publisher_node)
    except KeyboardInterrupt:
        pass
    finally:
        video_publisher_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 