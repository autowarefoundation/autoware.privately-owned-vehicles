#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import sys
import time

class VideoPublisher(Node):
    def __init__(self, video_path, topic_name, frame_rate):
        super().__init__('video_publisher_node')
        
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Image, topic_name, 10)
        
        # Open video file
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open video file: {video_path}")
            sys.exit(1)
            
        # Calculate timer period for desired frame rate
        timer_period = 1.0 / frame_rate
        self.timer = self.create_timer(timer_period, self.publish_frame)
        
        self.get_logger().info(f"Publishing video from '{video_path}' to '{topic_name}' at {frame_rate} FPS.")
        
    def publish_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert BGR to RGB for ROS
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to ROS Image message
            img_msg = self.bridge.cv2_to_imgmsg(frame_rgb, encoding='rgb8')
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = 'camera'
            
            self.publisher.publish(img_msg)
        else:
            # Loop video
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

def main(args=None):
    if len(sys.argv) < 4:
        print("Usage: video_publisher.py <video_path> <topic_name> <frame_rate>")
        sys.exit(1)
        
    video_path = sys.argv[1]
    topic_name = sys.argv[2] 
    frame_rate = float(sys.argv[3])
    
    rclpy.init(args=args)
    
    node = VideoPublisher(video_path, topic_name, frame_rate)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 