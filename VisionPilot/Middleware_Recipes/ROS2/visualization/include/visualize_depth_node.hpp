#ifndef VISUALIZE_DEPTH_NODE_HPP_
#define VISUALIZE_DEPTH_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <chrono>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "../../common/include/depth_visualization_engine.hpp"

namespace autoware_pov::visualization
{

class VisualizeDepthNode : public rclcpp::Node
{
public:
  explicit VisualizeDepthNode(const rclcpp::NodeOptions & options);

private:
  void onData(const sensor_msgs::msg::Image::ConstSharedPtr& msg);

  // ROS
  image_transport::Subscriber sub_;
  image_transport::Publisher pub_;
  
  // Common depth visualization engine
  std::unique_ptr<autoware_pov::common::DepthVisualizationEngine> depth_viz_engine_;
  
  // Latency monitoring (like original AUTOSEG)
  static constexpr size_t LATENCY_SAMPLE_INTERVAL = 100; // Log every 100 frames
  size_t frame_count_ = 0;
  bool measure_latency_ = false;
  std::chrono::steady_clock::time_point viz_start_time_;
};

}  // namespace autoware_pov::visualization

#endif  // VISUALIZE_DEPTH_NODE_HPP_