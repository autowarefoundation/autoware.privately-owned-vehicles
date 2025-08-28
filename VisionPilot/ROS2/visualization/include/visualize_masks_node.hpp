#ifndef VISUALIZE_MASKS_NODE_HPP_
#define VISUALIZE_MASKS_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "../../common/include/masks_visualization_engine.hpp"


namespace autoware_pov::visualization
{

class VisualizeMasksNode : public rclcpp::Node
{
public:
  explicit VisualizeMasksNode(const rclcpp::NodeOptions & options);

private:
  void onImageData(const sensor_msgs::msg::Image::ConstSharedPtr& msg);
  void onMaskData(const sensor_msgs::msg::Image::ConstSharedPtr& msg);

  // Parameters
  std::string viz_type_;
  
  // Common masks visualization engine
  std::unique_ptr<autoware_pov::common::MasksVisualizationEngine> viz_engine_;

  // ROS - Simple dual subscription with caching (no synchronizer!)
  image_transport::Publisher pub_;
  image_transport::Subscriber sub_mask_;
  image_transport::Subscriber sub_image_;
  
  // Cache latest image (simple approach, no timestamp matching)
  cv::Mat latest_image_;
  
  // Latency monitoring (like original AUTOSEG)
  static constexpr size_t LATENCY_SAMPLE_INTERVAL = 100; // Log every 100 frames
  size_t frame_count_ = 0;
  bool measure_latency_ = false;
  std::chrono::steady_clock::time_point viz_start_time_;
};

}  // namespace autoware_pov::visualization

#endif  // VISUALIZE_MASKS_NODE_HPP_