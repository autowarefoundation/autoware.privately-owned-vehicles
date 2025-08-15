#ifndef VISUALIZE_MASKS_NODE_HPP_
#define VISUALIZE_MASKS_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include "../../common/include/masks_visualization_engine.hpp"


namespace autoware_pov::visualization
{

class VisualizeMasksNode : public rclcpp::Node
{
public:
  explicit VisualizeMasksNode(const rclcpp::NodeOptions & options);

private:
  using Image = sensor_msgs::msg::Image;
  using SyncPolicy = message_filters::sync_policies::ApproximateTime<Image, Image>;
  using Synchronizer = message_filters::Synchronizer<SyncPolicy>;

  void onData(const Image::ConstSharedPtr& image, const Image::ConstSharedPtr& mask);

  // Parameters
  std::string viz_type_;
  
  // Common masks visualization engine
  std::unique_ptr<autoware_pov::common::MasksVisualizationEngine> viz_engine_;

  // ROS
  image_transport::Publisher pub_;
  message_filters::Subscriber<Image> sub_image_;
  message_filters::Subscriber<Image> sub_mask_;
  std::shared_ptr<Synchronizer> sync_;
  
  // Latency monitoring (like original AUTOSEG)
  static constexpr size_t LATENCY_SAMPLE_INTERVAL = 100; // Log every 100 frames
  size_t frame_count_ = 0;
  bool measure_latency_ = false;
  std::chrono::steady_clock::time_point viz_start_time_;
};

}  // namespace autoware_pov::visualization

#endif  // VISUALIZE_MASKS_NODE_HPP_