#include "visualize_masks_node.hpp"
#include "../../common/include/masks_visualization_engine.hpp"
#include <cv_bridge/cv_bridge.h>
#include <rclcpp_components/register_node_macro.hpp>

namespace autoware_pov::visualization
{

VisualizeMasksNode::VisualizeMasksNode(const rclcpp::NodeOptions & options)
: Node("visualize_masks_node", options)
{
  // Parameters - need both image and mask topics
  const std::string image_topic = this->declare_parameter<std::string>("image_topic");
  const std::string mask_topic = this->declare_parameter<std::string>("mask_topic");
  viz_type_ = this->declare_parameter<std::string>("viz_type", "scene");
  const std::string output_topic = this->declare_parameter<std::string>("output_topic", "~/out/image");
  measure_latency_ = this->declare_parameter<bool>("measure_latency", true);

  // Create common masks visualization engine
  viz_engine_ = std::make_unique<autoware_pov::common::MasksVisualizationEngine>(viz_type_);

  // Publisher
  pub_ = image_transport::create_publisher(this, output_topic);

  // Simple dual subscription - cache latest image, process on mask arrival
  sub_image_ = image_transport::create_subscription(
    this, image_topic, std::bind(&VisualizeMasksNode::onImageData, this, std::placeholders::_1), "raw");
  sub_mask_ = image_transport::create_subscription(
    this, mask_topic, std::bind(&VisualizeMasksNode::onMaskData, this, std::placeholders::_1), "raw");
}


void VisualizeMasksNode::onImageData(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
{
  // Simply cache the latest image - no processing here
  cv_bridge::CvImagePtr image_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  latest_image_ = image_ptr->image.clone();
}

void VisualizeMasksNode::onMaskData(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
{
  // Only process when we have a cached image
  if (latest_image_.empty()) {
    return;  // Wait for first image to arrive
  }

  // --- Latency Watcher Start ---
  if (measure_latency_ && (++frame_count_ % LATENCY_SAMPLE_INTERVAL == 0)) {
    viz_start_time_ = std::chrono::steady_clock::now();
  }
  // -----------------------------
  
  // Convert incoming mask message (back to simple approach)
  cv_bridge::CvImagePtr mask_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);

  // Simple mask visualization with blending
  cv::Mat blended_result = viz_engine_->visualize(mask_ptr->image, latest_image_);

  sensor_msgs::msg::Image::SharedPtr out_msg =
    cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::BGR8, blended_result).toImageMsg();
  pub_.publish(out_msg);
  
  // --- Latency Watcher End & Report ---
  if (measure_latency_ && (frame_count_ % LATENCY_SAMPLE_INTERVAL == 0)) {
    auto viz_end_time = std::chrono::steady_clock::now();
    auto latency_ms =
      std::chrono::duration<double, std::milli>(viz_end_time - viz_start_time_)
        .count();
    RCLCPP_INFO(
      this->get_logger(), "Frame %zu: Visualization Latency: %.2f ms (%.1f FPS)", 
      frame_count_, latency_ms, 1000.0 / latency_ms);
  }
  // ------------------------------------
}

}  // namespace autoware_pov::visualization

RCLCPP_COMPONENTS_REGISTER_NODE(autoware_pov::visualization::VisualizeMasksNode)