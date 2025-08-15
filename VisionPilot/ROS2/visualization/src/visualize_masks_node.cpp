#include "visualize_masks_node.hpp"
#include "../../common/include/masks_visualization_engine.hpp"
#include <cv_bridge/cv_bridge.h>
#include <rclcpp_components/register_node_macro.hpp>

namespace autoware_pov::visualization
{

VisualizeMasksNode::VisualizeMasksNode(const rclcpp::NodeOptions & options)
: Node("visualize_masks_node", options)
{
  // Parameters
  const std::string image_topic = this->declare_parameter<std::string>("image_topic");
  const std::string mask_topic = this->declare_parameter<std::string>("mask_topic");
  viz_type_ = this->declare_parameter<std::string>("viz_type", "scene");
  const std::string output_topic = this->declare_parameter<std::string>("output_topic", "~/out/image");
  measure_latency_ = this->declare_parameter<bool>("measure_latency", true);

  // Create common masks visualization engine
  viz_engine_ = std::make_unique<autoware_pov::common::MasksVisualizationEngine>(viz_type_);

  // Publisher
  pub_ = image_transport::create_publisher(this, output_topic);

  // Subscribers
  sub_image_.subscribe(this, image_topic, rmw_qos_profile_sensor_data);
  sub_mask_.subscribe(this, mask_topic, rmw_qos_profile_sensor_data);

  // Synchronizer
  sync_ = std::make_shared<Synchronizer>(SyncPolicy(10), sub_image_, sub_mask_);
  sync_->registerCallback(std::bind(&VisualizeMasksNode::onData, this, std::placeholders::_1, std::placeholders::_2));
}


void VisualizeMasksNode::onData(const Image::ConstSharedPtr& image_msg, const Image::ConstSharedPtr& mask_msg)
{
  // --- Latency Watcher Start ---
  if (measure_latency_ && (++frame_count_ % LATENCY_SAMPLE_INTERVAL == 0)) {
    viz_start_time_ = std::chrono::steady_clock::now();
  }
  // -----------------------------
  
  cv_bridge::CvImagePtr image_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
  cv_bridge::CvImagePtr mask_ptr = cv_bridge::toCvCopy(mask_msg, sensor_msgs::image_encodings::MONO8);

  // Use common visualization engine (framework-agnostic)
  cv::Mat blended_image = viz_engine_->visualize(mask_ptr->image, image_ptr->image);

  sensor_msgs::msg::Image::SharedPtr out_msg =
    cv_bridge::CvImage(image_msg->header, sensor_msgs::image_encodings::BGR8, blended_image).toImageMsg();
  pub_.publish(out_msg);
  
  // --- Latency Watcher End & Report ---
  if (measure_latency_ && (frame_count_ % LATENCY_SAMPLE_INTERVAL == 0)) {
    auto viz_end_time = std::chrono::steady_clock::now();
    auto latency_ms =
      std::chrono::duration<double, std::milli>(viz_end_time - viz_start_time_)
        .count();
    RCLCPP_INFO(
      this->get_logger(), "Frame %zu: Visualization Latency: %.2f ms (%.1f FPS)", frame_count_,
      latency_ms, 1000.0 / latency_ms);
  }
  // ------------------------------------
}

}  // namespace autoware_pov::visualization

RCLCPP_COMPONENTS_REGISTER_NODE(autoware_pov::visualization::VisualizeMasksNode)