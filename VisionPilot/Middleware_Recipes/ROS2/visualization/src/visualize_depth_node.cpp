#include "visualize_depth_node.hpp"
#include "../../common/include/depth_visualization_engine.hpp"
#include <cv_bridge/cv_bridge.h>
#include <rclcpp_components/register_node_macro.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace autoware_pov::visualization
{

VisualizeDepthNode::VisualizeDepthNode(const rclcpp::NodeOptions & options)
: Node("visualize_depth_node", options)
{
  // Parameters
  const std::string depth_topic = this->declare_parameter<std::string>("depth_topic");
  const std::string output_topic = this->declare_parameter<std::string>("output_topic", "~/out/image");
  measure_latency_ = this->declare_parameter<bool>("measure_latency", true);

  // Create common depth visualization engine
  depth_viz_engine_ = std::make_unique<autoware_pov::common::DepthVisualizationEngine>();

  // Publisher
  pub_ = image_transport::create_publisher(this, output_topic);

  // Subscriber
  sub_ = image_transport::create_subscription(
    this, depth_topic, std::bind(&VisualizeDepthNode::onData, this, std::placeholders::_1), "raw");
}

void VisualizeDepthNode::onData(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
{
  // --- Latency Watcher Start ---
  if (measure_latency_ && (++frame_count_ % LATENCY_SAMPLE_INTERVAL == 0)) {
    viz_start_time_ = std::chrono::steady_clock::now();
  }
  // -----------------------------
  
  cv_bridge::CvImagePtr depth_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);

  // Use common depth visualization engine (framework-agnostic)
  cv::Mat colorized_depth = depth_viz_engine_->visualize(depth_ptr->image);

  sensor_msgs::msg::Image::SharedPtr out_msg =
    cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::BGR8, colorized_depth).toImageMsg();
  pub_.publish(out_msg);
  
  // --- Latency Watcher End & Report ---
  if (measure_latency_ && (frame_count_ % LATENCY_SAMPLE_INTERVAL == 0)) {
    auto viz_end_time = std::chrono::steady_clock::now();
    auto latency_ms =
      std::chrono::duration<double, std::milli>(viz_end_time - viz_start_time_)
        .count();
    RCLCPP_INFO(
      this->get_logger(), "Frame %zu: Depth Visualization Latency: %.2f ms (%.1f FPS)", frame_count_,
      latency_ms, 1000.0 / latency_ms);
  }
  // ------------------------------------
}

}  // namespace autoware_pov::visualization

RCLCPP_COMPONENTS_REGISTER_NODE(autoware_pov::visualization::VisualizeDepthNode)