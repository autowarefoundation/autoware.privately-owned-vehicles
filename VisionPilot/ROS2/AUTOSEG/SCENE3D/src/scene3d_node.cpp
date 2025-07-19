#include "scene3d_node.hpp"
#include <cv_bridge/cv_bridge.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>

namespace autoware_pov::AutoSeg::Scene3D
{

Scene3DNode::Scene3DNode(const rclcpp::NodeOptions & node_options)
: Node("scene3d_node", node_options)
{
  // Declare and get parameters
  const std::string model_path = this->declare_parameter<std::string>("model_path");
  const std::string backend = this->declare_parameter<std::string>("backend", "onnxruntime");
  const std::string precision = this->declare_parameter<std::string>("precision", "cpu");
  const int gpu_id = this->declare_parameter<int>("gpu_id", 0);
  measure_latency_ = this->declare_parameter<bool>("measure_latency", false);

  // Initialize the depth estimation engine
  scene3d_ = std::make_unique<Scene3D>(model_path, backend, precision, gpu_id);
  
  // Setup publishers
  depth_pub_ = image_transport::create_publisher(this, "~/out/depth");
  color_depth_pub_ = image_transport::create_publisher(this, "~/out/color_depth");
  
  // Use a timer to defer subscriber creation until there's a connection
  using std::chrono_literals::operator""ms;
  timer_ = rclcpp::create_timer(this, get_clock(), 100ms, std::bind(&Scene3DNode::onConnect, this));
}

void Scene3DNode::onConnect()
{
  if (depth_pub_.getNumSubscribers() == 0 && color_depth_pub_.getNumSubscribers() == 0) {
    image_sub_.shutdown();
  } else if (!image_sub_) {
    image_sub_ = image_transport::create_subscription(
      this, "~/in/image", std::bind(&Scene3DNode::onImage, this, std::placeholders::_1), "raw",
      rmw_qos_profile_sensor_data);
  }
}

void Scene3DNode::onImage(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
  cv_bridge::CvImagePtr in_image_ptr;
  try {
    in_image_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  } catch (const cv_bridge::Exception & e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }

  // --- Latency Watcher Start ---
  if (measure_latency_ && (++frame_count_ % LATENCY_SAMPLE_INTERVAL == 0)) {
    inference_start_time_ = std::chrono::steady_clock::now();
  }
  // -----------------------------

  // Run inference
  if (!scene3d_->doInference(in_image_ptr->image)) {
    RCLCPP_WARN(this->get_logger(), "Failed to run depth estimation inference");
    return;
  }

  // --- Latency Watcher End & Report ---
  if (measure_latency_ && (frame_count_ % LATENCY_SAMPLE_INTERVAL == 0)) {
    auto inference_end_time = std::chrono::steady_clock::now();
    auto latency_ms =
      std::chrono::duration<double, std::milli>(inference_end_time - inference_start_time_)
        .count();
    RCLCPP_INFO(
      this->get_logger(), "Frame %zu: Inference Latency: %.2f ms (%.1f FPS)", frame_count_,
      latency_ms, 1000.0 / latency_ms);
  }
  // ------------------------------------
  
  // Decide whether we need to generate any outputs
  const bool need_depth = depth_pub_.getNumSubscribers() > 0;
  const bool need_color_depth = color_depth_pub_.getNumSubscribers() > 0;

  if (need_depth || need_color_depth) {
    cv::Mat depth_map;
    scene3d_->getDepthMap(depth_map, in_image_ptr->image.size());

    if (need_depth) {
      // Publish raw depth as 32-bit float
      sensor_msgs::msg::Image::SharedPtr out_depth_msg =
        cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::TYPE_32FC1, depth_map).toImageMsg();
      depth_pub_.publish(out_depth_msg);
    }
    
    if (need_color_depth) {
      // Generate colorized depth visualization
      cv::Mat colorized_depth;
      scene3d_->colorizeDepth(depth_map, colorized_depth);
      sensor_msgs::msg::Image::SharedPtr out_color_depth_msg =
        cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::BGR8, colorized_depth).toImageMsg();
      color_depth_pub_.publish(out_color_depth_msg);
    }
  }
}

}  // namespace autoware_pov::AutoSeg::Scene3D

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(autoware_pov::AutoSeg::Scene3D::Scene3DNode) 