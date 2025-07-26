#include "domain_seg_node.hpp"
#include <cv_bridge/cv_bridge.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>

namespace autoware_pov::AutoSeg::DomainSeg
{

DomainSegNode::DomainSegNode(const rclcpp::NodeOptions & node_options)
: Node("domain_seg_node", node_options)
{
  // Declare and get parameters
  const std::string model_path = this->declare_parameter<std::string>("model_path");
  const std::string backend = this->declare_parameter<std::string>("backend", "onnxruntime");
  const std::string precision = this->declare_parameter<std::string>("precision", "cpu");
  const int gpu_id = this->declare_parameter<int>("gpu_id", 0);
  measure_latency_ = this->declare_parameter<bool>("measure_latency", false);

  // Initialize the domain segmentation engine
  domain_seg_ = std::make_unique<DomainSeg>(model_path, backend, precision, gpu_id);
  
  // Setup publishers
  domain_pub_ = image_transport::create_publisher(this, "~/out/domain");
  color_domain_pub_ = image_transport::create_publisher(this, "~/out/color_domain");
  blended_pub_ = image_transport::create_publisher(this, "~/out/blended_vis");
  
  // Use a timer to defer subscriber creation until there's a connection
  using std::chrono_literals::operator""ms;
  timer_ = rclcpp::create_timer(this, get_clock(), 100ms, std::bind(&DomainSegNode::onConnect, this));
}

void DomainSegNode::onConnect()
{
  if (domain_pub_.getNumSubscribers() == 0 && color_domain_pub_.getNumSubscribers() == 0 && blended_pub_.getNumSubscribers() == 0) {
    image_sub_.shutdown();
  } else if (!image_sub_) {
    image_sub_ = image_transport::create_subscription(
      this, "~/in/image", std::bind(&DomainSegNode::onImage, this, std::placeholders::_1), "raw",
      rmw_qos_profile_sensor_data);
  }
}

void DomainSegNode::onImage(const sensor_msgs::msg::Image::ConstSharedPtr msg)
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
  if (!domain_seg_->doInference(in_image_ptr->image)) {
    RCLCPP_WARN(this->get_logger(), "Failed to run domain segmentation inference");
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
  const bool need_domain = domain_pub_.getNumSubscribers() > 0;
  const bool need_color_domain = color_domain_pub_.getNumSubscribers() > 0;
  const bool need_blended = blended_pub_.getNumSubscribers() > 0;

  if (need_domain || need_color_domain || need_blended) {
    cv::Mat domain_mask;
    domain_seg_->getDomainMask(domain_mask, in_image_ptr->image.size());

    if (need_domain) {
      // Publish raw domain mask as 32-bit float
      sensor_msgs::msg::Image::SharedPtr out_domain_msg =
        cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::TYPE_32FC1, domain_mask).toImageMsg();
      domain_pub_.publish(out_domain_msg);
    }
    
    if (need_color_domain) {
      // Generate colorized domain visualization
      cv::Mat colorized_domain;
      domain_seg_->colorizeDomain(domain_mask, colorized_domain);
      sensor_msgs::msg::Image::SharedPtr out_color_domain_msg =
        cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::BGR8, colorized_domain).toImageMsg();
      color_domain_pub_.publish(out_color_domain_msg);
    }
    
    if (need_blended) {
      // Generate weighted visualization (like Python video_visualization.py)
      cv::Mat blended_output;
      domain_seg_->createWeightedVisualization(in_image_ptr->image, domain_mask, blended_output, 0.5);
      sensor_msgs::msg::Image::SharedPtr out_blended_msg =
        cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::BGR8, blended_output).toImageMsg();
      blended_pub_.publish(out_blended_msg);
    }
  }
}

}  // namespace autoware_pov::AutoSeg::DomainSeg

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(autoware_pov::AutoSeg::DomainSeg::DomainSegNode) 