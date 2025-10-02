#include "run_autospeed_node.hpp"
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <cmath>

namespace autoware_pov::vision
{

RunAutoSpeedNode::RunAutoSpeedNode(const rclcpp::NodeOptions & options)
: Node("run_autospeed_node", options)
{
  // Get parameters
  const std::string model_path = this->declare_parameter<std::string>("model_path");
  const std::string precision = this->declare_parameter<std::string>("precision", "fp32");
  const int gpu_id = this->declare_parameter<int>("gpu_id", 0);
  const std::string input_topic = this->declare_parameter<std::string>("input_topic", "/sensors/camera/image_raw");
  output_topic_str_ = this->declare_parameter<std::string>("output_topic", "/autospeed/detections");
  
  // Detection thresholds
  conf_threshold_ = this->declare_parameter<double>("conf_threshold", 0.6);
  iou_threshold_ = this->declare_parameter<double>("iou_threshold", 0.45);

  // Create AutoSpeed TensorRT backend
  backend_ = std::make_unique<autospeed::AutoSpeedTensorRTEngine>(
    model_path, precision, gpu_id
  );

  // Benchmark timer
  timer_ = FpsTimer(BENCHMARK_OUTPUT_FREQUENCY);

  // Setup publishers and subscribers
  det_pub_ = this->create_publisher<vision_msgs::msg::Detection2DArray>(
    output_topic_str_, 10
  );
  
  sub_ = image_transport::create_subscription(
    this, input_topic, 
    std::bind(&RunAutoSpeedNode::onImage, this, std::placeholders::_1), 
    "raw", rmw_qos_profile_sensor_data
  );
  
  RCLCPP_INFO(this->get_logger(), "AutoSpeed node initialized");
  RCLCPP_INFO(this->get_logger(), "Model: %s", model_path.c_str());
  RCLCPP_INFO(this->get_logger(), "Precision: %s", precision.c_str());
  RCLCPP_INFO(this->get_logger(), "Confidence threshold: %.2f", conf_threshold_);
  RCLCPP_INFO(this->get_logger(), "IoU threshold: %.2f", iou_threshold_);
  RCLCPP_INFO(this->get_logger(), "Input topic: %s", input_topic.c_str());
  RCLCPP_INFO(this->get_logger(), "Output topic: %s", output_topic_str_.c_str());
}

void RunAutoSpeedNode::onImage(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
  // Benchmark: Start new frame
  timer_.startNewFrame();

  // Convert ROS image to OpenCV
  cv_bridge::CvImagePtr in_image_ptr;
  try {
    in_image_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  } catch (const cv_bridge::Exception & e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }

  // Benchmark: Preprocessing will be done in backend
  timer_.recordPreprocessEnd();

  // Run inference (preprocessing happens inside backend)
  if (!backend_->doInference(in_image_ptr->image)) {
    RCLCPP_WARN(this->get_logger(), "Inference failed.");
    return;
  }

  // Get raw tensor output
  const float* tensor_data = backend_->getRawTensorData();
  std::vector<int64_t> tensor_shape = backend_->getTensorShape();
  
  // Benchmark: Inference done
  timer_.recordInferenceEnd();
  
  // Post-process predictions
  auto detections = postProcessPredictions(tensor_data, tensor_shape);
  
  // Get letterbox parameters for coordinate transformation
  float scale;
  int pad_x, pad_y;
  backend_->getLetterboxParams(scale, pad_x, pad_y);
  
  // Get original image size
  int orig_width, orig_height;
  backend_->getOriginalImageSize(orig_width, orig_height);
  
  // Transform coordinates back to original image space
  transformCoordinates(detections, scale, pad_x, pad_y, orig_width, orig_height);
  
  // Create and publish Detection2DArray message
  vision_msgs::msg::Detection2DArray det_array_msg;
  det_array_msg.header = msg->header;
  
  for (const auto& det : detections) {
    vision_msgs::msg::Detection2D detection_msg;
    
    // Set bounding box center and size
    detection_msg.bbox.center.position.x = (det.x1 + det.x2) / 2.0;
    detection_msg.bbox.center.position.y = (det.y1 + det.y2) / 2.0;
    detection_msg.bbox.size_x = det.x2 - det.x1;
    detection_msg.bbox.size_y = det.y2 - det.y1;
    
    // Set detection result (class and confidence)
    vision_msgs::msg::ObjectHypothesisWithPose hyp;
    hyp.hypothesis.class_id = std::to_string(det.class_id);
    hyp.hypothesis.score = det.confidence;
    detection_msg.results.push_back(hyp);
    
    det_array_msg.detections.push_back(detection_msg);
  }
  
  det_pub_->publish(det_array_msg);
  
  // Benchmark: Output done
  timer_.recordOutputEnd();
  
  RCLCPP_DEBUG(this->get_logger(), "Published %zu detections", detections.size());
}

std::vector<Detection> RunAutoSpeedNode::postProcessPredictions(
  const float* raw_output,
  const std::vector<int64_t>& shape)
{
  // Shape: [1, num_predictions, num_attributes]
  // e.g., [1, 8400, 85] where 85 = 4 bbox + 1 objectness + 80 classes
  
  int num_predictions = static_cast<int>(shape[1]);
  int num_attributes = static_cast<int>(shape[2]);
  int num_classes = num_attributes - 5;  // First 4 are bbox, 5th is objectness
  
  std::vector<Detection> detections;
  
  // Iterate through all predictions
  for (int i = 0; i < num_predictions; ++i) {
    const float* pred = raw_output + i * num_attributes;
    
    // Get bounding box (cx, cy, w, h format)
    float cx = pred[0];
    float cy = pred[1];
    float w = pred[2];
    float h = pred[3];
    
    // Find max class score and ID
    float max_class_score = -1e9f;
    int best_class_id = 0;
    
    for (int c = 0; c < num_classes; ++c) {
      float class_score = pred[4 + c];
      
      // Apply sigmoid activation
      class_score = 1.0f / (1.0f + std::exp(-class_score));
      
      if (class_score > max_class_score) {
        max_class_score = class_score;
        best_class_id = c;
      }
    }
    
    // Apply confidence threshold
    if (max_class_score < conf_threshold_) {
      continue;
    }
    
    // Convert xywh to xyxy
    float x1, y1, x2, y2;
    xywh2xyxy(cx, cy, w, h, x1, y1, x2, y2);
    
    Detection det;
    det.x1 = x1;
    det.y1 = y1;
    det.x2 = x2;
    det.y2 = y2;
    det.confidence = max_class_score;
    det.class_id = best_class_id;
    
    detections.push_back(det);
  }
  
  // Apply Non-Maximum Suppression
  detections = applyNMS(detections, iou_threshold_);
  
  return detections;
}

void RunAutoSpeedNode::xywh2xyxy(
  float cx, float cy, float w, float h, 
  float& x1, float& y1, float& x2, float& y2)
{
  x1 = cx - w / 2.0f;
  y1 = cy - h / 2.0f;
  x2 = cx + w / 2.0f;
  y2 = cy + h / 2.0f;
}

float RunAutoSpeedNode::calculateIoU(const Detection& a, const Detection& b)
{
  // Calculate intersection area
  float x1 = std::max(a.x1, b.x1);
  float y1 = std::max(a.y1, b.y1);
  float x2 = std::min(a.x2, b.x2);
  float y2 = std::min(a.y2, b.y2);
  
  float intersection_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
  
  // Calculate union area
  float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
  float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
  float union_area = area_a + area_b - intersection_area;
  
  // Return IoU
  return (union_area > 0.0f) ? (intersection_area / union_area) : 0.0f;
}

std::vector<Detection> RunAutoSpeedNode::applyNMS(
  const std::vector<Detection>& detections,
  float iou_threshold)
{
  if (detections.empty()) {
    return {};
  }
  
  // Sort detections by confidence (descending)
  std::vector<Detection> sorted_dets = detections;
  std::sort(sorted_dets.begin(), sorted_dets.end(),
    [](const Detection& a, const Detection& b) {
      return a.confidence > b.confidence;
    });
  
  std::vector<Detection> result;
  std::vector<bool> suppressed(sorted_dets.size(), false);
  
  for (size_t i = 0; i < sorted_dets.size(); ++i) {
    if (suppressed[i]) {
      continue;
    }
    
    result.push_back(sorted_dets[i]);
    
    // Suppress overlapping detections
    for (size_t j = i + 1; j < sorted_dets.size(); ++j) {
      if (suppressed[j]) {
        continue;
      }
      
      float iou = calculateIoU(sorted_dets[i], sorted_dets[j]);
      if (iou > iou_threshold) {
        suppressed[j] = true;
      }
    }
  }
  
  return result;
}

void RunAutoSpeedNode::transformCoordinates(
  std::vector<Detection>& detections,
  float scale, int pad_x, int pad_y,
  int orig_width, int orig_height)
{
  for (auto& det : detections) {
    // Remove padding
    det.x1 = (det.x1 - pad_x) / scale;
    det.y1 = (det.y1 - pad_y) / scale;
    det.x2 = (det.x2 - pad_x) / scale;
    det.y2 = (det.y2 - pad_y) / scale;
    
    // Clamp to image bounds
    det.x1 = std::max(0.0f, std::min(static_cast<float>(orig_width), det.x1));
    det.y1 = std::max(0.0f, std::min(static_cast<float>(orig_height), det.y1));
    det.x2 = std::max(0.0f, std::min(static_cast<float>(orig_width), det.x2));
    det.y2 = std::max(0.0f, std::min(static_cast<float>(orig_height), det.y2));
  }
}

}  // namespace autoware_pov::vision

RCLCPP_COMPONENTS_REGISTER_NODE(autoware_pov::vision::RunAutoSpeedNode)
