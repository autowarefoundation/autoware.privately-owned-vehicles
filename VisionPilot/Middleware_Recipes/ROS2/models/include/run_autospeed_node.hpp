#ifndef RUN_AUTOSPEED_NODE_HPP_
#define RUN_AUTOSPEED_NODE_HPP_

#include "../../../common/include/fps_timer.hpp"
#include "../../../common/backends/autospeed/tensorrt_engine.hpp"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <image_transport/image_transport.hpp>
#include <memory>

#define BENCHMARK_OUTPUT_FREQUENCY 100

namespace autoware_pov::vision
{

struct Detection {
  float x1, y1, x2, y2;  // Bounding box corners (in original image coordinates)
  float confidence;       // Detection confidence score
  int class_id;          // Object class ID
};

class RunAutoSpeedNode : public rclcpp::Node
{
public:
  explicit RunAutoSpeedNode(const rclcpp::NodeOptions & options);

private:
  void onImage(const sensor_msgs::msg::Image::ConstSharedPtr msg);
  
  // Post-processing methods
  std::vector<Detection> postProcessPredictions(
    const float* raw_output,
    const std::vector<int64_t>& shape
  );
  
  void xywh2xyxy(float cx, float cy, float w, float h, float& x1, float& y1, float& x2, float& y2);
  
  std::vector<Detection> applyNMS(
    const std::vector<Detection>& detections,
    float iou_threshold
  );
  
  float calculateIoU(const Detection& a, const Detection& b);
  
  void transformCoordinates(
    std::vector<Detection>& detections,
    float scale, int pad_x, int pad_y,
    int orig_width, int orig_height
  );

  // Parameters
  float conf_threshold_{0.6f};
  float iou_threshold_{0.45f};
  std::string output_topic_str_;
  bool benchmark_{false};
  
  // Backend
  std::unique_ptr<autospeed::AutoSpeedTensorRTEngine> backend_;

  // ROS
  image_transport::Subscriber sub_;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr det_pub_;

  // Benchmark timer (optional)
  std::unique_ptr<FpsTimer> timer_;
};

}  // namespace autoware_pov::vision

#endif  // RUN_AUTOSPEED_NODE_HPP_

