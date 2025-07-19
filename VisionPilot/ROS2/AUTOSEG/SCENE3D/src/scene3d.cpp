#include "scene3d.hpp"
#include "onnx_runtime_backend.hpp"
#include "tensorrt_backend.hpp"
#include <stdexcept>
#include <rclcpp/rclcpp.hpp>

namespace autoware_pov::AutoSeg::Scene3D
{

Scene3D::Scene3D(
  const std::string & model_path, const std::string & backend, 
  const std::string & precision, int gpu_id)
{
  if (backend == "onnxruntime") {
    backend_ = std::make_unique<OnnxRuntimeBackend>(model_path, precision, gpu_id);
    RCLCPP_INFO(rclcpp::get_logger("scene3d"), "Using ONNX Runtime backend");
  } else if (backend == "tensorrt") {
    backend_ = std::make_unique<TensorRTBackend>(model_path, precision, gpu_id);
    RCLCPP_INFO(rclcpp::get_logger("scene3d"), "Using TensorRT backend");
  } else {
    throw std::invalid_argument("Unknown backend: " + backend + ". Use 'onnxruntime' or 'tensorrt'");
  }
}

bool Scene3D::doInference(const cv::Mat & input_image)
{
  return backend_->doInference(input_image);
}

void Scene3D::getDepthMap(cv::Mat & depth_map, const cv::Size & output_size) const
{
  backend_->getDepthMap(depth_map, output_size);
}

void Scene3D::colorizeDepth(const cv::Mat & depth_map, cv::Mat & colorized_depth) const
{
  // Normalize depth to 0-255 range
  cv::Mat normalized_depth;
  double min_val, max_val;
  cv::minMaxLoc(depth_map, &min_val, &max_val);
  
  if (max_val > min_val) {
    depth_map.convertTo(normalized_depth, CV_8UC1, 255.0 / (max_val - min_val), -min_val * 255.0 / (max_val - min_val));
  } else {
    // Handle case where all values are the same
    normalized_depth = cv::Mat::zeros(depth_map.size(), CV_8UC1);
  }
  
  // Apply viridis colormap (OpenCV's closest equivalent to matplotlib's viridis)
  cv::applyColorMap(normalized_depth, colorized_depth, cv::COLORMAP_VIRIDIS);
}

int Scene3D::getModelInputHeight() const
{
  return backend_->getModelInputHeight();
}

int Scene3D::getModelInputWidth() const
{
  return backend_->getModelInputWidth();
}

}  // namespace autoware_pov::AutoSeg::Scene3D 