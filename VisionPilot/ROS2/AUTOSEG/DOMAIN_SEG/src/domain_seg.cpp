#include "domain_seg.hpp"
#include "onnx_runtime_backend.hpp"
#include "tensorrt_backend.hpp"
#include <stdexcept>
#include <rclcpp/rclcpp.hpp>

namespace autoware_pov::AutoSeg::DomainSeg
{

DomainSeg::DomainSeg(
  const std::string & model_path, const std::string & backend, 
  const std::string & precision, int gpu_id)
{
  if (backend == "onnxruntime") {
    backend_ = std::make_unique<OnnxRuntimeBackend>(model_path, precision, gpu_id);
    RCLCPP_INFO(rclcpp::get_logger("domain_seg"), "Using ONNX Runtime backend");
  } else if (backend == "tensorrt") {
    backend_ = std::make_unique<TensorRTBackend>(model_path, precision, gpu_id);
    RCLCPP_INFO(rclcpp::get_logger("domain_seg"), "Using TensorRT backend");
  } else {
    throw std::invalid_argument("Unknown backend: " + backend + ". Use 'onnxruntime' or 'tensorrt'");
  }
}

bool DomainSeg::doInference(const cv::Mat & input_image)
{
  return backend_->doInference(input_image);
}

void DomainSeg::getDomainMask(cv::Mat & domain_mask, const cv::Size & output_size) const
{
  backend_->getDomainMask(domain_mask, output_size);
}

void DomainSeg::colorizeDomain(const cv::Mat & domain_mask, cv::Mat & colorized_domain) const
{
  // Create colorized visualization based on the Python implementation
  colorized_domain = cv::Mat::zeros(domain_mask.size(), CV_8UC3);
  
  // Convert domain mask (already processed: 0.0 or 1.0) to 8-bit mask for OpenCV operations
  cv::Mat binary_mask;
  domain_mask.convertTo(binary_mask, CV_8UC1, 255.0);
  
  // Python sets colors in RGB format: vis_predict_object[:,:,0] = R, [:,:,1] = G, [:,:,2] = B
  // So we keep RGB format here to match Python exactly
  // Background color: RGB(255, 93, 61) -> RGB(255, 93, 61) 
  colorized_domain.setTo(cv::Scalar(255, 93, 61));
  
  // Foreground color: RGB(28, 148, 255) -> RGB(28, 148, 255)
  colorized_domain.setTo(cv::Scalar(28, 148, 255), binary_mask);
}

void DomainSeg::createWeightedVisualization(const cv::Mat & input_image, const cv::Mat & domain_mask, cv::Mat & blended_output, double alpha) const
{
  // Create colorized domain visualization (in RGB format like Python)
  cv::Mat colorized_domain;
  colorizeDomain(domain_mask, colorized_domain);
  
  // Python behavior: blends RGB vis_obj with BGR frame directly
  // ROS2 gives us BGR image, so we keep it as BGR (like Python's frame)
  cv::Mat input_bgr;
  if (input_image.channels() == 3) {
    // Input from ROS2 is BGR, keep as BGR to match Python's frame
    input_bgr = input_image.clone();
  } else {
    input_bgr = input_image.clone();
  }
  
  // Resize colorized domain (RGB) to match input image size if needed
  cv::Mat resized_domain;
  if (colorized_domain.size() != input_bgr.size()) {
    cv::resize(colorized_domain, resized_domain, input_bgr.size());
  } else {
    resized_domain = colorized_domain;
  }
  
  // Create weighted blend: blended = alpha * domain_vis(RGB) + (1-alpha) * original_image(BGR)
  // This exactly matches Python: cv2.addWeighted(vis_obj(RGB), alpha, frame(BGR), 1 - alpha, 0)
  cv::addWeighted(resized_domain, alpha, input_bgr, 1.0 - alpha, 0, blended_output);
}

int DomainSeg::getModelInputHeight() const
{
  return backend_->getModelInputHeight();
}

int DomainSeg::getModelInputWidth() const
{
  return backend_->getModelInputWidth();
}

}  // namespace autoware_pov::AutoSeg::DomainSeg 