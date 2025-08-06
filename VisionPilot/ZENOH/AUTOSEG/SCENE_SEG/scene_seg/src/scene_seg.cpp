

#include "scene_seg.hpp"
#include <vector>
#include <stdexcept>
//#include "rclcpp/rclcpp.hpp"

#include "onnx_runtime_backend.hpp"
//#include "tensorrt_backend.hpp"

namespace autoware_pov::AutoSeg::SceneSeg
{

SceneSeg::SceneSeg(const std::string & model_path, const std::string & backend, const std::string & precision, int gpu_id)
{
  if (backend == "onnxruntime") {
    backend_ = std::make_unique<OnnxRuntimeBackend>(model_path, precision, gpu_id);
    //RCLCPP_INFO(rclcpp::get_logger("scene_seg"), "Using ONNX Runtime backend");
  //} else if (backend == "tensorrt") {
  //  backend_ = std::make_unique<TensorRTBackend>(model_path, precision, gpu_id);
    //RCLCPP_INFO(rclcpp::get_logger("scene_seg"), "Using TensorRT backend");
  } else {
    //RCLCPP_ERROR(rclcpp::get_logger("scene_seg"), "Unsupported backend type: %s", backend.c_str());
    throw std::invalid_argument("Unsupported backend type.");
  }

  createColorMap();
}

void SceneSeg::createColorMap()
{
  color_map_.push_back(cv::Vec3b(0, 0, 0));
  color_map_.push_back(cv::Vec3b(0, 0, 255));
  color_map_.push_back(cv::Vec3b(0, 0, 0));
}

bool SceneSeg::doInference(const cv::Mat & input_image)
{
  return backend_->doInference(input_image);
}

void SceneSeg::getRawMask(cv::Mat & raw_mask, const cv::Size & output_size) const
{
  backend_->getRawMask(raw_mask, output_size);
}

void SceneSeg::colorizeMask(const cv::Mat & raw_mask, const cv::Mat & original_image, cv::Mat & blended_image) const
{
    cv::Mat color_mask(raw_mask.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    
    for (int y = 0; y < raw_mask.rows; ++y) {
        const uint8_t* mask_row = raw_mask.ptr<uint8_t>(y);
        cv::Vec3b* color_row = color_mask.ptr<cv::Vec3b>(y);
        
        for (int x = 0; x < raw_mask.cols; ++x) {
            uint8_t class_id = mask_row[x];
            if (class_id < color_map_.size()) {
                color_row[x] = color_map_[class_id];
            }
        }
    }

    cv::addWeighted(color_mask, 0.5, original_image, 0.5, 0.0, blended_image);
}

}  // namespace autoware_pov::AutoSeg::SceneSeg 