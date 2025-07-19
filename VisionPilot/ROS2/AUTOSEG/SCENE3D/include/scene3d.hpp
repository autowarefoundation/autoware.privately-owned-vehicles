#pragma once

#include "inference_backend.hpp"
#include <memory>
#include <string>
#include <opencv2/opencv.hpp>

namespace autoware_pov::AutoSeg::Scene3D
{

class Scene3D
{
public:
  Scene3D(
    const std::string & model_path, const std::string & backend, 
    const std::string & precision, int gpu_id);
  
  ~Scene3D() = default;
  
  bool doInference(const cv::Mat & input_image);
  void getDepthMap(cv::Mat & depth_map, const cv::Size & output_size) const;
  void colorizeDepth(const cv::Mat & depth_map, cv::Mat & colorized_depth) const;
  
  int getModelInputHeight() const;
  int getModelInputWidth() const;

private:
  std::unique_ptr<InferenceBackend> backend_;
};

}  // namespace autoware_pov::AutoSeg::Scene3D 