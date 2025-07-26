#pragma once

#include "inference_backend.hpp"
#include <memory>
#include <string>
#include <opencv2/opencv.hpp>

namespace autoware_pov::AutoSeg::DomainSeg
{

class DomainSeg
{
public:
  DomainSeg(
    const std::string & model_path, const std::string & backend, 
    const std::string & precision, int gpu_id);
  
  ~DomainSeg() = default;
  
  bool doInference(const cv::Mat & input_image);
  void getDomainMask(cv::Mat & domain_mask, const cv::Size & output_size) const;
  void colorizeDomain(const cv::Mat & domain_mask, cv::Mat & colorized_domain) const;
  void createWeightedVisualization(const cv::Mat & input_image, const cv::Mat & domain_mask, cv::Mat & blended_output, double alpha = 0.5) const;
  
  int getModelInputHeight() const;
  int getModelInputWidth() const;

private:
  std::unique_ptr<InferenceBackend> backend_;
};

}  // namespace autoware_pov::AutoSeg::DomainSeg 