#pragma once

#include <opencv2/opencv.hpp>

namespace autoware_pov::AutoSeg::DomainSeg
{

class InferenceBackend
{
public:
  virtual ~InferenceBackend() = default;
  
  virtual bool doInference(const cv::Mat & input_image) = 0;
  virtual void getDomainMask(cv::Mat & domain_mask, const cv::Size & output_size) const = 0;
  
  virtual int getModelInputHeight() const = 0;
  virtual int getModelInputWidth() const = 0;
};

}  // namespace autoware_pov::AutoSeg::DomainSeg 