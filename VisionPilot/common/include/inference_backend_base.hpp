#ifndef INFERENCE_BACKEND_BASE_HPP_
#define INFERENCE_BACKEND_BASE_HPP_

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace autoware_pov::vision
{

class InferenceBackend
{
public:
  virtual ~InferenceBackend() = default;

  virtual bool doInference(const cv::Mat & input_image) = 0;
  
  virtual void getRawOutput(cv::Mat & output, const cv::Size & output_size, const std::string & model_type = "segmentation") const = 0;

  virtual int getModelInputHeight() const = 0;
  virtual int getModelInputWidth() const = 0;
};

}  // namespace autoware_pov::vision

#endif  // INFERENCE_BACKEND_BASE_HPP_