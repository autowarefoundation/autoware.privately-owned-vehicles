#ifndef INFERENCE_BACKEND_BASE_HPP_
#define INFERENCE_BACKEND_BASE_HPP_

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// Forward declare for tensor support
namespace Ort { class Value; }

namespace autoware_pov::vision
{

class InferenceBackend
{
public:
  virtual ~InferenceBackend() = default;

  virtual bool doInference(const cv::Mat & input_image) = 0;
  
  // Only tensor access - no cv::Mat nonsense
  virtual const float* getRawTensorData() const = 0;
  virtual std::vector<int64_t> getTensorShape() const = 0;

  virtual int getModelInputHeight() const = 0;
  virtual int getModelInputWidth() const = 0;
};

}  // namespace autoware_pov::vision

#endif  // INFERENCE_BACKEND_BASE_HPP_