#ifndef AUTOWARE_POV__AUTOSEG__SCENE_SEG__INFERENCE_BACKEND_HPP_
#define AUTOWARE_POV__AUTOSEG__SCENE_SEG__INFERENCE_BACKEND_HPP_

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace autoware_pov::AutoSeg::SceneSeg
{

class InferenceBackend
{
public:
  virtual ~InferenceBackend() = default;

  virtual bool doInference(const cv::Mat & input_image) = 0;
  virtual void getRawMask(cv::Mat & raw_mask, const cv::Size & output_size) const = 0;
};

}  // namespace autoware_pov::AutoSeg::SceneSeg

#endif  // AUTOWARE_POV__AUTOSEG__SCENE_SEG__INFERENCE_BACKEND_HPP_ 