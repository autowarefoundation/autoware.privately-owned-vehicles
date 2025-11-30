#ifndef AUTOWARE_POV_VISION_AUTOSTEER_DRAW_LANES_HPP_
#define AUTOWARE_POV_VISION_AUTOSTEER_DRAW_LANES_HPP_

#include "../inference/onnxruntime_engine.hpp"
#include <opencv2/opencv.hpp>

namespace autoware_pov::vision::autosteer
{

/**
 * @brief Visualize lane segmentation on input image
 * 
 * Draws colored circles on detected lane pixels:
 * - Blue: Ego left lane
 * - Magenta: Ego right lane  
 * - Green: Other lanes
 * 
 * @param input_image Original input image (any resolution)
 * @param lanes Lane segmentation masks (typically 320x640)
 * @param radius Radius of drawn circles (default: 2)
 * @return Annotated image (same size as input)
 */
cv::Mat drawLanes(
  const cv::Mat& input_image,
  const LaneSegmentation& lanes,
  int radius = 2
);

/**
 * @brief In-place lane visualization (modifies input image)
 * 
 * @param image Image to annotate (modified in-place)
 * @param lanes Lane segmentation masks
 * @param radius Radius of drawn circles
 */
void drawLanesInPlace(
  cv::Mat& image,
  const LaneSegmentation& lanes,
  int radius = 2
);

/**
 * @brief In-place visualization of filtered lanes
 * 
 * @param image Image to annotate (modified in-place)
 * @param lanes Filtered lane segmentation masks
 */
void drawFilteredLanesInPlace(
  cv::Mat& image,
  const LaneSegmentation& lanes);

}  // namespace autoware_pov::vision::autosteer

#endif  // AUTOWARE_POV_VISION_AUTOSTEER_DRAW_LANES_HPP_

/**
 * @brief Draws ONLY the raw 160x80 pixel masks overlay.
 * Useful for debugging the model output and RANSAC inputs.
 * Later on I might add some more, like sliding windows too etc.
 */
void drawRawMasksInPlace(
  cv::Mat& image,
  const LaneSegmentation& lanes
);

/**
 * @brief Draws ONLY the smooth polynomial fitted lines.
 * This represents the final product output Insha'Allah.
 */
void drawPolyFitLanesInPlace(
  cv::Mat& image,
  const LaneSegmentation& lanes
);