#include "visualization/draw_lanes.hpp"

namespace autoware_pov::vision::autosteer
{

cv::Mat drawLanes(
  const cv::Mat& input_image,
  const LaneSegmentation& lanes,
  int radius)
{
  // Clone input for visualization
  cv::Mat vis_image = input_image.clone();
  drawLanesInPlace(vis_image, lanes, radius);
  return vis_image;
}

void drawLanesInPlace(
  cv::Mat& image,
  const LaneSegmentation& lanes,
  int radius)
{
  // Calculate scale from lane mask to input image
  float scale_x = static_cast<float>(image.cols) / lanes.width;
  float scale_y = static_cast<float>(image.rows) / lanes.height;
  
  // Define colors (BGR format for OpenCV)
  cv::Scalar color_ego_left(255, 0, 0);      // Blue
  cv::Scalar color_ego_right(255, 0, 200);   // Magenta
  cv::Scalar color_other(0, 153, 0);         // Green
  
  // Draw ego left lane
  for (int y = 0; y < lanes.height; ++y) {
    for (int x = 0; x < lanes.width; ++x) {
      if (lanes.ego_left.at<float>(y, x) > 0.5f) {
        int scaled_x = static_cast<int>(x * scale_x);
        int scaled_y = static_cast<int>(y * scale_y);
        cv::circle(image, cv::Point(scaled_x, scaled_y), 
                   radius, color_ego_left, -1, cv::LINE_AA);
      }
    }
  }
  
  // Draw ego right lane
  for (int y = 0; y < lanes.height; ++y) {
    for (int x = 0; x < lanes.width; ++x) {
      if (lanes.ego_right.at<float>(y, x) > 0.5f) {
        int scaled_x = static_cast<int>(x * scale_x);
        int scaled_y = static_cast<int>(y * scale_y);
        cv::circle(image, cv::Point(scaled_x, scaled_y), 
                   radius, color_ego_right, -1, cv::LINE_AA);
      }
    }
  }
  
  // Draw other lanes
  for (int y = 0; y < lanes.height; ++y) {
    for (int x = 0; x < lanes.width; ++x) {
      if (lanes.other_lanes.at<float>(y, x) > 0.5f) {
        int scaled_x = static_cast<int>(x * scale_x);
        int scaled_y = static_cast<int>(y * scale_y);
        cv::circle(image, cv::Point(scaled_x, scaled_y), 
                   radius, color_other, -1, cv::LINE_AA);
      }
    }
  }
}

void drawFilteredLanesInPlace(
  cv::Mat& image, 
  const LaneSegmentation& lanes
)
{

    // Define colors (BGR format for OpenCV)
    cv::Scalar color_ego_left(255, 0, 0);      // Blue
    cv::Scalar color_ego_right(255, 0, 200);   // Magenta
    cv::Scalar color_other(0, 153, 0);         // Green

    // 1. Prep blank canvas
    cv::Mat overlay = cv::Mat::zeros(
      image.size(), 
      image.type()
    );

    // 2. Process per channel
    auto processChannel = [&](const cv::Mat& small_mask, const cv::Scalar& color) {
        // Threshold first to ensure binary (clean up the 160x80 mask)
        cv::Mat bin_mask;
        cv::threshold(small_mask, bin_mask, 0.5, 1.0, cv::THRESH_BINARY);

        // Resize to full image size (Linear interpolation smooths the jagged edges!)
        cv::Mat full_mask;
        cv::resize(bin_mask, full_mask, image.size(), 0, 0, cv::INTER_LINEAR);

        // Create a solid color layer
        cv::Mat color_layer(image.size(), image.type(), color);
        
        // Convert mask to 8-bit for copy operation
        cv::Mat full_mask_8u;
        full_mask.convertTo(full_mask_8u, CV_8U, 255.0);
        
        // Copy the color only where the mask is active onto the overlay
        color_layer.copyTo(overlay, full_mask_8u);
    };

}  // namespace autoware_pov::vision::autosteer

