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

// Helper func to gen points from polynimial with scaling
static std::vector<cv::Point> genSmoothCurve(
    const std::vector<double>& coeffs, 
    int img_width, 
    int img_height, 
    int model_width, 
    int model_height
)
{
    std::vector<cv::Point> points;
    if (coeffs.size() < 4) return points;

    double a = coeffs[0];
    double b = coeffs[1];
    double c = coeffs[2];
    double d = coeffs[3];

    // Scaling factors
    // Polyfit coeffs are calculated in model space (160x80)
    // But gotta bring em into image space (640x360 or smth)
    double scale_y = static_cast<double>(model_height) / img_height;
    double scale_x = static_cast<double>(img_width) / model_width;

    // Iterate over every single Y pixel in the FINAL image for maximum smoothness
    for (int y_img = 0; y_img < img_height; ++y_img) {
        
        // 1. Image Y -> model Y
        double y_model = y_img * scale_y;

        // 2. Calc X in model space via poly
        double x_model = a * std::pow(y_model, 3) + 
                         b * std::pow(y_model, 2) + 
                         c * y_model + 
                         d;

        // 3. Model X -> image X
        double x_img = x_model * scale_x;

        // 4. Store valid points
        if (x_img >= 0 && x_img < img_width) {
            points.push_back(
              cv::Point(
                static_cast<int>(x_img), 
                y_img
              )
            );
        }
    }
    return points;
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

    // 1. Other lines
    if (!lanes.other_lanes.empty()) {
        cv::Mat other_mask_resized;
        // Nearest neighbor resize to keep it binary-ish before smoothing
        cv::resize(
          lanes.other_lanes, 
          other_mask_resized, 
          image.size(), 
          0, 0, 
          cv::INTER_NEAREST
        );
        
        // Create green overlay
        cv::Mat green_layer(
          image.size(), 
          image.type(), 
          color_other
        );
        
        cv::Mat mask_8u;
        other_mask_resized.convertTo(
          mask_8u, 
          CV_8U, 
          255.0
        );
        
        // Apply simple threshold to clean up
        cv::threshold(
          mask_8u, 
          mask_8u, 
          127, 
          255, 
          cv::THRESH_BINARY
        );

        cv::Mat overlay;
        green_layer.copyTo(
          overlay, 
          mask_8u
        );
        
        // Add faint green glow
        cv::addWeighted(
          image, 
          1.0, 
          overlay, 
          0.4, 
          0, 
          image
        );
    }

    // 2. EgoLeft
    if (!lanes.left_coeffs.empty()) {
        auto left_points = genSmoothCurve(
            lanes.left_coeffs, 
            image.cols, 
            image.rows, 
            lanes.width, 
            lanes.height
        );
        // Polyline
        cv::polylines(
          image, 
          left_points, 
          false, 
          color_ego_left, 
          5, 
          cv::LINE_AA
        );
    }

    // 3. EgoRight
    if (!lanes.right_coeffs.empty()) {
        auto right_points = genSmoothCurve(
            lanes.right_coeffs, 
            image.cols, 
            image.rows, 
            lanes.width, 
            lanes.height
        );
        // Polyline
        cv::polylines(
          image, 
          right_points, 
          false, 
          color_ego_right, 
          5, 
          cv::LINE_AA
        );
      }
}

}  // namespace autoware_pov::vision::autosteer

