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
    if (coeffs.size() < 6) return points;

    double a         = coeffs[0];
    double b         = coeffs[1];
    double c         = coeffs[2];
    double d         = coeffs[3];
    double min_y_lim = coeffs[4];
    double max_y_lim = coeffs[5];

    // Scaling factors
    // Polyfit coeffs are calculated in model space (160x80)
    // But gotta bring em into image space (640x360 or smth)
    double scale_y = static_cast<double>(model_height) / img_height;
    double scale_x = static_cast<double>(img_width) / model_width;

    // Lims in image space
    int img_y_start = static_cast<int>(min_y_lim / scale_y);
    int img_y_end   = static_cast<int>(max_y_lim / scale_y);
    img_y_start = std::max(
      0, 
      img_y_start
    );
    img_y_end   = std::min(
      img_height - 1, 
      img_y_end
    );

    // Iterate ONLY within valid Y-range
    for (int y_img = img_y_start; y_img <= img_y_end; ++y_img) {
        
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

    // 1a. Raw mask of other lines
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

    // 1b. Raw mask of ego left line
    if (!lanes.ego_left.empty()) {
        cv::Mat ego_left_resized;
        cv::resize(
          lanes.ego_left,
          ego_left_resized,
          image.size(),
          0, 0,
          cv::INTER_NEAREST
        );

        cv::Mat mask_left_8u;
        ego_left_resized.convertTo(
          mask_left_8u,
          CV_8U,
          255.0
        );

        cv::threshold(
          mask_left_8u,
          mask_left_8u,
          127,
          255,
          cv::THRESH_BINARY
        );

        cv::Mat blue_layer(image.size(), image.type(), color_ego_left);
        cv::Mat overlay_left;
        blue_layer.copyTo(overlay_left, mask_left_8u);
        cv::addWeighted(image, 1.0, overlay_left, 0.35, 0, image);
    }

    // 1c. Raw mask of ego right line
    if (!lanes.ego_right.empty()) {
        cv::Mat ego_right_resized;
        cv::resize(
          lanes.ego_right,
          ego_right_resized,
          image.size(),
          0, 0,
          cv::INTER_NEAREST
        );

        cv::Mat mask_right_8u;
        ego_right_resized.convertTo(
          mask_right_8u,
          CV_8U,
          255.0
        );

        cv::threshold(
          mask_right_8u,
          mask_right_8u,
          127,
          255,
          cv::THRESH_BINARY
        );

        cv::Mat magenta_layer(image.size(), image.type(), color_ego_right);
        cv::Mat overlay_right;
        magenta_layer.copyTo(overlay_right, mask_right_8u);
        cv::addWeighted(image, 1.0, overlay_right, 0.35, 0, image);
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

// ========================== NEW VIS VIEWS - DEBUGGING + FINAL OUTPUTS ========================== //

// Helper func: draw mask overlay only
static void drawMaskOverlay(
  cv::Mat& image, 
  const cv::Mat& mask, 
  const cv::Scalar& color
) 
{
    if (mask.empty()) return;

    cv::Mat mask_resized;
    cv::resize(
      mask, 
      mask_resized, 
      image.size(), 
      0, 
      0, 
      cv::INTER_NEAREST
    ); 
    
    cv::Mat color_layer(
      image.size(), 
      image.type(), 
      color
    );
    cv::Mat mask_8u;
    mask_resized.convertTo(
      mask_8u, 
      CV_8U, 
      255.0
    );
    cv::threshold(
      mask_8u, 
      mask_8u, 
      127, 
      255, 
      cv::THRESH_BINARY
    );

    cv::Mat overlay;
    color_layer.copyTo(
      overlay, 
      mask_8u
    );
    cv::addWeighted(
      image, 
      1.0, 
      overlay, 
      0.4, 
      0, 
      image
    );
}

// Helper func: draw raw masks (debug view)
void drawRawMasksInPlace(
  cv::Mat& image, 
  const LaneSegmentation& lanes
)
{
    cv::Scalar color_ego_left(255, 0, 0);      // Blue
    cv::Scalar color_ego_right(255, 0, 200);   // Magenta
    cv::Scalar color_other(0, 153, 0);         // Green

    drawMaskOverlay(
      image, 
      lanes.other_lanes, 
      color_other
    );
    drawMaskOverlay(
      image, 
      lanes.ego_left, 
      color_ego_left
    );
    drawMaskOverlay(
      image, 
      lanes.ego_right, 
      color_ego_right
    );
    
    cv::putText(
      image, 
      "DEBUG: Raw masks", 
      cv::Point(20, 40), 
      cv::FONT_HERSHEY_SIMPLEX, 
      1.0, 
      cv::Scalar(0, 255, 255), 
      2
    );
}

}  // namespace autoware_pov::vision::autosteer

