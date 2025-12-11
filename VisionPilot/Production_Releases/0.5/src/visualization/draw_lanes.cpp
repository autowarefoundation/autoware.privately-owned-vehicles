#include "visualization/draw_lanes.hpp"
#include <opencv2/core/types.hpp>

namespace autoware_pov::vision::autosteer
{

cv::Mat drawLanes(
  const cv::Mat& input_image,
  const LaneSegmentation& lanes)
{
  // Clone input for visualization
  cv::Mat vis_image = input_image.clone();
  drawLanesInPlace(vis_image, lanes);
  return vis_image;
}

void drawLanesInPlace(
  cv::Mat& image,
  const LaneSegmentation& lanes)
{
  // Calculate scale from lane mask to input image
  float scale_x = static_cast<float>(image.cols) / lanes.width;
  float scale_y = static_cast<float>(image.rows) / lanes.height;
  
  // Choose radius based on scale
  float base_scale = std::min(scale_x, scale_y);
  int radius_ = std::max(1, static_cast<int>(std::round(base_scale * 0.5f)));
  
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
                   radius_, color_ego_left, -1, cv::LINE_AA);
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
                   radius_, color_ego_right, -1, cv::LINE_AA);
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
                   radius_, color_other, -1, cv::LINE_AA);
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

// ========================== MAIN VIS VIEWS - DEBUGGING + FINAL OUTPUTS ========================== //

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

    // // Draw sliding windows for debugging
    // float scale_x = static_cast<float>(image.cols) / lanes.width;
    // float scale_y = static_cast<float>(image.rows) / lanes.height;
    // cv::Scalar color_window(0, 0, 255); // Red

    // // a. Left windows
    // for (const auto& rect : lanes.left_sliding_windows) {
    //     cv::Rect scaled_rect(
    //         static_cast<int>(rect.x * scale_x),
    //         static_cast<int>(rect.y * scale_y),
    //         static_cast<int>(rect.width * scale_x),
    //         static_cast<int>(rect.height * scale_y)
    //     );
    //     cv::rectangle(image, scaled_rect, color_window, 1);
    // }

    // // b. Right windows
    // for (const auto& rect : lanes.right_sliding_windows) {
    //     cv::Rect scaled_rect(
    //         static_cast<int>(rect.x * scale_x),
    //         static_cast<int>(rect.y * scale_y),
    //         static_cast<int>(rect.width * scale_x),
    //         static_cast<int>(rect.height * scale_y)
    //     );
    //     cv::rectangle(image, scaled_rect, color_window, 1);
    // }
    
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

// Helper func: draw smooth polyfit lines only (final prod view)
void drawPolyFitLanesInPlace(
  cv::Mat& image, 
  const LaneSegmentation& lanes
)
{
    cv::Scalar color_ego_left(255, 0, 0);     // Blue
    cv::Scalar color_ego_right(255, 0, 200);  // Magenta
    cv::Scalar color_center(0, 255, 255);     // Yellow
    
    // Draw vectors

    // Egoleft
    if (!lanes.left_coeffs.empty()) {
        auto left_points = genSmoothCurve(
          lanes.left_coeffs, 
          image.cols, 
          image.rows, 
          lanes.width, 
          lanes.height
        );
        if (left_points.size() > 1) {
            cv::polylines(
              image, 
              left_points, 
              false, 
              color_ego_left, 
              15, 
              cv::LINE_AA
            );
            cv::polylines(
              image, 
              left_points, 
              false, 
              cv::Scalar(255, 200, 0), 
              5, 
              cv::LINE_AA
            );
        }
    }

    // Egoright
    if (!lanes.right_coeffs.empty()) {
        auto right_points = genSmoothCurve(
          lanes.right_coeffs, 
          image.cols, 
          image.rows, 
          lanes.width, 
          lanes.height
        );
        if (right_points.size() > 1) {
            cv::polylines(
              image, 
              right_points, 
              false, 
              color_ego_right, 
              15, 
              cv::LINE_AA
            );
            cv::polylines(
              image, 
              right_points, 
              false, 
              cv::Scalar(255, 150, 255), 
              5, 
              cv::LINE_AA
            );
        }
    }

    // Drivable path
    if (
      lanes.path_valid && 
      !lanes.center_coeffs.empty()
    ) {
        std::vector<double> viz_coeffs = lanes.center_coeffs;
        viz_coeffs[5] = static_cast<double>(lanes.height - 1);  // Extend to bottom

        auto center_points = genSmoothCurve(
          viz_coeffs, 
          image.cols, 
          image.rows, 
          lanes.width, 
          lanes.height
        );
        
        if (center_points.size() > 1) {
            cv::polylines(
              image, 
              center_points, 
              false, 
              color_center, 
              15, 
              cv::LINE_AA
            );
            cv::polylines(
              image, 
              center_points, 
              false, 
              cv::Scalar(255, 255, 255), 
              5, 
              cv::LINE_AA
            );
        }

        // Params info as text for now
        std::vector<std::string> lines;
        lines.push_back(cv::format("Lane offset: %.2f px", lanes.lane_offset));
        lines.push_back(cv::format("Yaw offset: %.2f rad", lanes.yaw_offset));
        // lines.push_back(cv::format("Steering angle: %.2f deg", lanes.steering_angle));
        lines.push_back(cv::format("Curvature: %.4f", lanes.curvature));

        int font = cv::FONT_HERSHEY_SIMPLEX;
        double scale = 1.2;
        int thickness = 2;
        int line_spacing = 10; // extra spacing
        int margin = 50;
        int y = margin;
        
        for (const auto& l : lines) {
            cv::Size textSize = cv::getTextSize(
              l, 
              font, 
              scale, 
              thickness, 
              nullptr
            );
            int x = image.cols - textSize.width - margin;  // Align right
            cv::putText(
              image, 
              l, 
              cv::Point(x, y), 
              font, 
              scale, 
              color_center, 
              thickness
            );
            y += textSize.height + line_spacing;
        }
    }
    
    cv::putText(
      image, 
      "FINAL: RANSAC polyfit", 
      cv::Point(20, 40), 
      cv::FONT_HERSHEY_SIMPLEX, 
      1.0, 
      cv::Scalar(0, 255, 0), 
      2
    );
}

// ========================== ADDITIONAL VIS VIEW - BEV ========================== //

// Helper func: gen points from coeffs directly in BEV space (no scaling needed)
static std::vector<cv::Point> genBEVPoints(
    const std::vector<double>& coeffs,
    int bev_height = 640
)
{
    std::vector<cv::Point> points;
    // Now using quadratic coeffs: [0, a, b, c, min_y, max_y]
    if (coeffs.size() < 6) return points;

    double a = coeffs[1];
    double b = coeffs[2];
    double c = coeffs[3];
    double min_y = coeffs[4];
    double max_y = coeffs[5];

    for (int y = 0; y < bev_height; ++y) {
        // Only draw within valid y-range defined by fitted points
        if (y < min_y || y > max_y) continue;

        // x = ay^2 + by + c
        double x = a*y*y + b*y + c;
        
        // BEV grid is 640 wide
        if (x >= 0 && x < 640) {
            points.push_back(cv::Point(
              static_cast<int>(x), 
              y
            ));
        }
    }
    return points;
}

// Helper func: draw BEV vis
void drawBEVVis(
  cv::Mat& image,
  const cv::Mat& orig_frame,
  const BEVVisuals& bev_data
)
{
    // 1. Warp orig frame to BEV (640 x 640)
    if (image.size() != cv::Size(640, 640)) {
        image.create(
          640, 
          640, 
          orig_frame.type()
        );
    }

    cv::warpPerspective(
        orig_frame,
        image,
        bev_data.H_orig_to_bev,
        cv::Size(
          640, 
          640
        )
    );

    if (!bev_data.valid) {
        cv::putText(
          image, 
          "BEV Tracking: Waiting...", 
          cv::Point(20, 40), 
          cv::FONT_HERSHEY_SIMPLEX, 
          1.0, 
          cv::Scalar(0, 0, 255), 
          2
        );
        return;
    }

    int bev_h = 640;
    cv::Scalar color_left(255, 0, 0);     // Blue
    cv::Scalar color_right(255, 0, 200);  // Magenta
    cv::Scalar color_center(0, 255, 255); // Yellow
    int thickness = 4;

}  // namespace autoware_pov::vision::autosteer

