// This script takes the output of the lane fitler.

// The blue and pink perspective image polynomial fitted lines need to be sampled and the samples should be transformed into a BEV space using a Homography transform (please note - we are not creating a BEV image, only transforming discrete points from the perspective coordinates to the BEV coordinates).

// To do this, you can simply multiply the coordinates of the polyfit line samples by the Homography matrix.

// Verify this is working by visualizing the transformed points.

// Once this has been verfied, we will proceed with the drivable corridor parameter estimation and temporal tracking.

#include "lane_tracking/lane_tracking.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>


namespace autoware_pov::vision::autosteer
{

LaneTracker::LaneTracker() {
    // Homography is pre-computed elsewhere and hard-coded.
    // If you got any problem with it, ask me.
}

void LaneTracker::initHomography(const cv::Size& image_size) {

    if (
        homography_inited && 
        cached_image_size == image_size
    ) return;

    cached_image_size = image_size;
    homography_inited = true;
    
}

std::pair<LaneSegmentation, DualViewMetrics> LaneTracker::update(
    const LaneSegmentation& input_lanes,
    const cv::Size& image_size
) {
    
    // Ensure homography ready
    initHomography(image_size);

    LaneSegmentation output_lanes = input_lanes;
    DualViewMetrics metrics;

    // Coeffs in input_lanes are normalized to 160 x 80.
    // Must upscale em to image_size for warping.
    
    double scale_x = static_cast<double>(image_size.width) / input_lanes.width;
    double scale_y = static_cast<double>(image_size.height) / input_lanes.height;

    // Helper lambda to upscale coeffs
    auto upscaleCoeffs = [&](const std::vector<double>& c) {
        std::vector<double> up(6);
        
        // y_img = y_model * scale_y
        // x_img = x_model * scale_x
        // x_model = ay^2 + by + c
        // x_img/sx = a(y_img/sy)^2 + b(y_img/sy) + c
        // x_img = (a*sx/sy^2)*y_img^2 + (b*sx/sy)*y_img + (c*sx)

        if (c.size() < 6) return up;
        up[0] = 0; // Cubic term ignored for now if we use quadratic
        if (c.size() == 6) { // Assuming quadratic storage [0, a, b, c, min, max]
             up[1] = c[1] * scale_x / (scale_y * scale_y);
             up[2] = c[2] * scale_x / scale_y;
             up[3] = c[3] * scale_x;
             up[4] = c[4] * scale_y;
             up[5] = c[5] * scale_y;
        }
        return up;
    };

    bool left_valid = !input_lanes.left_coeffs.empty();
    bool right_valid = !input_lanes.right_coeffs.empty();

    std::vector<cv::Point2f> left_pts_bev, right_pts_bev;


    // 1. Warp existing lines to BEV
    if (left_valid) {
        auto up_coeffs = upscaleCoeffs(input_lanes.left_coeffs);
        auto pts_pers = genPointsFromCoeffs(
            up_coeffs, 
            image_size.height
        );
        left_pts_bev = warpPoints(
            pts_pers, 
            H_orig_to_bev
        );
    }

    if (right_valid) {
        auto up_coeffs = upscaleCoeffs(input_lanes.right_coeffs);
        auto pts_pers = genPointsFromCoeffs(
            up_coeffs, 
            image_size.height
        );
        right_pts_bev = warpPoints(
            pts_pers, 
            H_orig_to_bev
        );
    }