#include "lane_filtering/lane_filter.hpp"
#include "inference/onnxruntime_engine.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace autoware_pov::vision::autosteer
{

// Helper func: drawing polynomial curve on mask
static void drawPolyOnMask(
    cv::Mat& mask, 
    const std::vector<double>& coeffs
) {
    if (coeffs.size() < 4) return;
    
    std::vector<cv::Point> curve_points;
    // Iterate from top to bottom of the mask
    for (int y = 0; y < mask.rows; y++) {
        // x = ay^3 + by^2 + cy + d
        double y_d = static_cast<double>(y);
        double x_d = coeffs[0] * pow(y_d, 3) + 
                     coeffs[1] * pow(y_d, 2) + 
                     coeffs[2] * y_d + 
                     coeffs[3];
        
        int x = static_cast<int>(std::round(x_d));
        if (x >= 0 && x < mask.cols) {
            curve_points.push_back(cv::Point(x, y));
        }
    }

    if (curve_points.size() > 1) {
        cv::polylines(
            mask, 
            curve_points, 
            false, 
            cv::Scalar(1.0f), 
            2, 
            cv::LINE_AA
        );
    }
}

LaneFilter::LaneFilter(float smoothing_factor) 
    : smoothing_factor(smoothing_factor) 
{
    reset();
}

void LaneFilter::reset() {
    prev_left_fit.valid = false;
    prev_right_fit.valid = false;
}

// Master update func
LaneSegmentation LaneFilter::update(const LaneSegmentation& raw_input) {
    LaneSegmentation clean_output;
    clean_output.width = raw_input.width;   // 160
    clean_output.height = raw_input.height; // 80
    
    // Initialize blank masks
    clean_output.ego_left = cv::Mat::zeros(
        clean_output.height, 
        clean_output.width, 
        CV_32FC1
    );
    clean_output.ego_right = cv::Mat::zeros(
        clean_output.height, 
        clean_output.width, 
        CV_32FC1
    );
    clean_output.other_lanes = raw_input.other_lanes.clone(); // Pass through others for now

    // Step 1: ROI for starting points
    std::vector<int> start_left_vec;
    std::vector<int> start_right_vec;
    findStartingPoints(
        raw_input, 
        start_left_vec, 
        start_right_vec
    );