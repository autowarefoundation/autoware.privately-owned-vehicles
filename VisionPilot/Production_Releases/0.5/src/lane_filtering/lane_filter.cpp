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

    // Process left line
    if (!start_left_vec.empty()) {
        cv::Point start_pt(
            start_left_vec[0], 
            start_left_vec[1]
        );
        
        // Step 2: sliding window search
        auto left_points = slidingWindowSearch(
            raw_input, 
            start_pt, 
            true
        );
        
        // Step 3: polyfit (cubic)
        LanePolyFit left_fit = fitPoly(left_points);
        
        // Temporal smoothing
        if (left_fit.valid) {
            if (prev_left_fit.valid) {
                for (size_t i = 0; i < 4; i++) {
                    left_fit.coeffs[i] = smoothing_factor * left_fit.coeffs[i] + 
                                       (1.0f - smoothing_factor) * prev_left_fit.coeffs[i];
                }
            }
            prev_left_fit = left_fit;
            drawPolyOnMask(
                clean_output.ego_left, 
                left_fit.coeffs
            );
        }
    } else {
        // If detection lost, maybe keep previous for a few frames? 
        // For now, invalidating.
        prev_left_fit.valid = false;
    }

    // Process right line
    if (!start_right_vec.empty()) {
        cv::Point start_pt(
            start_right_vec[0], 
            start_right_vec[1]
        );
        
        auto right_points = slidingWindowSearch(
            raw_input, 
            start_pt, 
            false
        );
        LanePolyFit right_fit = fitPoly(right_points);

        if (right_fit.valid) {
            if (prev_right_fit.valid) {
                for (size_t i = 0; i < 4; i++) {
                    right_fit.coeffs[i] = smoothing_factor * right_fit.coeffs[i] + 
                                        (1.0f - smoothing_factor) * prev_right_fit.coeffs[i];
                }
            }
            prev_right_fit = right_fit;
            drawPolyOnMask(
                clean_output.ego_right, 
                right_fit.coeffs
            );
        }
    } else {
        prev_right_fit.valid = false;
    }

    return clean_output;
}

// Step 1: find starting points in ROI
void LaneFilter::findStartingPoints(
    const LaneSegmentation& raw,
    std::vector<int>& start_left,
    std::vector<int>& start_right
)
{
    // Clear outputs
    start_left.clear();
    start_right.clear();

    int mid_x = raw.width / 2; // 80

}