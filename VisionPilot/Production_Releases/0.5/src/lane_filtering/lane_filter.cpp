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