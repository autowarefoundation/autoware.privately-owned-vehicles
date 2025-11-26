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

    // Mid boundary
    int mid_x = raw.width / 2; // 80

    // Search left side to determine egoleft start
    for (int x = mid_x - 1; x >= 0; x--) {
        float sum = 0.0f;
        // Check vertical strip in ROI
        for (int y = roi_y_min; y <= roi_y_max; y++) {
            // Priority to ego_left, but robustly we check for any signal if we assume
            // the geometric position defines the class. 
            // However, sticking to the specific channel is safer for start points.
            sum += raw.ego_left.at<float>(y, x);
        }
        
        // Threshold for "finding a line start"
        if (sum > 2.0f) { 
            // Found it! Calculate centroid Y
            start_left.push_back(x);
            start_left.push_back((roi_y_min + roi_y_max) / 2);
            break; // Stop at first line found (closest to center)
        }
    }

    // Search right side for egoright start
    for (int x = mid_x; x < raw.width; x++) {
        float sum = 0.0f;
        for (int y = roi_y_min; y <= roi_y_max; y++) {
            sum += raw.ego_right.at<float>(y, x);
        }
        
        if (sum > 2.0f) {
            start_right.push_back(x);
            start_right.push_back((roi_y_min + roi_y_max) / 2);
            break;
        }
    }
}

// Step 2: sliding window search
std::vector<cv::Point> LaneFilter::slidingWindowSearch(
    const LaneSegmentation& raw,
    cv::Point start_point,
    bool is_left_lane)
{
    std::vector<cv::Point> lane_points;
    cv::Point current_pos = start_point;
    
    // Initial direction: Straight Up
    float dir_x = 0.0f; 
    float dir_y = -1.0f; 

    int num_windows = raw.height / sliding_window_height;

    for (int w = 0; w < num_windows; w++) {
        // Safety bounds
        if (
            current_pos.y < 0 || 
            current_pos.x < 0 || 
            current_pos.x >= raw.width
        ) break;

        // Define window boundaries around current_pos
        int win_y_low = std::max(
            0, 
            current_pos.y - sliding_window_height
        );
        int win_y_high = current_pos.y;
        int win_x_low = std::max(
            0, 
            current_pos.x - sliding_window_width
        );
        int win_x_high = std::min(
            raw.width, 
            current_pos.x + sliding_window_width
        );

        std::vector<cv::Point> window_pixels;
        float sum_x = 0.0f;
        float sum_y = 0.0f;

        // Scan inside window
        for (int y = win_y_low; y < win_y_high; y++) {
            for (int x = win_x_low; x < win_x_high; x++) {
                
                // --- MIXING LOGIC ---
                // "Regardless of the line type... include those markings"
                float val_l = raw.ego_left.at<float>(y, x);
                float val_r = raw.ego_right.at<float>(y, x);
                float val_o = raw.other_lanes.at<float>(y, x);
                
                // If ANY channel shows a lane here, take it
                if (
                    val_l > 0.5f || 
                    val_r > 0.5f || 
                    val_o > 0.5f
                ) {
                    window_pixels.push_back(cv::Point(x, y));
                    sum_x += x;
                    sum_y += y;
                }
            }
        }

        // If we found pixels in this window
        if (!window_pixels.empty()) {
            lane_points.insert(
                lane_points.end(), 
                window_pixels.begin(), 
                window_pixels.end()
            );
            
            // Calculate new centroid
            float centroid_x = sum_x / window_pixels.size();
            float centroid_y = sum_y / window_pixels.size();

            // ANGLE CALCULATION
            // Update search direction based on vector [Old -> New]
            float dx = centroid_x - current_pos.x;
            float dy = centroid_y - current_pos.y;
            
            float len = std::sqrt(dx*dx + dy*dy);
            if (len > 0.1f) { // Avoid normalization of zero vector
                dir_x = dx / len;
                dir_y = dy / len;
            }

            // Move window to new centroid
            current_pos = cv::Point(
                static_cast<int>(std::round(centroid_x)), 
                static_cast<int>(std::round(centroid_y))
            );
        } else {
            // --- OCCLUSION HANDLING ---
            // "Keep sliding window in the current direction"
            // Multiply by height to jump the gap
            current_pos.x += static_cast<int>(dir_x * sliding_window_height * 2.0f);
            current_pos.y += static_cast<int>(dir_y * sliding_window_height * 2.0f);
        }

        // Force upward movement if we got stuck horizontally to prevent infinite loops
        if (current_pos.y >= win_y_high - 1) {
            current_pos.y -= sliding_window_height;
        }
    }

    return lane_points;
}