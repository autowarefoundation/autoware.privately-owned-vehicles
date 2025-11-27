#include "lane_filtering/lane_filter.hpp"
#include "inference/onnxruntime_engine.hpp"
#include <cmath>
#include <algorithm>

namespace autoware_pov::vision::autosteer
{

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
            clean_output.left_coeffs = prev_left_fit.coeffs; 
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
            clean_output.right_coeffs = prev_right_fit.coeffs;
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

    // EGO LEFT START POINT
    bool found_left = false;
    // Search Y from bottom to top of ROI
    for (int y = roi_y_max; y >= roi_y_min; y--) {
        // Search X from mid to left
        for (int x = mid_x - 1; x >= 0; x--) {
            if (raw.ego_left.at<float>(y, x) > 0.5f) {
                start_left.push_back(x);
                start_left.push_back(y);
                found_left = true;
                break;
            }
        }
        if (found_left) break;
    }

    // EGO RIGHT START POINT
    bool found_right = false;
    // Search Y from bottom to top of ROI
    for (int y = roi_y_max; y >= roi_y_min; y--) {
        // Search X from mid to right
        for (int x = mid_x; x < raw.width; x++) {
            if (raw.ego_right.at<float>(y, x) > 0.5f) {
                start_right.push_back(x);
                start_right.push_back(y);
                found_right = true;
                break;
            }
        }
        if (found_right) break;
    }
}

// Step 2: sliding window search (now with perspective-aware window size)
std::vector<cv::Point> LaneFilter::slidingWindowSearch(
    const LaneSegmentation& raw,
    cv::Point start_point,
    bool is_left_lane)
{
    std::vector<cv::Point> lane_points;
    cv::Point current_pos = start_point;
    
    // Initial direction:
    // - Left lane: upwards-left
    // - Right lane: upwards-right
    float dir_x = (
        is_left_lane ? 
        0.1f : 
        -0.1f
    );
    float dir_y = -1.0f;
    int consecutive_empty = 0; 

    // Normalize initial dir
    float len = std::sqrt(dir_x * dir_x + dir_y * dir_y);
    dir_x /= len; dir_y /= len;

    // Step by a percentage of the window height to ensure overlap
    float step_size = sliding_window_height * 0.8f; 

    int max_steps = (raw.height / step_size) * 1.5; // Safety limit

    for (int i = 0; i < max_steps; i++) {
        // 1. Boundary checks
        if (
            current_pos.y < 0 || 
            current_pos.y >= raw.height || 
            current_pos.x < 0 || 
            current_pos.x >= raw.width
        ) {
            break;
        }

        // 2. Define window

        // Dynamic window width based on Y-position (which I call "perspective-aware")
        // At bottom (y = mask.height = 80): full width (100%) ~ 8 pixels
        // At top (y = 0): narrow width to infinitesimal ~ 0 pixels
        int min_window_width = 0;
        int max_window_width = 8;
        float width_factor = static_cast<float>(current_pos.y) / raw.height;
        int dynamic_width = static_cast<int>(
            min_window_width + 
            width_factor * (max_window_width - min_window_width)
        );
        int current_width  = std::max(
            1, 
            dynamic_width
        );

        int win_y_low = std::max(
            0, 
            current_pos.y - sliding_window_height
        );
        int win_y_high = current_pos.y;
        int win_x_low = std::max(
            0, 
            current_pos.x - current_width
        );
        int win_x_high = std::min(
            raw.width, 
            current_pos.x + current_width
        );

        // Buckets for priority logic
        std::vector<cv::Point> ego_pixels;
        std::vector<cv::Point> other_pixels;
        
        long sum_x_ego = 0, sum_y_ego = 0;
        long sum_x_other = 0, sum_y_other = 0;

        // 3. Collect pixels via class-agnostic search
        for (int y = win_y_low; y < win_y_high; y++) {

            for (int x = win_x_low; x < win_x_high; x++) {

                float val_ego = (
                    is_left_lane ? 
                    raw.ego_left.at<float>(y, x) : 
                    raw.ego_right.at<float>(y, x)
                );
                float val_other = raw.other_lanes.at<float>(y, x);
                
                // Sort pixels into buckets
                if (val_ego > 0.5f) {
                    ego_pixels.push_back(cv::Point(x, y));
                    sum_x_ego += x;
                    sum_y_ego += y;
                }
                if (val_other > 0.5f) {
                    other_pixels.push_back(cv::Point(x, y));
                    sum_x_other += x;
                    sum_y_other += y;
                }
            }
        }

        // PRIORITY DECISION
        float centroid_x, centroid_y;
        bool found_valid = false;

        // 1. Primary: Do we have strong EGO signal? (>= 3 pixels)
        if (ego_pixels.size() >= 3) {
            lane_points.insert(
                lane_points.end(), 
                ego_pixels.begin(), 
                ego_pixels.end()
            );
            centroid_x = static_cast<float>(sum_x_ego) / ego_pixels.size();
            centroid_y = static_cast<float>(sum_y_ego) / ego_pixels.size();
            found_valid = true;
        } 
        // 2. Secondary: If Ego is missing, do we have OTHER signal?
        else if (other_pixels.size() >= 3) {
            lane_points.insert(
                lane_points.end(), 
                other_pixels.begin(), 
                other_pixels.end()
            );
            centroid_x = static_cast<float>(sum_x_other) / other_pixels.size();
            centroid_y = static_cast<float>(sum_y_other) / other_pixels.size();
            found_valid = true;
        }

        // 4. Update state
        if (found_valid) {
            consecutive_empty = 0;
            
            // Update Momentum
            float dx = centroid_x - current_pos.x;
            float dy = centroid_y - current_pos.y;
            
            float len = std::sqrt(dx*dx + dy*dy);
            if (len > 0.1f) {
                dir_x = dx / len;
                dir_y = dy / len;
            }
            current_pos = cv::Point(static_cast<int>(std::round(centroid_x)), 
                                    static_cast<int>(std::round(centroid_y)));
        } else {
            // Horizon cutoff
            if (current_pos.y < raw.height * 0.25) break; 

            consecutive_empty++;
            if (consecutive_empty >= 3) break; 

            // Advance blindly
            current_pos.x += static_cast<int>(dir_x * sliding_window_height);
            current_pos.y += static_cast<int>(dir_y * sliding_window_height);
        }

        if (current_pos.y >= win_y_high - 1) {
            current_pos.y -= sliding_window_height;
        }
    }

    return lane_points;
}

// Step 3: cubic polynomial fit (now with range limits)
LanePolyFit LaneFilter::fitPoly(const std::vector<cv::Point>& points) {
    
    LanePolyFit result;
    result.valid = false;
    
    if (points.size() < static_cast<size_t>(min_pixels_for_fit)) {
        return result;
    }

    // Calc Y-range for packed coeffs later
    double min_y = 1000.0, max_y = -1.0;
    for (const auto& p : points) {
        if (p.y < min_y) min_y = p.y;
        if (p.y > max_y) max_y = p.y;
    }

    // Design matrix for cubic fit: y^3, y^2, y, 1
    cv::Mat A(
        points.size(), 
        4, 
        CV_64F
    );
    cv::Mat B(
        points.size(), 
        1, 
        CV_64F
    );

    for (size_t i = 0; i < points.size(); ++i) {
        double y = static_cast<double>(points[i].y);
        double x = static_cast<double>(points[i].x);

        A.at<double>(i, 0) = y * y * y;
        A.at<double>(i, 1) = y * y;
        A.at<double>(i, 2) = y;
        A.at<double>(i, 3) = 1.0;

        B.at<double>(i, 0) = x;
    }

    cv::Mat coeffs;
    // SVD is robust against singular matrices
    if (
        cv::solve(
            A, 
            B, 
            coeffs, 
            cv::DECOMP_SVD
        )
    ) {
        result.coeffs.resize(6);
        result.coeffs[0] = coeffs.at<double>(0);    // a
        result.coeffs[1] = coeffs.at<double>(1);    // b
        result.coeffs[2] = coeffs.at<double>(2);    // c
        result.coeffs[3] = coeffs.at<double>(3);    // d
        result.coeffs[4] = min_y;   // Top lim
        result.coeffs[5] = max_y;   // Bottom lim
        result.valid = true;
    }

    return result;
}

}  // namespace autoware_pov::vision::autosteer