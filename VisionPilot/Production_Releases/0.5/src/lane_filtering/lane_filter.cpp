#include "lane_filtering/lane_filter.hpp"
#include "inference/onnxruntime_engine.hpp"
#include <cmath>
#include <algorithm>
#include <random>

namespace autoware_pov::vision::autosteer
{

LaneFilter::LaneFilter(float smoothing_factor) 
    : smoothing_factor(smoothing_factor) 
{
    std::random_device rd;
    rng = std::mt19937(rd());   // Seed RNG
    reset();
}

void LaneFilter::reset() {
    prev_left_fit.valid = false;
    prev_right_fit.valid = false;
}

// ============================== RANSAC-RELATED STUFFS ============================== //

// RANSAC helper func: calc X-error
double LaneFilter::getError(
    const std::vector<double>& c, 
    const cv::Point& p
) {
    double y = static_cast<double>(p.y);
    double x_pred = 0.0;
    
    // Evaluate based on model order
    // This is so that I can freely test different fit orders later
    if (c.size() == 4) {            // Cubic
        x_pred =    c[0]*pow(y,3) + \
                    c[1]*pow(y,2) + \
                    c[2]*y + \
                    c[3];

    } else if (c.size() == 3) {     // Quadratic
        x_pred =    c[0]*pow(y,2) + \
                    c[1]*y + \
                    c[2];

    } else if (c.size() == 2) {     // Linear
        x_pred =    c[0]*y + c[1];
    }
    
    return std::abs(x_pred - p.x);
}

// RANSAC helper func: least squares polyfit on subset
std::vector<double> LaneFilter::fitPolySimple(
    const std::vector<cv::Point>& subset, 
    int order
) {
    int n = subset.size();
    if (n <= order) return {}; 

    cv::Mat A(
        n, 
        order + 1, 
        CV_64F
    );
    cv::Mat B(
        n, 
        1, 
        CV_64F
    );

    for (int i = 0; i < n; ++i) {

        double y = static_cast<double>(subset[i].y);
        double x = static_cast<double>(subset[i].x);

        if (order == 1) {           // Linear: x = ay + b
            A.at<double>(i, 0) = y;
            A.at<double>(i, 1) = 1.0;

        } else if (order == 2) {    // Quad: x = ay^2 + by + c
            A.at<double>(i, 0) = y * y;
            A.at<double>(i, 1) = y;
            A.at<double>(i, 2) = 1.0;

        } else {                    // Cubic: x = ay^3 + by^2 + cy + d
            A.at<double>(i, 0) = y * y * y;
            A.at<double>(i, 1) = y * y;
            A.at<double>(i, 2) = y;
            A.at<double>(i, 3) = 1.0;
        }

        B.at<double>(i, 0) = x;
    }

    cv::Mat coeffs;
    if (cv::solve(
        A, 
        B, 
        coeffs, 
        cv::DECOMP_SVD
    )) {
        std::vector<double> res;
        for(int i = 0; i < coeffs.rows; i++) {
            res.push_back(coeffs.at<double>(i));
        }
        return res;
    }

    return {};
}

// RANSAC helper func: main fit logic
LanePolyFit LaneFilter::fitPoly(
    const std::vector<cv::Point>& points
) {
    
    LanePolyFit result;
    result.valid = false;
    
    int n = points.size();
    if (n < min_pixels_for_fit) return result;

    // 1. Declare Y-range
    double min_y = 1000.0, max_y = -1.0;
    for (const auto& p : points) {
        if (p.y < min_y) min_y = p.y;
        if (p.y > max_y) max_y = p.y;
    }

    // 2. Dynamic order selection
    int order = 3; 
    if (n < 30) order = 1; 
    else if (n < 60) order = 2;

    // 3. RANSAC loop
    std::vector<double> best_model;
    std::vector<cv::Point> best_inliers;
    best_inliers = points; // Default to all points if RANSAC finds nothing better

    // Only run RANSAC if we have enough points to spare
    if (n > 12) {
        int points_needed = order + 1;
        std::vector<cv::Point> sample_pool = points;
        
        for (int i = 0; i < ransac_iterations; ++i) {
            // Shuffle and pick small sample
            std::shuffle(
                sample_pool.begin(), 
                sample_pool.end(), 
                rng
            );
            std::vector<cv::Point> sample(
                sample_pool.begin(), 
                sample_pool.begin() + points_needed
            );
            
            // Fit temporary model
            std::vector<double> model = fitPolySimple(sample, order);
            if (model.empty()) continue;

            // Count inliers
            std::vector<cv::Point> current_inliers;
            for (const auto& p : points) {
                if (getError(model, p) < ransac_threshold) {
                    current_inliers.push_back(p);
                }
            }

            // Keep if better
            if (current_inliers.size() > best_inliers.size()) {
                best_inliers = current_inliers;
                best_model = model;
            }
        }
    }

    // 4. Least squares refit on inliers
    // This rejects the outliers (noise) that RANSAC identified
    if (best_inliers.size() >= static_cast<size_t>(order + 1)) {
        std::vector<double> final_coeffs = fitPolySimple(best_inliers, order);
        
        if (!final_coeffs.empty()) {
            // Norm output to always size 6 (cubic + lims)
            result.coeffs.assign(6, 0.0);
            
            if (order == 1) {           // Linear: [a, b] -> [0, 0, a, b]
                result.coeffs[2] = final_coeffs[0];
                result.coeffs[3] = final_coeffs[1];
            } else if (order == 2) {    // Quadratic: [a, b, c] -> [0, a, b, c]
                result.coeffs[1] = final_coeffs[0];
                result.coeffs[2] = final_coeffs[1];
                result.coeffs[3] = final_coeffs[2];
            } else {                    // Cubic: [a, b, c, d]
                result.coeffs[0] = final_coeffs[0];
                result.coeffs[1] = final_coeffs[1];
                result.coeffs[2] = final_coeffs[2];
                result.coeffs[3] = final_coeffs[3];
            }

            // Pack limits
            result.coeffs[4] = min_y;
            result.coeffs[5] = max_y;
            result.valid = true;
        }
    }

    return result;
}

// ============================================================================= //

// Master update func
LaneSegmentation LaneFilter::update(const LaneSegmentation& raw_input) {
    LaneSegmentation clean_output;
    clean_output.width = raw_input.width;   // 160
    clean_output.height = raw_input.height; // 80
    
    clean_output.ego_left = raw_input.ego_left.clone();
    clean_output.ego_right = raw_input.ego_right.clone();
    clean_output.other_lanes = raw_input.other_lanes.clone();

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
            clean_output.left_coeffs = left_fit.coeffs; 
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
            clean_output.right_coeffs = right_fit.coeffs;
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

// Step 2: sliding window search, now with:
// - Perspective-aware window size
// - Priority logic
// - Bi-directional search
std::vector<cv::Point> LaneFilter::slidingWindowSearch(
    const LaneSegmentation& raw,
    cv::Point start_point,
    bool is_left_lane)
{
    std::vector<cv::Point> lane_points;
    
    // Here a helper lambda func to allow bi-directional search
    // step_y = -1 for upwards, +1 for downwards 
    // (yeah basically here for my downward search)
    auto runSearch = [&](int step_y) {

        cv::Point current_pos = start_point;
        
        // If going DOWN, start one step below the start point to avoid duplication
        if (step_y > 0) current_pos.y += sliding_window_height;

        float dir_x = 0.0f;
        float dir_y = static_cast<float>(step_y);
        int consecutive_empty = 0; 

        // Normalize initial dir
        float len = std::sqrt(dir_x * dir_x + dir_y * dir_y);
        dir_x /= len; dir_y /= len;

        // Step by a percentage of the window height to ensure overlap
        float step_size = sliding_window_height * 0.8f; 

        int max_steps = (raw.height / step_size); // Safety limit

        for (int i = 0; i < max_steps; i++) {
            // 1. Boundary checks
            if (
                current_pos.x < 0 || 
                current_pos.x >= raw.width
            ) break;
            if (
                step_y < 0 && 
                current_pos.y < 0
            ) break;            // Hit Top
            if (
                step_y > 0 && 
                current_pos.y >= raw.height
            ) break;            // Hit Bottom

            // 2. Define window

            // Dynamic window width based on Y-position (which I call "perspective-aware")
            // Near bottom (y >= 60%) : forgiving, width = 8 pixels
            // Rest (y < 60%) : surgically precise, width = 2 pixels
            int min_window_width = 1;
            int max_window_width = 8;
            float threshold = 0.6f;

            float height_thresold = raw.height * threshold;
            int current_width;

            if (current_pos.y < height_thresold) {
                current_width = min_window_width;
            } else {
                current_width = max_window_width;
            }

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

            // Y-coord-based priority strategy switch
            // If we are above y=60 (pixels 0-59), we enter stricter ego-prioritized mode
            // In this mode, we completely ignore "other_lanes" masks to avoid false positives
            bool strict_ego_mode = (current_pos.y < 60);

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
                    // Only consider other_lanes if NOT in strict mode
                    if (!strict_ego_mode && val_other > 0.5f) {
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
            // Other_lanes will be empty if strict_ego_mode is true, effectively disabling this branch
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
                if (consecutive_empty >= 4) break; 

                // Advance blindly
                current_pos.x += static_cast<int>(dir_x * sliding_window_height);
                current_pos.y += static_cast<int>(dir_y * sliding_window_height);
            }

            if (current_pos.y >= win_y_high - 1) {
                current_pos.y -= sliding_window_height;
            }
        }
    };

    return lane_points;
}

}  // namespace autoware_pov::vision::autosteer