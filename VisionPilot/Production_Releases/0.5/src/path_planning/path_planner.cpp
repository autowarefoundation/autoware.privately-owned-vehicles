/**
 * @file path_planner.cpp
 * @brief Path planner implementation
 * 
 * Adapted from PATHFINDER's cb_drivCorr() and timer_callback()
 */

#include "path_planning/path_planner.hpp"
#include <iostream>
#include <fstream>
#include <random>
#include <limits>

namespace autoware_pov::vision::path_planning {

PathPlanner::PathPlanner(const cv::Mat& homography_matrix, double default_lane_width)
    : H_(homography_matrix.clone()), default_lane_width_(default_lane_width)
{
    if (H_.rows != 3 || H_.cols != 3) {
        throw std::runtime_error("Homography matrix must be 3x3");
    }
    
    initializeBayesFilter();
}

void PathPlanner::initializeBayesFilter()
{
    // Configure fusion groups (from PATHFINDER pathfinder_node constructor)
    // Fusion rules: {start_idx, end_idx} → fuse indices in [start, end) → result at end_idx
    bayes_filter_.configureFusionGroups({
        {0, 3},   // CTE: fuse [0,1,2] (path,left,right) → [3] (fused)
        {5, 7},   // Yaw: fuse [5,6] (left,right) → [7] (fused)
        {9, 11}   // Curvature: fuse [9,10] (left,right) → [11] (fused)
    });
    
    // Initialize state (large variance = uncertain)
    Gaussian default_state = {0.0, 1e3};
    std::array<Gaussian, STATE_DIM> init_state;
    init_state.fill(default_state);
    
    // Initialize lane width with reasonable prior
    init_state[12].mean = default_lane_width_;
    init_state[12].variance = 0.5 * 0.5;
    
    bayes_filter_.initialize(init_state);
    
    std::cout << "[PathPlanner] Initialized with default lane width: " 
              << default_lane_width_ << " m" << std::endl;
}

std::vector<cv::Point2f> PathPlanner::transformToBEV(
    const std::vector<cv::Point2f>& pixel_pts) const
{
    if (pixel_pts.empty()) {
        return {};
    }
    
    std::vector<cv::Point2f> bev_pts;
    cv::perspectiveTransform(pixel_pts, bev_pts, H_);
    
    return bev_pts;
}

PathPlanningOutput PathPlanner::update(
    const std::vector<cv::Point2f>& left_pts_pixel,
    const std::vector<cv::Point2f>& right_pts_pixel)
{
    PathPlanningOutput output;
    output.left_valid = false;
    output.right_valid = false;
    output.fused_valid = false;
    
    // 1. Predict step (add process noise, like timer_callback in PATHFINDER)
    std::array<Gaussian, STATE_DIM> process;
    std::random_device rd;
    std::default_random_engine generator(rd());
    const double epsilon = 0.00001;
    std::uniform_real_distribution<double> dist(-epsilon, epsilon);
    
    for (size_t i = 0; i < STATE_DIM; ++i) {
        process[i].mean = dist(generator);
        process[i].variance = PROC_SD * PROC_SD;
    }
    bayes_filter_.predict(process);
    
    // 2. Transform to BEV
    std::vector<cv::Point2f> left_bev = transformToBEV(left_pts_pixel);
    std::vector<cv::Point2f> right_bev = transformToBEV(right_pts_pixel);
    
    // 3. Fit polynomials
    auto left_coeff = fitQuadPoly(left_bev);
    auto right_coeff = fitQuadPoly(right_bev);
    
    FittedCurve left_curve(left_coeff);
    FittedCurve right_curve(right_coeff);
    
    output.left_coeff = left_coeff;
    output.right_coeff = right_coeff;
    output.left_valid = !std::isnan(left_curve.cte);
    output.right_valid = !std::isnan(right_curve.cte);
    
    // Store individual metrics
    output.left_cte = left_curve.cte;
    output.left_yaw_error = left_curve.yaw_error;
    output.left_curvature = left_curve.curvature;
    output.right_cte = right_curve.cte;
    output.right_yaw_error = right_curve.yaw_error;
    output.right_curvature = right_curve.curvature;
    
    // 4. Create measurement (adapted from cb_drivCorr)
    std::array<Gaussian, STATE_DIM> measurement;
    
    // Set measurement variances
    measurement[0].variance = STD_M_CTE * STD_M_CTE;
    measurement[1].variance = STD_M_CTE * STD_M_CTE;
    measurement[2].variance = STD_M_CTE * STD_M_CTE;
    measurement[3].variance = STD_M_CTE * STD_M_CTE;
    
    measurement[4].variance = STD_M_YAW * STD_M_YAW;
    measurement[5].variance = STD_M_YAW * STD_M_YAW;
    measurement[6].variance = STD_M_YAW * STD_M_YAW;
    measurement[7].variance = STD_M_YAW * STD_M_YAW;
    
    measurement[8].variance = STD_M_CURV * STD_M_CURV;
    measurement[9].variance = STD_M_CURV * STD_M_CURV;
    measurement[10].variance = STD_M_CURV * STD_M_CURV;
    measurement[11].variance = STD_M_CURV * STD_M_CURV;
    
    measurement[12].variance = STD_M_WIDTH * STD_M_WIDTH;
    measurement[13].variance = STD_M_WIDTH * STD_M_WIDTH;
    
    // Get current tracked width for CTE offset
    auto width = bayes_filter_.getState()[12].mean;
    
    // Set measurement means
    // [0,4,8] = ego path (we don't have it, set to NaN)
    measurement[0].mean = std::numeric_limits<double>::quiet_NaN();
    measurement[4].mean = std::numeric_limits<double>::quiet_NaN();
    measurement[8].mean = std::numeric_limits<double>::quiet_NaN();
    
    // [1,5,9] = left lane (offset CTE to lane center)
    measurement[1].mean = left_curve.cte + width / 2.0;
    measurement[5].mean = left_curve.yaw_error;
    measurement[9].mean = left_curve.curvature;
    
    // [2,6,10] = right lane (offset CTE to lane center)
    measurement[2].mean = right_curve.cte - width / 2.0;
    measurement[6].mean = right_curve.yaw_error;
    measurement[10].mean = right_curve.curvature;
    
    // [3,7,11] = fused (computed by Bayes filter)
    measurement[3].mean = std::numeric_limits<double>::quiet_NaN();
    measurement[7].mean = std::numeric_limits<double>::quiet_NaN();
    measurement[11].mean = std::numeric_limits<double>::quiet_NaN();
    
    // Lane width measurement (adapted from cb_drivCorr logic)
    if (std::isnan(left_curve.cte) && std::isnan(right_curve.cte)) {
        // Both lanes missing → use default
        measurement[12].mean = default_lane_width_;
    } else if (std::isnan(left_curve.cte)) {
        // Left missing → keep current tracked width
        measurement[12].mean = width;
    } else if (std::isnan(right_curve.cte)) {
        // Right missing → keep current tracked width
        measurement[12].mean = width;
    } else {
        // Both present → direct measurement
        measurement[12].mean = right_curve.cte - left_curve.cte;
    }
    
    measurement[13].mean = std::numeric_limits<double>::quiet_NaN();
    
    // 5. Update Bayes filter
    bayes_filter_.update(measurement);
    
    // 6. Extract fused state
    const auto& state = bayes_filter_.getState();
    
    output.cte = state[3].mean;
    output.yaw_error = state[7].mean;
    output.curvature = state[11].mean;
    output.lane_width = state[12].mean;
    
    output.cte_variance = state[3].variance;
    output.yaw_variance = state[7].variance;
    output.curv_variance = state[11].variance;
    
    output.fused_valid = !std::isnan(output.cte) && 
                         !std::isnan(output.yaw_error) && 
                         !std::isnan(output.curvature);
    
    // 7. Calculate center polynomial coefficients (average left/right)
    if (output.left_valid && output.right_valid) {
        FittedCurve center = calculateEgoPath(left_curve, right_curve);
        output.center_coeff = center.coeff;
    } else if (output.left_valid) {
        output.center_coeff = left_coeff;
    } else if (output.right_valid) {
        output.center_coeff = right_coeff;
    } else {
        output.center_coeff = {
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN()
        };
    }
    
    return output;
}

const std::array<Gaussian, STATE_DIM>& PathPlanner::getState() const
{
    return bayes_filter_.getState();
}

void PathPlanner::reset()
{
    initializeBayesFilter();
}

cv::Mat loadHomographyFromYAML(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open homography file: " + filename);
    }
    
    cv::Mat H = cv::Mat::eye(3, 3, CV_64F);
    
    std::string line;
    while (std::getline(file, line)) {
        // Look for "H:" line
        if (line.find("H:") != std::string::npos) {
            // Extract matrix values (space or comma separated)
            size_t start = line.find("[");
            size_t end = line.find("]");
            
            if (start != std::string::npos && end != std::string::npos) {
                std::string values_str = line.substr(start + 1, end - start - 1);
                
                // Parse 9 values
                std::stringstream ss(values_str);
                std::vector<double> values;
                double val;
                
                while (ss >> val) {
                    values.push_back(val);
                    // Skip comma if present
                    if (ss.peek() == ',') ss.ignore();
                }
                
                if (values.size() == 9) {
                    for (int i = 0; i < 9; ++i) {
                        H.at<double>(i / 3, i % 3) = values[i];
                    }
                    std::cout << "[PathPlanner] Loaded homography matrix from: " 
                              << filename << std::endl;
                    return H;
                } else {
                    throw std::runtime_error("Expected 9 values in H matrix, got " + 
                                           std::to_string(values.size()));
                }
            }
        }
    }
    
    throw std::runtime_error("Could not find 'H:' in file: " + filename);
}

} // namespace autoware_pov::vision::path_planning

