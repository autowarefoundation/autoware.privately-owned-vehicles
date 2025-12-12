/**
 * @file steering_controller.hpp
 * @brief Steering controller for path following
 * 
 * Combines Stanley controller (PI on CTE) with curvature feedforward
 * Input: CTE, yaw_error, forward_velocity, curvature (from PathFinder)
 * Output: Steering angle (radians)
 */

#pragma once

#include <iostream>
#include <cmath>

namespace autoware_pov::vision::steering_control {

/**
 * @brief Default steering controller parameters
 */
namespace SteeringControllerDefaults {
    constexpr double WHEELBASE = 2.85;        // Vehicle wheelbase (meters)
    constexpr double K_P = 0.8;                // Proportional gain for yaw error
    constexpr double K_I = 1.6;                // Integral gain for CTE (Stanley controller)
    constexpr double K_D = 1.0;                // Derivative gain for yaw error
    constexpr double FORWARD_VELOCITY = 10.0;  // Default forward velocity (m/s)
}

class SteeringController
{
public:
    /**
     * @brief Constructor
     * @param wheelbase Distance between front and rear axles (meters)
     * @param K_p Proportional gain for yaw error
     * @param K_i Integral gain for CTE (Stanley controller)
     * @param K_d Derivative gain for yaw error
     */
    SteeringController(double wheelbase,
                       double K_p,
                       double K_i,
                       double K_d);
    
    /**
     * @brief Compute steering angle
     * @param cte Cross-track error (meters)
     * @param yaw_error Yaw error (radians)
     * @param forward_velocity Forward velocity (m/s)
     * @param curvature Path curvature (1/meters)
     * @return Steering angle (radians)
     */
    double computeSteering(double cte, double yaw_error, double forward_velocity, double curvature);

private:
    double wheelbase_;
    double K_p, K_i, K_d;
    double prev_yaw_error;
};

} // namespace autoware_pov::vision::steering_control

