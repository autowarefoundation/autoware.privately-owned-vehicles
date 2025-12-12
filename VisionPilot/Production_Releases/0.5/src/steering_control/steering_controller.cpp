/**
 * @file steering_controller.cpp
 * @brief Implementation of steering controller
 */

#include "steering_control/steering_controller.hpp"

namespace autoware_pov::vision::steering_control {

SteeringController::SteeringController(double wheelbase,
                                       double K_p,
                                       double K_i,
                                       double K_d)
    : wheelbase_(wheelbase), K_p(K_p), K_i(K_i), K_d(K_d)
{
    std::cout << "Steering controller initialized with parameters:\n"
              << "  wheelbase: " << wheelbase_ << " m\n"
              << "  K_p: " << K_p << "\n"
              << "  K_i: " << K_i << "\n"
              << "  K_d: " << K_d << std::endl;
    prev_yaw_error = 0.0;
}

double SteeringController::computeSteering(double cte, double yaw_error, double forward_velocity, double curvature)
{
    // Combined controller:
    // - Derivative term: K_d * (yaw_error - prev_yaw_error)
    // - Stanley controller: atan(K_i * cte / (forward_velocity + eps))
    // - Proportional term: K_p * yaw_error
    // - Curvature feedforward: -atan(curvature * wheelbase_)
    double steering_angle = K_d * (yaw_error - prev_yaw_error) 
                          + std::atan(K_i * cte / (forward_velocity + 1e-3)) 
                          + K_p * yaw_error 
                          - std::atan(curvature * wheelbase_);
    prev_yaw_error = yaw_error;
    return steering_angle;
}

} // namespace autoware_pov::vision::steering_control

