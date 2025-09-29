#include "steering_controller.hpp"

SteeringController::SteeringController(double wheelbase, double K_yaw, double K_cte, double K_damp, double K_ff)
    : wheelbase_(wheelbase), K_yaw(K_yaw), K_cte(K_cte), K_damp(K_damp), K_ff(K_ff)
{
    std::cout << "SteeringController initialized." << std::endl;
}

double SteeringController::computeSteering(double cte, double yaw_error, double forward_velocity, double curvature)
{
    // Stanley + curvature feedforward
    double steering_angle = K_yaw * yaw_error + std::atan(K_cte * cte / (forward_velocity + K_damp)) - K_ff * std::atan(curvature * wheelbase_);
    // std::cout << "Steering angle: " << steering_angle << std::endl;
    return steering_angle;
}