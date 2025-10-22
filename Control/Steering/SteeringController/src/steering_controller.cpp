#include "steering_controller.hpp"

SteeringController::SteeringController(double wheelbase, double K_cte, double K_damp)
    : wheelbase_(wheelbase), K_cte(K_cte), K_damp(K_damp)
{
    std::cout << "Steering controller init with Params \n" 
    << "wheelbase_\t" << wheelbase_   << "\n"
    << "K_cte\t"      << K_cte        << "\n"
    << "K_damp\t"     << K_damp       << "\n";
}

double SteeringController::computeSteering(double cte, double yaw_error, double forward_velocity, double curvature)
{
    // Stanley + curvature feedforward
    double steering_angle = std::atan(K_cte * cte / (forward_velocity + K_damp)) + yaw_error - std::atan(curvature * wheelbase_);
    return steering_angle;
}