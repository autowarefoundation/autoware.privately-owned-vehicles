#include "steering_controller.hpp"

SteeringController::SteeringController(double wheelbase,
                                       double K_p,
                                       double K_i,
                                       double K_d)
    : wheelbase_(wheelbase), K_p(K_p),K_i(K_i),K_d(K_d)
    {
        std::cout << "Steering controller init with Params \n"
                  << "wheelbase_\t" << wheelbase_ << "\n"
                  << "K_p\t" << K_p << "\n"
                  << "K_i\t" << K_i << "\n"
                  << "K_d\t" << K_d << "\n";
        prev_yaw_error = 0.0;
    }

    double SteeringController::computeSteering(double cte, double yaw_error, double forward_velocity, double curvature)
    {
        // yaw error derivative (D) + Stanley (PI) +  curvature feedforward
        double steering_angle = K_d * (yaw_error - prev_yaw_error) + std::atan(K_i * cte / (forward_velocity + 1e-3)) + K_p * yaw_error - std::atan(curvature * wheelbase_);
        prev_yaw_error = yaw_error;
        return steering_angle;
    }