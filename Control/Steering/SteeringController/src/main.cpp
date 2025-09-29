#include "steering_controller.hpp"

int main()
{
    SteeringController sc(2.85, 2.0, 2.8, 1.0, 4.0);

    double cte = 0.5;               // cross-track error
    double yaw_error = 0.1;         // yaw error in radians
    double forward_velocity = 10.0; // forward velocity in m/s
    double curvature = 0.01;        // path curvature in 1/m
    while (true)
    {
        sc.computeSteering(cte, yaw_error, forward_velocity, curvature);
        double steering_angle_ = 2.0 * yaw_error + std::atan(2.8 * cte / (forward_velocity + 1)) - 4.0 * std::atan(curvature * 2.85);
        std::cout << "Steering Angle: " << steering_angle_ << std::endl;
    }

    return 0;
}