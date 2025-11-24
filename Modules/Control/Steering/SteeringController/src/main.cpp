#include "steering_controller.hpp"

int main()
{
    SteeringController sc(2.85, 0.8, 1.6, 1.0);

    double cte = 0.5;               // cross-track error
    double yaw_error = 0.1;         // yaw error in radians
    double forward_velocity = 10.0; // forward velocity in m/s
    double curvature = 0.01;        // path curvature in 1/m
    int i = 5;
    while (i--)
    {
        std::cout << sc.computeSteering(cte, yaw_error, forward_velocity, curvature) << std::endl;
    }

    return 0;
}