#pragma once

#include <iostream>
#include <cmath>

class SteeringController
{
public:
    SteeringController(double wheelbase,
                       double K_p,
                       double K_i,
                       double K_d);
    double computeSteering(double cte, double yaw_error, double forward_velocity, double curvature);

private:
    double wheelbase_,
        K_p, K_i, K_d,
        prev_yaw_error;
};