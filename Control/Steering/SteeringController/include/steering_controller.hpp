#pragma once

#include <iostream>
#include <cmath>

class SteeringController
{
public:
    SteeringController(double wheelbase,
                       double K_yaw,
                       double K_cte,
                       double K_damp,
                       double K_ff);
    double computeSteering(double cte, double yaw_error, double forward_velocity, double curvature);

private:
    double wheelbase_,
        K_yaw,
        K_cte,
        K_damp,
        K_ff;
};