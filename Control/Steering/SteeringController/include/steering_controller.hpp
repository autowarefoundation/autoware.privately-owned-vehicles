#pragma once

#include <iostream>
#include <cmath>

class SteeringController
{
public:
    SteeringController(double wheelbase,
                       double K_cte,
                       double K_damp);
    double computeSteering(double cte, double yaw_error, double forward_velocity, double curvature);

private:
    double wheelbase_,
        K_cte,
        K_damp;
};