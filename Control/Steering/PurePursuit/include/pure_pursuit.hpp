#pragma once

#include <iostream>
#include <cmath>

class PurePursuit
{
public:
    PurePursuit(double wheelbase);
    double computeSteering(double cte, double yaw_error, double curvature, double forward_velocity);

private:
    double lookahead_distance_;
    double wheelbase_;
    double steering_angle_;
};