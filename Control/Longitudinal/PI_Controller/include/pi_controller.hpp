#pragma once

#include <iostream>
#include <cmath>

class PI_Controller
{
public:
    PI_Controller(double K_p, double K_i);
    double computeEffort(double current_speed_, double target_speed_);

private:
    double K_p, K_i, integral_error;
};