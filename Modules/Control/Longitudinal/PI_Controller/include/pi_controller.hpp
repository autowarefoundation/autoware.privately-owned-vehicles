#pragma once

#include <iostream>
#include <cmath>
#include <algorithm>

class PI_Controller
{
public:
    PI_Controller(double K_p, double K_i, double K_d);
    double computeEffort(double current_speed_, double target_speed_);

private:
    double K_p,
        K_i,
        K_d,
        integral_error,
        prev_error;
};