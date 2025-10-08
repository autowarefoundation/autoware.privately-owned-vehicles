#include "pi_controller.hpp"

PI_Controller::PI_Controller(double K_p, double K_i)
{
    std::cout << "PI controller init with Params \n"
              << "K_p\t" << K_p << "\n"
              << "K_i\t" << K_i << "\n";
    integral_error = 0.0;
    this->K_p = K_p;
    this->K_i = K_i;
}

double PI_Controller::computeEffort(double current_speed_, double target_speed_)
{
    double error = target_speed_ - current_speed_;
    integral_error += error;
    double effort = K_p * error + K_i * integral_error;
    return effort;
}