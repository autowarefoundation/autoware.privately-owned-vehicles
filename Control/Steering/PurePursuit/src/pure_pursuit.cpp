#include "pure_pursuit.hpp"

PurePursuit::PurePursuit(double wheelbase)
    : wheelbase_(wheelbase)
{
    std::cout << "PurePursuit initialized." << std::endl;
}

double PurePursuit::computeSteering(double cte, double yaw_error, double curvature, double forward_velocity)
{
    lookahead_distance_ = 1.0 + forward_velocity * 0.5;
    steering_angle_ = std::atan2(2 * wheelbase_ * std::sin(yaw_error), lookahead_distance_) + std::atan2(curvature * wheelbase_, 1.0);
    std::cout << "Steering angle: " << steering_angle_ << std::endl;
    return steering_angle_;
}