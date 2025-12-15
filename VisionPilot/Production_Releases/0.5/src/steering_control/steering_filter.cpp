//
// Created by atanasko on 12/14/25.
//

#include "steering_control/steering_filter.hpp"

namespace autoware_pov::vision::steering_control
{
SteeringFilter::SteeringFilter(const float smoothing_factor, float initial)
  : tau_(smoothing_factor), previous_steering(initial)
{
}

float SteeringFilter::filter(const float current_steering, const float dt)
{
  const float alpha = dt / (tau_ + dt);

  previous_steering = alpha * current_steering + (1.0f - alpha) * previous_steering;

  return previous_steering;
}

void SteeringFilter::reset(float value)
{
  previous_steering = value;
}
} // namespace autoware_pov::vision::steering_control