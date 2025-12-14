#ifndef AUTOWARE_POV_DRIVERS_CAN_INTERFACE_HPP_
#define AUTOWARE_POV_DRIVERS_CAN_INTERFACE_HPP_

#include <string>
#include <optional>
#include <fstream>
#include <vector>
#include <cmath>
#include <limits>
#include <chrono>

namespace autoware_pov::drivers {

struct CanVehicleState {
    double speed_kmph = std::numeric_limits<double>::quiet_NaN();           // Speed, CAN ID 0xA1
    double steering_angle_deg = std::numeric_limits<double>::quiet_NaN();   // Steering, CAN ID 0xA4
    bool is_valid = false;
};