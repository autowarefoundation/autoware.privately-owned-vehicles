#include "drivers/can_interface.hpp"

#include <iostream>
#include <cstring>
#include <sstream>
#include <iomanip>

// Linux SocketCAN headers
#include <unistd.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <fcntl.h>


namespace autoware_pov::drivers {

// Constructor
CanInterface::CanInterface(
    const std::string& interface_name
) {

    // If file path (.asc) : file replay
    if (interface_name.find(".asc") != std::string::npos) {
        std::cout << "[CanInterface] Detected .asc file extension. Initializing file replay mode: " 
                  << interface_name << std::endl;
        setupFile(interface_name);
    } 
    // Else : real-time inference via SocketCAN
    else {
        std::cout << "[CanInterface] Initializing real-time inference mode (SocketCAN): " 
                  << interface_name << std::endl;
        setupSocket(interface_name);
    }
}

// Destructor
CanInterface::~CanInterface() {

    if (
        !is_file_mode_ && 
        socket_fd_ >= 0
    ) {
        close(socket_fd_);
    }

    if (
        is_file_mode_ && 
        file_stream_.is_open()
    ) {
        file_stream_.close();
    }
}

// Main update loop
bool CanInterface::update() {

    if (is_file_mode_) {
        return readFileLine();
    } else {
        return readSocket();
    }
}

// Get latest vehicle state
CanVehicleState CanInterface::getState() const {

    return current_state_;
}

// Decoding from CAN frame
void CanInterface::parseFrame(
    int can_id, 
    const std::vector<uint8_t>& data
) {

    if (data.empty()) return;

    // ID 0xA1 (A1, 161) => Speed
    if (can_id == ID_SPEED) {
        double val = decodeSpeed(data);
        current_state_.speed_kmph = val;
        current_state_.is_valid = true;
    } 
    // ID 0xA4 (A4, 164) => Steering angle
    else if (can_id == ID_STEERING) {
        double val = decodeSteering(data);
        current_state_.steering_angle_deg = val;
        current_state_.is_valid = true;
    }
}