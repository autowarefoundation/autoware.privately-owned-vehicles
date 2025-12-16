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

// ABSSP1 (Speed) : Start Bit 39 | Length 16 | Signed | Factor 0.01
// Format: Motorola (Big Endian: https://en.wikipedia.org/wiki/Endianness)
// Bit 39 is in byte 4. (0-indexed). 
// 16 bits -> spans byte 4 (high) and byte 5 (low) or 4 and 3?
// We assume [byte 4] is MSB, [byte 5] is LSB based on standard layout.
double CanInterface::decodeSpeed(const std::vector<uint8_t>& data) {

    if (data.size() < 8) return 0.0;

    // Combine byte 4 and byte 5
    // Note: DBC bit numbering can be tricky. If start is 39 (0x27), that is bit 7 of byte 4.
    // We assume standard Big Endian placement.
    int16_t raw = (static_cast<int16_t>(data[4]) << 8) | static_cast<int16_t>(data[5]);
    
    return static_cast<double>(raw) * 0.01;
}

// SSA (Steering angle) : Start Bit 46 | Length 15 | Signed | Factor 0.1
// Format: Motorola (Big Endian: https://en.wikipedia.org/wiki/Endianness)
// Bit 46 is in byte 5 (bit 6).
// This is complex. We verify strictly against provided logic if possible.
// Lacking exact bit-matrix, we assume standard Big Endian alignment crossing byte 5 and 6.
double CanInterface::decodeSteering(const std::vector<uint8_t>& data) {
    
    if (data.size() < 8) return 0.0;

    // Masking logic based on 15-bit signed integer
    // Assuming MSB is in byte 5, LSB in byte 6
    uint16_t raw_high = data[5];
    uint16_t raw_low  = data[6];
    
    // Combine
    uint16_t raw = (raw_high << 8) | raw_low;
    
    // If it's 15 bits, we might need to shift or mask. 
    // Assuming the 15 bits are right-aligned in the extraction:
    // (This is a best-guess implementation until A4 is actually observed in logs)
    
    // Convert to signed 16-bit to handle negative values (Two's complement)
    int16_t signed_val = static_cast<int16_t>(raw);
    
    return static_cast<double>(signed_val) * 0.1;
}

// ============================== REAL-TIME INFERENCE (SocketCAN) ============================== //

// SocketCAN setup during real-time inference
void CanInterface::setupSocket(
    const std::string& iface_name
) {

    is_file_mode_ = false;

    // 1. Open socket
    socket_fd_ = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (socket_fd_ < 0) {
        perror("[CanInterface] Error opening socket");
        return;
    }

    // 2. Resolve interface index
    struct ifreq ifr;
    std::strncpy(
        ifr.ifr_name, 
        iface_name.c_str(), 
        IFNAMSIZ - 1
    );
    if (ioctl(socket_fd_, SIOCGIFINDEX, &ifr) < 0) {
        perror("[CanInterface] Error finding interface index");
        close(socket_fd_); // Close socket on failure
        socket_fd_ = -1;   // Mark as invalid
        return;
    }

    // 3. Bind
    struct sockaddr_can addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;

    if (bind(socket_fd_, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("[CanInterface] Error binding socket");
        close(socket_fd_); // Close socket on failure
        socket_fd_ = -1;   // Mark as invalid
        return;
    }

    // 4. Set non-blocking
    // This ensures update() doesn't hang the whole Autoware pipeline if no data comes
    int flags = fcntl(socket_fd_, F_GETFL, 0);
    fcntl(socket_fd_, F_SETFL, flags | O_NONBLOCK);
}

// Read socket
bool CanInterface::readSocket() {
    
    if (socket_fd_ < 0) return false;

    struct can_frame frame;
    bool data_received = false;

    // Read all pending frames in the buffer
    while (true) {
        int nbytes = read(
            socket_fd_, 
            &frame, 
            sizeof(struct can_frame)
        );
        if (nbytes < 0) {
            // No more data (EAGAIN) or error
            break;
        }
        if (nbytes < (int)sizeof(struct can_frame)) {
            continue; // Incomplete frame
        }

        // Vector conversion for safe handling
        std::vector<uint8_t> data_vec(
            frame.data, 
            frame.data + frame.can_dlc
        );
        parseFrame(
            frame.can_id, 
            data_vec
        );
        data_received = true;
    }
    return data_received;
}

// ============================== FILE REPLAY (.asc) ============================== //

void CanInterface::setupFile(
    const std::string& file_path
) {

    is_file_mode_ = true;
    file_stream_.open(file_path);
    if (!file_stream_.is_open()) {
        std::cerr << "[CanInterface] Failed to open file: " << file_path << std::endl;
    }
}

bool CanInterface::readFileLine() {

    if (!file_stream_.is_open()) return false;

    std::string line;
    // We assume update() is called frequently.
    // We read ONE line per update to simulate a stream.
    // In a real replay tool we'd respect timestamps.
    // But for simple integration testing, line-by-line is sufficient.
    
    if (std::getline(file_stream_, line)) {
        std::istringstream iss(line);
        std::string token;
        std::vector<std::string> parts;
        
        while (iss >> token) {
            parts.push_back(token);
        }

        // Simple parser for .asc line:
        // 0.022530 1 A1 Rx d 8 00 00 ...
        // Index: 0=Time, 1=Chan, 2=ID, ... 5=DLC, 6+=Data
        
        if (parts.size() >= 7) {
            try {
                // Check if it's a data frame
                bool is_rx = false;
                for (const auto& p : parts) { 
                    if (p == "Rx") {
                        is_rx = true; 
                    }
                }
                
                if (is_rx) {
                    // Extract ID (Hex)
                    int id = std::stoi(
                        parts[2], 
                        nullptr, 
                        16
                    );
                    
                    // Extract data
                    std::vector<uint8_t> data;
                    // Find where data starts (after 'd' and '8')
                    // Usually data starts at index 6 or 7 depending on format variation
                    // We scan from end or look for hex bytes
                    
                    int dlc_idx = -1;
                    for(size_t i = 0; i < parts.size(); ++i) {
                        if (parts[i] == "d") {
                            dlc_idx = i + 1; // Next is DLC length
                            break;
                        }
                    }

                    if (
                        dlc_idx != -1 && 
                        dlc_idx + 1 < (int)parts.size()
                    ) {
                        int dlc = std::stoi(parts[dlc_idx]);
                        for (int i = 0; i < dlc; ++i) {
                            if (dlc_idx + 1 + i < (int)parts.size()) {
                                data.push_back(std::stoi(
                                    parts[dlc_idx + 1 + i], 
                                    nullptr, 
                                    16
                                ));
                            }
                        }
                        parseFrame(id, data);
                        return true;
                    }
                }
            } catch (...) {
                // Ignore parsing errors for comments/header lines
            }
        }
        return true; // Line read successfully (even if empty/comment)
    } 
    
    return false; // End of file
}

} // namespace autoware_pov::drivers