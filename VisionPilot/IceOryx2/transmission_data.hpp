#ifndef TRANSMISSION_DATA_HPP
#define TRANSMISSION_DATA_HPP

#include <cstdint>
#include <iostream>

// ============================================================================
// RawFrame: Zero-copy image frame 
// ============================================================================
struct RawFrame {
    uint64_t frame_id;                  // Sequential frame counter
    uint64_t timestamp_ns;              // Capture timestamp (nanoseconds)
    uint64_t publish_timestamp_ns;      // Publish timestamp (for IPC latency measurement)
    uint32_t width;                     // Frame width (1920)
    uint32_t height;                    // Frame height (1280)
    uint32_t channels;                  // Color channels (3 = BGR)
    uint32_t step;                      // Row stride in bytes
    uint8_t data[1920 * 1280 * 3];      // Fixed-size image payload
    
    bool is_valid;                      // Frame validity flag
    uint8_t source_id;                  // Camera/source identifier
    
    RawFrame() : frame_id(0), timestamp_ns(0), publish_timestamp_ns(0), 
                 width(1920), height(1280), channels(3), step(1920 * 3), 
                 is_valid(false), source_id(0) {}
};

// ============================================================================
// CIPOMessage: Main CIPO tracking information 
// ============================================================================
struct CIPOMessage {
    // Frame association
    uint64_t frame_id;                  // Links to RawFrame
    uint64_t timestamp_ns;              // Processing timestamp
    uint64_t publish_timestamp_ns;      // Publish timestamp (for IPC latency measurement)
    
    // Main CIPO data
    bool exists;                        // Main CIPO detected
    int32_t track_id;                   // Tracking ID (-1 if no CIPO)
    int32_t class_id;                   // CIPO level (1, 2, 3)
    float distance_m;                   // Distance in meters
    float velocity_ms;                  // Velocity in m/s (Kalman filtered)
    
    // Bounding box (for visualization)
    float bbox_x1;
    float bbox_y1;
    float bbox_x2;
    float bbox_y2;
    float confidence;                   // Detection confidence
    
    // Event flags
    bool cut_in_detected;               // Cut-in event flag
    bool kalman_reset;                  // Kalman filter reset flag
    
    // Status
    uint8_t num_tracked_objects;        // Total number of tracked objects
    float inference_latency_ms;         // Inference latency (ms)
    float tracking_latency_ms;          // Tracking latency (ms)
    float ipc_latency_us;               // IPC transfer latency from frame_node (microseconds)
    
    CIPOMessage() : frame_id(0), timestamp_ns(0), publish_timestamp_ns(0), 
                    exists(false), track_id(-1), class_id(0), distance_m(0.0f), 
                    velocity_ms(0.0f), bbox_x1(0.0f), bbox_y1(0.0f), 
                    bbox_x2(0.0f), bbox_y2(0.0f), confidence(0.0f), 
                    cut_in_detected(false), kalman_reset(false),
                    num_tracked_objects(0), inference_latency_ms(0.0f),
                    tracking_latency_ms(0.0f), ipc_latency_us(0.0f) {}
};

// ============================================================================
// Convenience: Printing for debugging
// ============================================================================
inline std::ostream& operator<<(std::ostream& os, const RawFrame& frame) {
    os << "RawFrame[id=" << frame.frame_id 
       << ", " << frame.width << "x" << frame.height 
       << ", ts=" << frame.timestamp_ns << "ns]";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const CIPOMessage& cipo) {
    os << "CIPO[id=" << cipo.frame_id;
    if (cipo.exists) {
        os << ", track=" << cipo.track_id 
           << ", level=" << cipo.class_id
           << ", dist=" << cipo.distance_m << "m"
           << ", vel=" << cipo.velocity_ms << "m/s";
        if (cipo.cut_in_detected) os << ", CUT-IN";
        if (cipo.kalman_reset) os << ", RESET";
    } else {
        os << ", NO CIPO";
    }
    os << "]";
    return os;
}

#endif // TRANSMISSION_DATA_HPP

