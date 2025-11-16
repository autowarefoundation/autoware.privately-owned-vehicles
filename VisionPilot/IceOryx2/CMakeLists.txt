cmake_minimum_required(VERSION 3.16)
project(VisionPilot_IceOryx2 VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# Find Dependencies
# ============================================================================

# iceoryx2
find_package(iceoryx2-cxx REQUIRED)


# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# GStreamer
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(GSTREAMER_APP REQUIRED gstreamer-app-1.0)
include_directories(${GSTREAMER_INCLUDE_DIRS} ${GSTREAMER_APP_INCLUDE_DIRS})

# yaml-cpp for homography loading
find_package(yaml-cpp REQUIRED)

# ONNX Runtime (from environment variable)
if(NOT DEFINED ENV{ONNXRUNTIME_ROOT})
    message(FATAL_ERROR "ONNXRUNTIME_ROOT environment variable not set. Please export it.")
endif()

set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT})
set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_ROOT}/include)
set(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_ROOT}/lib)
set(ONNXRUNTIME_LIBRARY ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so)

include_directories(${ONNXRUNTIME_INCLUDE_DIR})

message(STATUS "=====================================")
message(STATUS "  VisionPilot IceOryx2 Build Config")
message(STATUS "=====================================")
message(STATUS "OpenCV:        ${OpenCV_VERSION} at ${OpenCV_DIR}")
message(STATUS "GStreamer:     ${GSTREAMER_VERSION}")
message(STATUS "ONNX Runtime:  ${ONNXRUNTIME_ROOT}")
message(STATUS "iceoryx2:      Found")
message(STATUS "=====================================")

# ============================================================================
# Common Library (shared across all nodes)
# ============================================================================

add_library(visionpilot_common SHARED
    # Sensors
    ../common/sensors/gstreamer_engine.cpp
    
    # Tracking
    ../common/tracking/object_finder.cpp
    ../common/tracking/kalman_filter.cpp
    ../common/tracking/tracking_utils.cpp
    ../common/tracking/cipo_history.cpp
    ../common/tracking/cipo_utils.cpp
    ../common/tracking/feature_matching_utils.cpp
    
    # Backends
    ../common/backends/autospeed/onnxruntime_engine.cpp
    ../common/backends/autospeed/onnxruntime_session.cpp
)

target_include_directories(visionpilot_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/backends/autospeed
    ${OpenCV_INCLUDE_DIRS}
    ${GSTREAMER_INCLUDE_DIRS}
    ${GSTREAMER_APP_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
)

# Disable ROS2 logging for standalone build (use printf instead)
target_compile_definitions(visionpilot_common PUBLIC LOG_TYPE=0)

target_link_libraries(visionpilot_common
    ${OpenCV_LIBS}
    ${GSTREAMER_LIBRARIES}
    ${GSTREAMER_APP_LIBRARIES}
    ${ONNXRUNTIME_LIBRARY}
    yaml-cpp
)

# Set RPATH for runtime library discovery
set_target_properties(visionpilot_common PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}"
    INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
)

# ============================================================================
# Node 1: Frame Publisher
# ============================================================================

add_executable(frame_node frame_node.cpp)

target_link_libraries(frame_node
    iceoryx2-cxx::static-lib-cxx
    visionpilot_common
    ${OpenCV_LIBS}
    ${GSTREAMER_LIBRARIES}
    ${GSTREAMER_APP_LIBRARIES}
)

set_target_properties(frame_node PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}"
    INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
)

# ============================================================================
# Node 2: Inference + Tracking
# ============================================================================

add_executable(inference_node inference_node.cpp)

target_link_libraries(inference_node
    iceoryx2-cxx::static-lib-cxx    # Use static library
    visionpilot_common
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARY}
)

set_target_properties(inference_node PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}"
    INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
)

# ============================================================================
# Node 3: Visualization (optional)
# ============================================================================

add_executable(viz_node viz_node.cpp)

target_link_libraries(viz_node
    iceoryx2-cxx::static-lib-cxx
    ${OpenCV_LIBS}
)

# ============================================================================
# Install Targets
# ============================================================================

install(TARGETS frame_node inference_node viz_node visionpilot_common
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)

message(STATUS "=====================================")
message(STATUS "Build targets:")
message(STATUS "  - frame_node       (Frame publisher)")
message(STATUS "  - inference_node   (Inference + tracking)")
message(STATUS "  - viz_node         (Visualization)")
message(STATUS "=====================================")

