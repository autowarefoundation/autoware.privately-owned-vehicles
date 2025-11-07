cmake_minimum_required(VERSION 3.16)
project(autospeed_standalone LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(OpenCV REQUIRED)

# GStreamer
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(GSTREAMER_APP REQUIRED gstreamer-app-1.0)

# yaml-cpp for calibration loading
find_package(yaml-cpp REQUIRED)

# ONNX Runtime - user must set ONNXRUNTIME_ROOT
if(NOT DEFINED ENV{ONNXRUNTIME_ROOT})
    message(FATAL_ERROR "Please set ONNXRUNTIME_ROOT environment variable to ONNX Runtime installation directory\n"
                        "Example: export ONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-gpu-1.22.0")
endif()

set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT})
set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_ROOT}/include)
set(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_ROOT}/lib)

# Find library (handle versioned .so files)
file(GLOB ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so*")
list(GET ONNXRUNTIME_LIBRARY 0 ONNXRUNTIME_LIBRARY)

if(NOT EXISTS ${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h)
    message(FATAL_ERROR "ONNX Runtime headers not found at: ${ONNXRUNTIME_INCLUDE_DIR}")
endif()

if(NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "ONNX Runtime library not found at: ${ONNXRUNTIME_LIB_DIR}")
endif()

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${GSTREAMER_INCLUDE_DIRS}
    ${GSTREAMER_APP_INCLUDE_DIRS}
    ../../common/include
)

# AutoSpeed ONNX Runtime backend
add_library(autospeed_backend
    ../../common/backends/autospeed/onnxruntime_engine.cpp
    ../../common/backends/autospeed/onnxruntime_session.cpp
)
target_compile_definitions(autospeed_backend PRIVATE LOG_TYPE=0)  # Use non-ROS logging
target_link_libraries(autospeed_backend
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARY}
)
# Add RPATH so executable can find libonnxruntime.so at runtime
set_target_properties(autospeed_backend PROPERTIES
    BUILD_RPATH ${ONNXRUNTIME_LIB_DIR}
    INSTALL_RPATH ${ONNXRUNTIME_LIB_DIR}
)

# GStreamer engine
add_library(gstreamer_engine
    ../../common/sensors/gstreamer_engine.cpp
)
target_compile_definitions(gstreamer_engine PRIVATE LOG_TYPE=0)  # Use non-ROS logging
target_link_libraries(gstreamer_engine
    ${OpenCV_LIBS}
    ${GSTREAMER_LIBRARIES}
    ${GSTREAMER_APP_LIBRARIES}
)

# FPS Timer
add_library(fps_timer
    ../../common/benchmark/fps_timer.cpp
)
target_compile_definitions(fps_timer PRIVATE LOG_TYPE=0)  # Use non-ROS logging

# ObjectFinder (tracking + distance/velocity estimation)
add_library(object_finder
    ../../common/tracking/object_finder.cpp
    ../../common/tracking/kalman_filter.cpp
    ../../common/tracking/tracking_utils.cpp
    ../../common/tracking/cipo_history.cpp
    ../../common/tracking/cipo_utils.cpp
    ../../common/tracking/feature_matching_utils.cpp
)
target_compile_definitions(object_finder PRIVATE LOG_TYPE=0)  # Use non-ROS logging
target_link_libraries(object_finder
    ${OpenCV_LIBS}
    yaml-cpp
)

# Main executable - infer_stream (bbox visualization with tracking)
add_executable(autospeed_infer_stream
    infer_stream.cpp
)
target_compile_definitions(autospeed_infer_stream PRIVATE LOG_TYPE=0)  # Use non-ROS logging
target_link_libraries(autospeed_infer_stream
    autospeed_backend
    gstreamer_engine
    fps_timer
    object_finder
    ${OpenCV_LIBS}
    pthread
)

# Installation
install(TARGETS autospeed_infer_stream
    RUNTIME DESTINATION bin
)

message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "ONNX Runtime: ${ONNXRUNTIME_ROOT}")
message(STATUS "  - Library: ${ONNXRUNTIME_LIBRARY}")
message(STATUS "  - Include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "GStreamer: ${GSTREAMER_VERSION}")

