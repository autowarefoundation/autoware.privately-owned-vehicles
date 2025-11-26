cmake_minimum_required(VERSION 3.16)
project(autosteer_pipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

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
)

# AutoSteer inference backend
add_library(autosteer_inference
    src/inference/onnxruntime_session.cpp
    src/inference/onnxruntime_engine.cpp
)
target_compile_definitions(autosteer_inference PRIVATE LOG_TYPE=0)  # Use non-ROS logging
target_link_libraries(autosteer_inference
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARY}
)
# Add RPATH so executable can find libonnxruntime.so at runtime
set_target_properties(autosteer_inference PROPERTIES
    BUILD_RPATH ${ONNXRUNTIME_LIB_DIR}
    INSTALL_RPATH ${ONNXRUNTIME_LIB_DIR}
)

# Visualization module
add_library(autosteer_visualization
    src/visualization/draw_lanes.cpp
)
target_compile_definitions(autosteer_visualization PRIVATE LOG_TYPE=0)  # Use non-ROS logging
target_link_libraries(autosteer_visualization
    autosteer_inference
    ${OpenCV_LIBS}
)

# Lane filtering module
add_library(autosteer_lane_filtering
    src/lane_filtering/lane_filter.cpp
)
target_link_libraries(autosteer_lane_filtering
    autosteer_inference
)

# Main executable
add_executable(autosteer_pipeline
    main.cpp
)
target_compile_definitions(autosteer_pipeline PRIVATE LOG_TYPE=0)  # Use non-ROS logging
target_link_libraries(autosteer_pipeline
    autosteer_inference
    autosteer_visualization
    autosteer_lane_filtering
    ${OpenCV_LIBS}
    pthread
)

# Installation
install(TARGETS autosteer_pipeline
    RUNTIME DESTINATION bin
)

message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "ONNX Runtime: ${ONNXRUNTIME_ROOT}")
message(STATUS "  - Library: ${ONNXRUNTIME_LIBRARY}")
message(STATUS "  - Include: ${ONNXRUNTIME_INCLUDE_DIR}")

