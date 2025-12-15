cmake_minimum_required(VERSION 3.16)
project(egolanes_pipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)

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
    include
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

# EgoLanes inference backend
add_library(egolanes_inference
    src/inference/onnxruntime_session.cpp
    src/inference/onnxruntime_engine.cpp
    src/inference/autosteer_engine.cpp
)
target_compile_definitions(egolanes_inference PRIVATE LOG_TYPE=0)  # Use non-ROS logging
target_link_libraries(egolanes_inference
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARY}
)
# Add RPATH so executable can find libonnxruntime.so at runtime
set_target_properties(egolanes_inference PROPERTIES
    BUILD_RPATH ${ONNXRUNTIME_LIB_DIR}
    INSTALL_RPATH ${ONNXRUNTIME_LIB_DIR}
)

# Visualization module
add_library(egolanes_visualization
    src/visualization/draw_lanes.cpp
)
target_compile_definitions(egolanes_visualization PRIVATE LOG_TYPE=0)  # Use non-ROS logging
target_link_libraries(egolanes_visualization
    egolanes_inference
    ${OpenCV_LIBS}
)

# Lane filtering module
add_library(egolanes_lane_filtering
    src/lane_filtering/lane_filter.cpp
)
target_link_libraries(egolanes_lane_filtering
    egolanes_inference
    ${OpenCV_LIBS}
)

# Camera utilities module
add_library(camera_utils
    src/camera/camera_utils.cpp
)
target_link_libraries(camera_utils
    ${OpenCV_LIBS}
)

# Path planning module (polynomial fitting + Bayes filter)
add_library(path_planning
    src/path_planning/estimator.cpp
    src/path_planning/poly_fit.cpp
    src/path_planning/path_finder.cpp
)
target_link_libraries(path_planning
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

# Steering control module
add_library(steering_control
    src/steering_control/steering_controller.cpp
        src/steering_control/steering_filter.cpp
        include/steering_control/steering_filter.hpp
)
target_link_libraries(steering_control
    # No external dependencies, pure C++ math
)

# CAN Drivers module
add_library(can_drivers
    src/drivers/can_interface.cpp
)
target_link_libraries(can_drivers
    # Uses standard Linux headers (socketcan)
)

# Rerun SDK (optional, for visualization/logging)
option(ENABLE_RERUN "Enable Rerun logging and visualization" OFF)

if(ENABLE_RERUN)
    message(STATUS "Rerun support: ENABLED - Downloading Rerun SDK via FetchContent...")
    
    # Try to find system-installed Arrow first (avoids building from source)
    # This makes the build faster and more reliable
    set(CMAKE_PREFIX_PATH "/usr/local;/usr;${CMAKE_PREFIX_PATH}")
    find_package(Arrow QUIET)
    
    # Download Rerun SDK using FetchContent (official method)
    # This will download pre-built Rerun C static libraries and C++ sources
    include(FetchContent)
    FetchContent_Declare(rerun_sdk 
        URL https://github.com/rerun-io/rerun/releases/latest/download/rerun_cpp_sdk.zip
    )
    
    # Configure Rerun to use system Arrow if available (faster, more reliable)
    # Otherwise, it will download and build Arrow automatically
    if(Arrow_FOUND)
        message(STATUS "  - Found system Arrow: ${Arrow_DIR}")
        message(STATUS "  - Using system Arrow (skipping build from source)")
        set(RERUN_DOWNLOAD_AND_BUILD_ARROW OFF CACHE BOOL "Use system Arrow" FORCE)
    else()
        message(STATUS "  - System Arrow not found")
        message(STATUS "  - Will download and build Arrow automatically (may take 10-20 minutes)")
        message(STATUS "  - Tip: Install Arrow for faster builds: sudo apt-get install libarrow-dev")
        set(RERUN_DOWNLOAD_AND_BUILD_ARROW ON CACHE BOOL "Download and build Arrow" FORCE)
    endif()
    
    # Make Rerun SDK available
    # If system Arrow found, uses it. Otherwise downloads and builds Arrow.
    FetchContent_MakeAvailable(rerun_sdk)
    
    # Check if rerun_sdk target was created successfully
    if(TARGET rerun_sdk)
        message(STATUS "Rerun SDK: Successfully downloaded and configured")
        
        # Rerun logger module
        add_library(rerun_logger
            src/rerun/rerun_logger.cpp
        )
        target_compile_definitions(rerun_logger PRIVATE ENABLE_RERUN)
        target_link_libraries(rerun_logger
            egolanes_inference
            ${OpenCV_LIBS}
            rerun_sdk  # Link against rerun_sdk (official method)
        )
        
        set(RERUN_ENABLED TRUE)
        message(STATUS "Rerun support: ENABLED")
    else()
        # Arrow build likely failed - provide helpful error message
        message(FATAL_ERROR "\n"
            "=================================================================\n"
            "Failed to configure Rerun SDK (Arrow build failed).\n"
            "=================================================================\n"
            "\n"
            "Arrow build from source is complex and often fails.\n"
            "\n"
            "QUICK FIX - Install Arrow via package manager:\n"
            "  sudo apt-get update\n"
            "  sudo apt-get install -y libarrow-dev libparquet-dev\n"
            "\n"
            "Then clean and rebuild:\n"
            "  rm -rf build/_deps/rerun_sdk-build\n"
            "  cd build && cmake -DENABLE_RERUN=ON .. && make\n"
            "\n"
            "After installing Arrow, Rerun will automatically detect and use it.\n"
            "\n"
            "Build logs: build/_deps/rerun_sdk-build/arrow/\n"
            "=================================================================\n"
        )
    endif()
else()
    message(STATUS "Rerun support: DISABLED (use -DENABLE_RERUN=ON to enable)")
endif()

# Lane tracking module
add_library(egolanes_lane_tracking
    src/lane_tracking/lane_tracking.cpp
)
target_link_libraries(egolanes_lane_tracking
    egolanes_inference
    ${OpenCV_LIBS}
)

# Main executable
file(COPY ${CMAKE_SOURCE_DIR}/images
        DESTINATION ${CMAKE_BINARY_DIR})

add_executable(egolanes_pipeline
    main.cpp
)
target_compile_definitions(egolanes_pipeline PRIVATE LOG_TYPE=0)  # Use non-ROS logging

# Link libraries
set(EGOLANES_LIBS
    egolanes_inference
    egolanes_visualization
    egolanes_lane_filtering
    egolanes_lane_tracking
    camera_utils
    path_planning
    steering_control
    can_drivers
    ${OpenCV_LIBS}
    pthread
)

if(ENABLE_RERUN AND RERUN_ENABLED)
    target_compile_definitions(egolanes_pipeline PRIVATE ENABLE_RERUN)
    list(APPEND EGOLANES_LIBS rerun_logger)
endif()

target_link_libraries(egolanes_pipeline ${EGOLANES_LIBS})

# Test executable for AutoSteer
add_executable(test_autosteer
    test/test_autosteer.cpp
)
target_compile_definitions(test_autosteer PRIVATE LOG_TYPE=0)
target_link_libraries(test_autosteer ${EGOLANES_LIBS})

# Installation
install(TARGETS egolanes_pipeline
    RUNTIME DESTINATION bin
)

message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "ONNX Runtime: ${ONNXRUNTIME_ROOT}")
message(STATUS "  - Library: ${ONNXRUNTIME_LIBRARY}")
message(STATUS "  - Include: ${ONNXRUNTIME_INCLUDE_DIR}")

