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
    include
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

# Camera utilities module
add_library(camera_utils
    src/camera/camera_utils.cpp
)
target_link_libraries(camera_utils
    ${OpenCV_LIBS}
)

# Rerun SDK (optional, for visualization/logging)
option(ENABLE_RERUN "Enable Rerun logging and visualization" OFF)

if(ENABLE_RERUN)
    if(NOT DEFINED ENV{RERUN_SDK_ROOT})
        message(WARNING "RERUN_SDK_ROOT not set. Disabling Rerun support.\n"
                        "To enable: export RERUN_SDK_ROOT=/path/to/rerun_cpp_sdk\n"
                        "Download from: https://github.com/rerun-io/rerun/releases")
        set(ENABLE_RERUN OFF)
    else()
        set(RERUN_SDK_ROOT $ENV{RERUN_SDK_ROOT})
        
        # Check if Rerun SDK exists
        if(NOT EXISTS ${RERUN_SDK_ROOT}/include/rerun.hpp)
            message(WARNING "Rerun SDK not found at ${RERUN_SDK_ROOT}. Disabling Rerun support.")
            set(ENABLE_RERUN OFF)
        else()
            message(STATUS "Rerun SDK: ${RERUN_SDK_ROOT}")
            include_directories(${RERUN_SDK_ROOT}/include)
            
            # Find pre-built rerun_c library
            file(GLOB RERUN_C_LIBRARY "${RERUN_SDK_ROOT}/lib/librerun_c.*")
            if(RERUN_C_LIBRARY)
                list(GET RERUN_C_LIBRARY 0 RERUN_C_LIBRARY)
                message(STATUS "  - Library: ${RERUN_C_LIBRARY}")
                
                # Also need Arrow library (bundled with SDK or system)
                find_library(ARROW_LIBRARY arrow PATHS ${RERUN_SDK_ROOT}/lib NO_DEFAULT_PATH)
                if(NOT ARROW_LIBRARY)
                    find_library(ARROW_LIBRARY arrow)
                endif()
                
                if(ARROW_LIBRARY)
                    message(STATUS "  - Arrow: ${ARROW_LIBRARY}")
                else()
                    message(WARNING "Arrow library not found. Rerun may not work correctly.")
                endif()
                
                # Rerun logger module
                add_library(rerun_logger
                    src/rerun/rerun_logger.cpp
                )
                target_compile_definitions(rerun_logger PRIVATE ENABLE_RERUN)
                target_link_libraries(rerun_logger
                    autosteer_inference
                    ${OpenCV_LIBS}
                    ${RERUN_C_LIBRARY}
                    ${ARROW_LIBRARY}
                )
                set_target_properties(rerun_logger PROPERTIES
                    BUILD_RPATH ${RERUN_SDK_ROOT}/lib
                    INSTALL_RPATH ${RERUN_SDK_ROOT}/lib
                )
                
                set(RERUN_ENABLED TRUE)
            else()
                message(WARNING "Rerun C library not found. Disabling Rerun support.")
                set(ENABLE_RERUN OFF)
            endif()
        endif()
    endif()
endif()

# Main executable
add_executable(autosteer_pipeline
    main.cpp
)
target_compile_definitions(autosteer_pipeline PRIVATE LOG_TYPE=0)  # Use non-ROS logging

# Link libraries
set(AUTOSTEER_LIBS
    autosteer_inference
    autosteer_visualization
    autosteer_lane_filtering
    camera_utils
    ${OpenCV_LIBS}
    pthread
)

if(ENABLE_RERUN AND RERUN_ENABLED)
    target_compile_definitions(autosteer_pipeline PRIVATE ENABLE_RERUN)
    list(APPEND AUTOSTEER_LIBS rerun_logger)
    message(STATUS "Rerun support: ENABLED")
else()
    message(STATUS "Rerun support: DISABLED")
endif()

target_link_libraries(autosteer_pipeline ${AUTOSTEER_LIBS})

# Installation
install(TARGETS autosteer_pipeline
    RUNTIME DESTINATION bin
)

message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "ONNX Runtime: ${ONNXRUNTIME_ROOT}")
message(STATUS "  - Library: ${ONNXRUNTIME_LIBRARY}")
message(STATUS "  - Include: ${ONNXRUNTIME_INCLUDE_DIR}")

