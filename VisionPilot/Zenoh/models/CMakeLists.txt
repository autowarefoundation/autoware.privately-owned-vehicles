cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(zenoh_models)

set(USE_CUDNN OFF)
set(CAFFE2_USE_CUDNN OFF)
set(USE_CUSPARSELT OFF)

if(USE_CUDA_BACKEND)
    add_definitions(-DUSE_EP_CUDA)
    message(STATUS "Using CUDA Execution Provider Backend")
elseif(USE_DNNL_BACKEND)
    add_definitions(-DUSE_EP_DNNL)
    message(STATUS "Using DNNL Execution Provider Backend")
endif()

## Find OpenCV installation
find_package(OpenCV REQUIRED HINTS ${OPENCV_INSTALL_ROOT}/build/ QUIET)
find_package(Torch REQUIRED HINTS ${LIBTORCH_INSTALL_ROOT}/share/cmake/Torch/ QUIET)
find_package(zenohc REQUIRED)

string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra")
string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")

## ONNX Runtime Providers
option(onnxruntime_USE_CUDA "Build with CUDA support" OFF)
option(onnxruntime_USE_TENSORRT "Build with TensorRT support" OFF)
option(LIBPNG_ROOTDIR "libpng root dir")
option(ONNXRUNTIME_ROOTDIR "onnxruntime root dir")
set(CMAKE_CXX_STANDARD 17)

if(NOT ONNXRUNTIME_ROOTDIR)
    message(STATUS "You must provide an onnx runtime root directory!")
endif()

# set the MODEL_ROOTDIR variable to the path where the model is located
set(MODEL_ROOTDIR "${CMAKE_CURRENT_SOURCE_DIR}/../../common")

# The ORT package has a different include directory structure to a local install via cmake.
# We added the path for the pre-built package above. Add the path for a local install to support either usage.
# TODO: If we want to support additional EPs being loadable from a local install we also need to add EP specific
# directories under /include/onnxruntime/core/providers
include_directories(${ONNXRUNTIME_ROOTDIR}/include
                    ${ONNXRUNTIME_ROOTDIR}/include/onnxruntime
                    ${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session
                    ${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/providers/dnnl    # DNNL Backend Support
                    ${OpenCV_INCLUDE_DIRS}                                            # OpenCV
                    ${TORCH_INCLUDE_DIRS}                                             # Torch
                    ${MODEL_ROOTDIR}/include                                          # Common libraries
                   )

link_directories("${ONNXRUNTIME_ROOTDIR}/lib")

if(onnxruntime_USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif()

## Build & Link
add_executable(run_model run_model.cpp 
                         #${MODEL_ROOTDIR}/backends/tensorrt_backend.cpp
                         ${MODEL_ROOTDIR}/backends/onnx_runtime_backend.cpp
                         ${MODEL_ROOTDIR}/visualizers/masks_visualization_engine.cpp
                         ${MODEL_ROOTDIR}/visualizers/depth_visualization_engine.cpp
              )
target_compile_definitions(run_model PRIVATE LOG_TYPE=0)
target_link_libraries(run_model PRIVATE onnxruntime ${TORCH_LIBRARIES} ${OpenCV_LIBS} opencv_videoio opencv_imgproc opencv_highgui zenohc::lib)
