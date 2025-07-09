cmake_minimum_required(VERSION 3.4 FATAL_ERROR)
project(deploy_libtorch)

## These should be variables (TODO)
set(USE_CUDNN OFF)
set(CAFFE2_USE_CUDNN OFF)
set(USE_CUSPARSELT OFF)

## Find libTorch installation
find_package(Torch REQUIRED HINTS ${LIBTORCH_INSTALL_ROOT}/share/cmake/Torch/ QUIET)
find_package(OpenCV REQUIRED HINTS ${OPENCV_INSTALL_ROOT}/build/ QUIET)

include_directories(
  ${TORCH_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

## Build & Link
add_executable(deploy_libtorch main.cpp)
target_link_libraries(deploy_libtorch ${OpenCV_LIBS} ${TORCH_LIBRARIES} opencv_imgcodecs)