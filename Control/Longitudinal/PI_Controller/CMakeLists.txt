cmake_minimum_required(VERSION 3.10)
project(PI_Controller)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

add_executable(PI_Controller
    src/main.cpp
    src/pi_controller.cpp
)

add_library(pi_controller_lib
    src/pi_controller.cpp
)
target_include_directories(pi_controller_lib PUBLIC include)