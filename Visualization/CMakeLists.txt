cmake_minimum_required(VERSION 3.20)
project(AutowareHMI CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets)
find_package(OpenCV REQUIRED)

include(FetchContent)
FetchContent_Declare(
    json 
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(json)

add_executable(AutowareHMI
    src/main.cpp
    src/window.cpp
    src/window.h
    src/io/openlane_loader.cpp
    src/io/openlane_loader.h
    src/visualization/annotation_visualizer.cpp
    src/visualization/annotation_visualizer.h
)

set_target_properties(AutowareHMI PROPERTIES
    AUTOMOC ON
    AUTORCC ON 
    AUTOUIC ON
)

target_link_libraries(AutowareHMI
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    ${OpenCV_LIBS}
    nlohmann_json::nlohmann_json
)

target_include_directories(AutowareHMI PRIVATE src)