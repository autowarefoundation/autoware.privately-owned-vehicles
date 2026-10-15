cmake_minimum_required(VERSION 3.22.1)

# Optional Occupancy BEV window (heuristic). OFF → stock HUD only.
option(ENABLE_OCCUPANCY "Build heuristic Occupancy BEV module" OFF)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(Boost 1.70 REQUIRED COMPONENTS system)

pkg_check_modules(WEBCONN IMPORTED_TARGET
        gstreamer-1.0
        gstreamer-app-1.0
        gstreamer-sdp-1.0
        gstreamer-webrtc-1.0
)

find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp)

if (WEBCONN_FOUND AND Boost_FOUND AND NLOHMANN_JSON_INCLUDE_DIR)
    set(WEBRTC_DEPS_FOUND TRUE)
    list(APPEND VIS_SOURCES src/webrtc_stream.cpp)
else ()
    set(WEBRTC_DEPS_FOUND FALSE)
    message(WARNING "ENABLE_WEBRTC=ON but required WebRTC dependencies not found; building without WebRTC")
endif ()


add_library(visualization
        ../common/include/common/types.hpp
        src/visual_interface.cpp
        src/local_display.cpp
#        src/debug_draw.cpp
        ${VIS_SOURCES}
)

if(BUILD_TESTING)
    target_sources(visualization PRIVATE
            src/debug_display.cpp)
    target_compile_definitions(visualization PUBLIC BUILD_TESTING)
endif()

if (ENABLE_OCCUPANCY)
    target_sources(visualization PRIVATE
        src/occupancy_view.cpp
        src/occupancy_bridge.cpp
    )
    target_compile_definitions(visualization PUBLIC ENABLE_OCCUPANCY=1)
endif ()

if (WEBRTC_DEPS_FOUND)
    target_compile_definitions(visualization PUBLIC ENABLE_WEBRTC=1 PRIVATE GST_USE_UNSTABLE_API=1)
    target_include_directories(visualization PRIVATE
            ${Boost_INCLUDE_DIRS}
            ${NLOHMANN_JSON_INCLUDE_DIR}
    )
    target_link_libraries(visualization
            PUBLIC
            PkgConfig::WEBCONN
            PRIVATE
            Boost::system
    )
endif ()

target_include_directories(visualization
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(visualization
        PUBLIC
        ${OpenCV_LIBRARIES}
        Threads::Threads
        models
        common
)
