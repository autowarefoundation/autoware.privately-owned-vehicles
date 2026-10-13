# Dev/runtime image for the PoV_Highway demo: builds and runs VisionPilot (ROS 2 + Occupancy),
# the CARLA relays and the scenario. Sources are bind-mounted, nothing is baked in.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ARG ORT_VERSION=1.23.2
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake pkg-config git curl ca-certificates gnupg \
        libopencv-dev coinor-libipopt-dev liblapack-dev libblas-dev libcppad-dev \
        libboost-system-dev nlohmann-json3-dev \
        libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev \
        gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
        python3 python3-numpy python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# CppAD includes <coin-or/...>; Ubuntu 24.04 ships the Ipopt headers under coin/.
RUN [ -e /usr/include/coin-or ] || ln -s coin /usr/include/coin-or

# hadolint ignore=DL3008
RUN curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu noble main" \
        > /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-ros-base ros-jazzy-cv-bridge ros-jazzy-image-transport \
        ros-jazzy-ackermann-msgs ros-jazzy-rosidl-default-generators \
        python3-colcon-common-extensions python3-setuptools \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz" \
        -o /tmp/ort.tgz \
    && mkdir -p /opt/onnxruntime \
    && tar -xzf /tmp/ort.tgz -C /opt/onnxruntime --strip-components=1 \
    && rm /tmp/ort.tgz \
    && echo /opt/onnxruntime/lib > /etc/ld.so.conf.d/onnxruntime.conf \
    && ldconfig

ENV ONNXRUNTIME_ROOT=/opt/onnxruntime
WORKDIR /work
