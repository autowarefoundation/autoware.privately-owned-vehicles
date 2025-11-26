FROM ubuntu:24.04

# x64 or aarch64
ARG ARCH=x64 
ARG ONNXRUNTIME_VERSION=1.22.0

# Install visionpilot dependencies
WORKDIR /visionpilot
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    x11-apps \
    build-essential cmake pkg-config git wget curl nano \
    libopencv-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libyaml-cpp-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install model dependencies
COPY Models /visionpilot/Models
ENV PYTHONPATH=/visionpilot/Models
RUN pip install --no-cache-dir -r /visionpilot/Models/requirements.txt --break-system-packages \
    && pip install --no-cache-dir --break-system-packages gdown 

# Install onnxruntime
COPY VisionPilot /visionpilot/VisionPilot
ENV ONNXRUNTIME_ROOT=/visionpilot/onnxruntime-linux-${ARCH}-${ONNXRUNTIME_VERSION}
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-${ARCH}-${ONNXRUNTIME_VERSION}.tgz && \
    tar -xzf onnxruntime-linux-${ARCH}-${ONNXRUNTIME_VERSION}.tgz && \
    rm onnxruntime-linux-${ARCH}-${ONNXRUNTIME_VERSION}.tgz

# Build AutoSpeed
RUN cd /visionpilot/VisionPilot/Standalone/AutoSpeed && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    mv /visionpilot/VisionPilot/Standalone/AutoSpeed/build/autospeed_infer_stream /visionpilot/autospeed_infer_stream

# Set the entrypoint
COPY Modules/SDV/OpenADKit/Docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
