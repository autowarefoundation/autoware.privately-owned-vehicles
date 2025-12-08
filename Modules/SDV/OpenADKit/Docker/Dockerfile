FROM ubuntu:24.04

# x64 or aarch64
ARG ARCH=x64 
ARG ONNXRUNTIME_VERSION=1.22.0

# Install visionpilot dependencies
WORKDIR /autoware
ARG DEBIAN_FRONTEND=noninteractive
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
COPY Models /autoware/Models
ENV PYTHONPATH=/autoware/Models
RUN pip install --no-cache-dir -r /autoware/Models/requirements.txt --break-system-packages \
    && pip install --no-cache-dir --break-system-packages gdown 

# Install onnxruntime
COPY VisionPilot /autoware/VisionPilot
ENV ONNXRUNTIME_ROOT=/autoware/onnxruntime-linux-${ARCH}-${ONNXRUNTIME_VERSION}
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-${ARCH}-${ONNXRUNTIME_VERSION}.tgz && \
    tar -xzf onnxruntime-linux-${ARCH}-${ONNXRUNTIME_VERSION}.tgz && \
    rm onnxruntime-linux-${ARCH}-${ONNXRUNTIME_VERSION}.tgz

# Build AutoSpeed
RUN cd /autoware/VisionPilot/Middleware_Recipes/Standalone/AutoSpeed && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    mv /autoware/VisionPilot/Middleware_Recipes/Standalone/AutoSpeed/build/autospeed_infer_stream /autoware/autospeed_infer_stream

# Add Visualizer: Install X11, window manager, VNC, and NoVNC
RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb x11-xserver-utils x11-utils x11-apps dbus-x11 \
    openbox obconf menu \
    tigervnc-standalone-server tigervnc-tools \
    novnc websockify \
    openssl \
    curl wget ca-certificates procps net-tools supervisor \
    fonts-liberation fonts-dejavu-core \
    python3 python3-numpy \
    && rm -rf /var/lib/apt/lists/*

# Create SSL certificate for NoVNC
RUN openssl req -x509 -nodes -newkey rsa:2048 \
    -keyout /etc/ssl/private/novnc.key \
    -out /etc/ssl/certs/novnc.crt \
    -days 365 \
    -subj "/O=GUI-Visualizer/CN=localhost"

# Create working directory
WORKDIR /app

# Expose VNC and NoVNC ports
# 5900: VNC direct connection
# 6080: NoVNC web interface
EXPOSE 5900 6080

# Copy startup scripts
COPY Modules/SDV/OpenADKit/Docker/xstartup /root/.vnc/xstartup
COPY Modules/SDV/OpenADKit/Docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh /root/.vnc/xstartup

# Environment defaults
ENV DISPLAY=:99 \
    VNC_GEOMETRY=1280x720 \
    VNC_DEPTH=24 \
    VNC_PASSWORD=visualizer \
    GUI_APP=""

ENTRYPOINT ["/entrypoint.sh"]
