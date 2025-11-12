FROM ubuntu:24.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    x11-apps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY Models /visionpilot/Models
RUN pip install --no-cache-dir -r /visionpilot/Models/requirements.txt --break-system-packages \
    && pip install --no-cache-dir --break-system-packages gdown 

# Set the Python path
ENV PYTHONPATH=/visionpilot/Models

# Set the entrypoint
COPY SDV/OpenADKit/docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
