# Zenoh

This directory provides a Zenoh-based CARLA bridge and related pipelines.

## Dependencies

Please refer to the dependencies described in the following page:

[VisionPilot/Middleware_Recipes/Zenoh](../../../../VisionPilot/Middleware_Recipes/Zenoh)

## Usage

### Setup (run once)

```sh
just setup
```

### Build

```sh
# Build all
export LIBTORCH_INSTALL_ROOT=/path/to/libtorch/
export ONNXRUNTIME_ROOTDIR=/path/to/onnxruntime-linux-x64-gpu-1.22.0
just build
# Optional (build components separately)
just build_bridge
just build_video_pubsub
just build_models
```

### Run

#### Start CARLA server

```sh
just run_carla
```

Start the CARLA simulator.
The Docker image may take a long time to download on the first run.

#### Run pipelines

```sh
# Original video pub/sub
just run_video_pubsub

# SceneSeg
just run_sceneseg

# DomainSeg
just run_domainseg
```

### Cleanup

```sh
just clean
```

