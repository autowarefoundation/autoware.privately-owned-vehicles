# IMAGE_PUBSUB

The project demonstrates publishing/subscribing images/videos with Zenoh.

## Build

- Configure with cmake

```shell
mkdir build && cd build
cmake ..
```

- Build

```shell
make
```

## Usage

- Publish the Zenoh video

```shell
./video_publisher <path_to_input_video.mp4>
# Assign the key
./video_publisher -k video/raw <path_to_input_video.mp4>
```

- Subscribe the Zenoh video

```shell
./video_subscriber
# Assign the key
./video_subscriber -k video/raw
```
