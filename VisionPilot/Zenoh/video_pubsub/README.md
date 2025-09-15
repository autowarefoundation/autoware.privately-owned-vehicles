# IMAGE_PUBSUB

The project demonstrates publishing/subscribing images/videos with Zenoh.

## Build

- Configure with cmake

```shell
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
```

- Build

```shell
make
```

## Usage

- Publish the Zenoh video

```shell
./video_publisher ../../../data/video.mp4
# Assign the key
./video_publisher -k video/raw ../../../data/video.mp4
```

- Subscribe the Zenoh video

```shell
./video_subscriber
# Assign the key
./video_subscriber -k video/raw
```
