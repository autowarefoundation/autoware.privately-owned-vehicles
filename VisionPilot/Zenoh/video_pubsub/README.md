# IMAGE_PUBSUB

The project demonstrates publishing/subscribing images/videos with Zenoh.

## Usage

- Switch to the install folder

- Publish the Zenoh video

```shell
./video_publisher ../../data/video.mp4
# Assign the key
./video_publisher -k video/raw ../../data/video.mp4
```

- Subscribe the Zenoh video

```shell
./video_subscriber
# Assign the key
./video_subscriber -k video/raw
```
