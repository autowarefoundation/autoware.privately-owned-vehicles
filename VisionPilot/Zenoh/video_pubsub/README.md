# IMAGE_PUBSUB

The project demonstrates publishing/subscribing images/videos with Zenoh.

## Dependencies

* **OpenCV**: For image and video processing.
  * Ubuntu: `sudo apt install libopencv-dev`
* **Zenoh C library**: Required for the transportation.
  * Download from [the GitHub release](https://github.com/eclipse-zenoh/zenoh-c/releases)
  * You can also add the Eclipse repository for apt server.
  
    ```shell
    echo "deb [trusted=yes] https://download.eclipse.org/zenoh/debian-repo/ /" | sudo tee -a /etc/apt/sources.list > /dev/null
    sudo apt update
    sudo apt install libzenohc-dev
    ```

* **CLI11**: Used for the command line interface.
  * Ubuntu: `sudo apt install libcli11-dev`

## Build

* Configure with cmake

```shell
mkdir build && cd build
cmake ..
```

* Build

```shell
make
```

## Usage

* Publish the Zenoh video

```shell
./video_publisher <path_to_input_video.mp4>
# Assign the key
./video_publisher -k video/raw <path_to_input_video.mp4>
```

* Subscribe the Zenoh video

```shell
./video_subscriber
# Assign the key
./video_subscriber -k video/raw
```
