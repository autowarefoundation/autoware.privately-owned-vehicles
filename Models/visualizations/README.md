# Visualizations
Contains various scripts to run inference using trained models and create visualizations, including single image visualzation, video visualization and if applicable, pointcloud visualization

# Examples

## image_visualization.py
Read an image using OpenCV, run the network and visualize the output and display in an OpenCV window

### Example usage
```bash
  python3 Scene3D/image_visualization.py -p /path/to/Scene3D/weights.pth -i /path/to/image.jpg
```

```bash
  python3 SceneSeg/image_visualization.py -p /path/to/SceneSeg/weights.pth -i /path/to/image.jpg
```

```bash
  python3 DomainSeg/image_visualization.py -p /path/to/DomainSeg/weights.pth -i /path/to/image.jpg
```

```bash
  python3 EgoSpace/image_visualization.py -p /path/to/EgoSpace/weights.pth -i /path/to/image.jpg
```
### Parameters:

*-p , --model_checkpoint_path* : path to model weights *.pth* file

*-i , --input_image_filepath* : path input image to be processed and visualized (.jpg and .png formats supported)

## video_visualization.py
Read a video file using OpenCV, run the network and visualize the output and save the visualization frames as a video in *.avi* format, optionally display in-progress frames in an OpenCV window

### Example usage
```bash
  python3 Scene3D/video_visualization.py -p /path/to/Scene3D/weights.pth -i /path/to/raw_video.mp4 -o /path/to/output/video_name -v
```

```bash
  python3 SceneSeg/video_visualization.py -p /path/to/SceneSeg/weights.pth -i /path/to/raw_video.mp4 -o /path/to/output/video_name -v
```

```bash
  python3 DomainSeg/video_visualization.py -p /path/to/DomainSeg/weights.pth -i /path/to/raw_video.mp4 -o /path/to/output/video_name -v
```

```bash
  python3 EgoSpace/video_visualization.py -p /path/to/DomainSeg/weights.pth -i /path/to/raw_video.mp4 -o /path/to/output/video_name -v
```
### Parameters:

*-p , --model_checkpoint_path* : path to model weights *.pth* file

*-i , --video_filepath* : path to input video which will be processed

*-o , --output_file* : path to output video visualization file, must include output file name

*-v , --vis* : flag for whether to show frame by frame visualization while processing is occuring