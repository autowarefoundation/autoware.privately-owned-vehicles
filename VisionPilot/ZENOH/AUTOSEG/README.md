# AUTOSEG

In the demonstration, IMAGE_PUBLISHER will publish the image to SCENE_SEG via Zenoh.
After processing with ONNX models, SCENE_SEG will publish the result to SEMANTIC_VISUALIZATION.
SEMANTIC_VISUALIZATION can be a simple Zenoh subscriber or Foxglove visualization tool.

![Zenoh Architecture](media/Zenoh_Architecture.svg)
