# ONNX Model Quantization and Benchmarking Tools

This folder adds two new utilities for ONNX model optimization and performance analysis:

1. `quantize_onnx_model.py`: Converts FP32 ONNX models to INT8 quantized versions
2. `benchmark_onnx_models.py`: Benchmarks and compares PyTorch, ONNX FP32, and ONNX INT8 model performance

## Features

### Quantization Tool
- Converts FP32 models to INT8 using post-training static quantization
- Uses QDQ (QuantizeLinear/DeQuantizeLinear) format for optimal compatibility
- Supports per-channel quantization for weights
- Includes comprehensive calibration data handling
- Maintains float32 precision during preprocessing

### Benchmarking Tool
- Measures inference speed, model size, and memory usage
- Evaluates output accuracy using mean Intersection over Union (mIoU)
- Provides class-specific IoU metrics (Background, Foreground, Road)
- Uses model-centric evaluation approach (each model processes all samples)
- Ensures IoU calculation consistency with SceneSegTrainer methodology
- Supports visualization of model outputs with side-by-side comparisons
- Uses real validation data for accurate benchmarking
- Provides detailed performance metrics and statistics per model and dataset

## FP32 Model Pre-processing

Before quantization, it's recommended to pre-process your FP32 model to ensure optimal results. This involves several steps that improve quantization effectiveness:

1. **Symbolic Shape Inference**: Particularly helpful for transformer models
2. **ONNX Runtime Model Optimization**: Combines operations for better performance
3. **ONNX Shape Inference**: Ensures all tensor shapes are available for quantization

Pre-processing can be performed using ONNX Runtime's built-in tools:

```bash
python -m onnxruntime.quantization.preprocess \
    --input /path/to/model.onnx \
    --output /path/to/model-infer.onnx
```

For scene segmentation models, this pre-processing helps by:
- Merging operations like Convolution+BatchNormalization 
- Adding shape information required for accurate quantization
- Optimizing model structure for better inference performance

After pre-processing, the resulting model (with suffix "-infer") can be used as input for the quantization tool.

For more details on quantization techniques, see the [ONNX Runtime Quantization Examples](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/quantization/image_classification/cpu).

## Calibration Dataset Structure
The calibration tools for scene segmentation models expect the dataset to be organized in the following structure:

```
SceneSeg/
├── ACDC/
│   └── ACDC/
│       ├── images/
│       │   ├── image1.jpg
│       │   └── ...
│       └── gt_masks/
│           ├── mask1.png
│           └── ...
├── BDD100K/
│   └── BDD100K/
│       ├── images/
│       │   ├── image1.jpg
│       │   └── ...
│       └── gt_masks/
│           ├── mask1.png
│           └── ...
└── ... (other datasets)
```

Each dataset should contain RGB images and corresponding ground truth masks. The tools will automatically sample from these directories for calibration and benchmarking.

## Usage

### 1. Model Quantization

Convert your FP32 model to INT8:

```bash
python quantize_onnx_model.py \
    --input_model /path/to/fp32_model.onnx \
    --output_model /path/to/int8_model.onnx \
    --calibration_data_dir /path/to/datasets/
```

### 2. Model Benchmarking

Compare PyTorch, FP32, and INT8 model performance:

```bash
python benchmark_onnx_models.py \
    --pytorch_model_path /path/to/pytorch_model.pth \
    --fp32_model_path /path/to/fp32_model.onnx \
    --int8_model_path /path/to/int8_model.onnx \
    --dataset_root /path/to/datasets/ \
    --num_samples_per_dataset 20 \
    --visualizations_per_dataset 2 \
    --save_outputs \
    --device cuda
```

The benchmark tool will:
1. Load all specified models
2. Process samples from each available dataset 
3. Calculate IoU scores (overall and per-class)
4. Measure inference times
5. Generate combined visualizations (Original | GT | Prediction)
6. Produce a comprehensive comparison report

#### Optional Parameters:
- `--pytorch_target_hw` - Target height,width for PyTorch model (e.g., "320,640")
- `--onnx_target_hw` - Target height,width for ONNX models (derived from model if not specified)
- `--warmup_runs` - Number of warmup inference runs (default: 3)
- `--output_dir_base` - Directory for saving results (default: "./benchmark_multimodel_results")

## Benefits of INT8 Quantization

- **Reduce model size**: INT8 quantization typically reduces model size by 75%
- **Improve inference speed**: Up to 4x speedup on compatible hardware
- **Maintain accuracy**: Minimal impact on model output quality (verified with IoU metrics)
- **Standardize workflow**: Consistent process for quantizing and benchmarking segmentation models

## Dependencies
- onnx
- onnxruntime
- numpy
- opencv-python
- PIL (Pillow)
- torch (for PyTorch models)
- tqdm (optional, for progress bars)

## Testing
The tools have been tested with:
- Various scene segmentation models
- Different input sizes and resolutions
- Multiple validation datasets (ACDC, BDD100K, IDDAW, MUSES, etc.)
- Different hardware configurations (CPU and CUDA)

## Additional Utilities

### trace_pytorch_model.py
A script to load a pytorch model from among the AutoSeg Vision Foundation Model networks via a .pth checkpoint file, trace and then export the traced model as a *.pt file.

#### Example Usage
```bash
python3 trace_pytorch_model.py -n SceneSeg -p /path/to/SceneSeg/weights.pth -o /path/to/SceneSeg_Export/traced_model.pt
```

### convert_pytorch_to_onnx.py
A script to load a pytorch model from among the AutoSeg Vision Foundation Model networks and convert and export that model to ONNX format at FP32 precision as a *.onnx file.

#### Example Usage
```bash
python3 convert_pytorch_to_onnx.py -n SceneSeg -p /path/to/SceneSeg/weights.pth -o /path/to/SceneSeg_Export/converted_model.onnx
```

