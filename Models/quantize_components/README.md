# PyTorch 2.0 (PT2E) Quantization Workflow for Scene Segmentation

This document outlines the end-to-end workflow for training a scene segmentation model using Quantization-Aware Training (QAT) and converting it to a final, optimized INT8 ONNX model. The process uses the modern PyTorch 2.0 Export (PT2E) framework.

The pipeline consists of two main scripts:
1.  `train_scene_seg_qat.py`: Fine-tunes a pre-trained FP32 model with QAT to make it robust to quantization errors.
2.  `benchmark_qat_model.py`: Takes the QAT-trained checkpoint, performs calibration, converts it to a final INT8 ONNX model, and runs a validation benchmark.

---

## Prerequisites
- A prepared dataset (ACDC, IDDAW, etc.) located in a root directory.
- A pre-trained FP32 model checkpoint for the `SceneSegNetwork`.
- A Python environment with `torch`, `onnx`, `onnxruntime`, `opencv-python`, and `tqdm`.

---

## Workflow

### Step 1: Quantization-Aware Training (QAT)

This step fine-tunes the model to learn weights that are resilient to quantization, which helps minimize accuracy loss after conversion.

**Script**: `train_scene_seg_qat.py`

**Usage**:
```bash
python3 autoware.privately-owned-vehicles/Models/quantize_components/train_scene_seg_qat.py \\
    --model_save_root_path /path/to/save/checkpoints/ \\
    --root /path/to/dataset/root/ \\
    --fp32_model /path/to/your/pretrained_fp32_model.pth \\
    --epochs 5
```

**Key Arguments**:
- `--model_save_root_path`: Directory where the QAT checkpoints will be saved.
- `--root`: The root directory containing all your datasets (e.g., ACDC, IDDAW).
- `--fp32_model`: Path to the starting FP32 model weights.
- `--epochs`: Number of epochs to fine-tune the model.

**Output**:
This script will produce one or more QAT checkpoint files (e.g., `qat_checkpoint_epoch_4_step_31057.pth`) in your specified save path. You should use the final or best-performing checkpoint for the next step.

---

### Step 2: Convert to INT8 ONNX and Validate

This script takes the QAT checkpoint, performs calibration to determine optimal quantization parameters, converts it into a final INT8 ONNX model, and runs a benchmark to validate its performance.

**Script**: `benchmark_qat_model.py`

**Usage**:
```bash
python3 autoware.privately-owned-vehicles/Models/quantize_components/benchmark_qat_model.py \\
    --qat_checkpoint_path /path/to/save/checkpoints/qat_checkpoint_epoch_4_step_31057.pth \\
    --dataset_root /path/to/dataset/root/
```

**Key Arguments**:
- `--qat_checkpoint_path`: Path to the QAT checkpoint generated in Step 1.
- `--dataset_root`: The root directory for your datasets, used for calibration and validation.
- `--num_calibration_samples` (optional): Number of samples for calibration (default: 100).
- `--num_samples_per_dataset` (optional): Number of samples to use for validation from each dataset (default: 20).

**Output**:
1.  **Final ONNX Model**: A deployable INT8 model named `SceneSegNetwork_QAT_INT8_final.onnx` will be saved in an `onnx_models` sub-directory within the checkpoint folder.
2.  **Validation Scores**: The script will print the final mIoU scores and average inference time to the console, giving you a clear picture of the quantized model's performance.

---

### Final Directory Structure

After completing both steps, your output directory should look something like this:
```
/path/to/save/checkpoints/
├── qat_checkpoint_epoch_4_step_31057.pth
└── onnx_models/
    └── SceneSegNetwork_QAT_INT8_final.onnx
```

