# 🧠 Lite Models – Architectural Overview and Design Rationale

This section provides a comprehensive overview of the three Lite models:

- **EgoLanesLite**
- **SceneSegLite**
- **Scene3DLite**

Their architectural design results from extensive architecture exploration with a clear objective:

---

# 1️⃣ Architectural Foundation

All three Lite models are built on top of the **DeepLabV3+** architecture.

DeepLabV3+ was selected after multiple architecture explorations because it provides:

- Strong multi-scale context modeling  
- Efficient decoder structure  
- High receptive field without excessive depth  
- Clean encoder–decoder separation  

---

## Why DeepLabV3+?

DeepLabV3+ combines:

- A variable encoder
- An ASPP-based decoder
- Dilated convolutions for enlarged receptive field

The key advantage lies in the **ASPP (Atrous Spatial Pyramid Pooling)** module:

- Uses dilated convolutions
- Emulates large receptive fields
- Avoids going excessively deep
- Keeps the number of operations controlled

This is particularly important for lightweight perception models.

---

# 2️⃣ Backbone Choice – EfficientNet Family

Originally, DeepLabV3+ is commonly used with:

- ResNet101  
- ResNet18  

However, following the Autoware perception family direction, the Lite models adopt:

> **EfficientNet backbone family**

- Strong accuracy-to-compute tradeoff  
- Scalable architecture (B0 → B7)  
- Efficient compound scaling  

During experimentation:

- EfficientNetB1 provided the best trade-off  
- B0 (used in Autoware) was slightly underperforming  
- B3 and above significantly increased MACs without proportional gains  

Therefore:

> **EfficientNet-B1 was selected as the default backbone**

---

# 3️⃣ Backbone Sharing & Partial Freezing Strategy

A key research direction explored was **partial backbone sharing**.

### Motivation

In Autoware perception models:

- The encoder is fully shared across tasks
- However, due to large model size, the encoder accounts for ~1% of total compute
- Therefore, sharing provides negligible computational savings

In contrast, Lite models are much smaller.


---

## Explored Strategy

1. Pretrain backbone on segmentation (ImageNet + segmentation task)
2. Freeze encoder partially
3. Share encoder between models (e.g., EgoLanesLite + Scene3DLite)
4. Run encoder once
5. Attach task-specific decoders

This reduces redundant computation when running multiple perception heads.

---

## Findings

- Partial freezing influences performance
- Best absolute performance is obtained when:
  - Each model is trained independently
  - No weight sharing
  - ImageNet-pretrained EfficientNet initialization

However:

- Backbone sharing remains viable for multi-task inference pipelines
- Especially useful in embedded or perception-stack deployment

---

# 4️⃣ Decoder Design and Channel Reduction

DeepLabV3+ decoder default channel count: **256**

For **EgoLanesLite**, due to task simplicity:

- Decoder channels reduced to **64** (¼ of default)
- Performance remained nearly unchanged

This reduces:

- Parameter count
- MACs
- Memory usage

Without degrading lane detection quality.

---

# 5️⃣ Flexible Input Resolution

Unlike SceneSeg family (fixed at 640×320), Lite models:

- Accept variable input resolutions
- Maintain output stride of 1/4
- Can scale to arbitrary input sizes

For example:

- EgoLanesLite preserves 1/4 output stride
- Supports dynamic image resolutions
- Enables flexible deployment scenarios

---

# 6️⃣ Configurable Architecture

All Lite models are fully configurable via configuration files.

Adjustable parameters include:

- Backbone type (EfficientNet family variants)
- Decoder channel count
- Decoder architecture (e.g., DeepLabV3+ or UNet++)
- Bottleneck layer insertion
- Attention blocks between encoder and decoder
- Output head configuration

### Alternative Decoder Exploration

UNet++ was studied:

- More complex skip connections
- Slightly higher memory footprint
- Lower performance-to-compute ratio in this setting

DeepLabV3+ remained the most effective solution.

---

# 8️⃣ Output Head

Final stage consists of:

- A simple 3×3 convolution
- Applied on decoder output
- Maps features to task-specific outputs

This head is configurable and replaceable.

---

# 9️⃣ Implementation Framework

All models are built on top of:

> **Segmentation Models PyTorch (SMP)**

Installation:

```bash
pip install segmentation-models-pytorch
```

The SMP library was wrapped to:

- Provide extended configurability
- Enable modular backbone/decoder swapping
- Support custom bottlenecks
- Allow dynamic channel scaling

The project extends SMP rather than modifying it directly.

---

# 📊 EgoLanesLite – Validation Results

**Note:** The output class order in EgoLanesLite differs from the original EgoLanes model:

- **EgoLanes:** `otherlanes`, `rightmostlane`, `leftmostlane`
- **EgoLanesLite:** `leftmostlane`, `rightmostlane`, `otherlanes`

Be sure to account for this difference when interpreting outputs or comparing results.

Training and validation were performed exclusively on:

- **CurveLanes**
- **TuSimple**

For both datasets, validation sets were created by randomly extracting **500 images from the respective training sets**, following a consistent internal evaluation protocol.

All results are reported in **FP32 precision**.

---

## Performance Comparison (at 640x320 input resolution) on the validation sets

| Dataset | Metric | Original EgoLanes | EgoLanesLite |
|----------|--------|------------------|---------------|
| CurveLanes | mIoU | 46.9 | 44.63 |
| TuSimple | mIoU | 43.6 | 51.4 |

---

## Computational Cost (FP32)

| Model | GOPs |
|--------|--------|
| Original EgoLanes | 119 |
| EgoLanesLite | 4.85 |

This corresponds to a **~24× reduction in operations**, while maintaining competitive performance on those datasets.


## Inference Speed

The following results report **forward-pass performance only** (no preprocessing).  
The forward pass includes:

- Host → Device transfer (H2D)  
- Network inference  
- Device → Host transfer (D2H)  

All benchmarks were executed on a **Jetson Orin Nano (8GB)**.

---

### Forward Pass Performance (Segmentation, Depth, EgoLanes)

| Model              | Backend        | Forward [ms] | FPS |
|--------------------|---------------|--------------|---------------|
| **SceneSeg**       | FP32 ONNX     | 159.6        | 6.3           |
|                    | FP32 TensorRT | 98.1         | 10.2          |
|                    | INT8 TensorRT | —            | —             |
| **SceneSegLite**   | FP32 ONNX     | 43.5         | 23.0          |
|                    | FP32 TensorRT | 23.9         | 41.8          |
|                    | INT8 TensorRT | 11.4         | 87.6          |
|--------------------|---------------|--------------|---------------|
| **Scene3D**        | FP32 ONNX     | 168.4        | 5.9           |
|                    | FP32 TensorRT | 99.7         | 10.0          |
|                    | INT8 TensorRT | —            | —             |
| **Scene3DLite**    | FP32 ONNX     | 41.0         | 24.4          |
|                    | FP32 TensorRT | 23.4         | 42.7          |
|                    | INT8 TensorRT | 10.9         | 91.4          |
|--------------------|---------------|--------------|---------------|
| **EgoLanes**       | FP32 ONNX     | 94.9         | 10.5          |
|                    | FP32 TensorRT | 48.8         | 20.5          |
|                    | INT8 TensorRT | —            | —             |
| **EgoLanesLite**   | FP32 ONNX     | 38.1         | 26.2          |
|                    | FP32 TensorRT | 21.5         | 46.6          |
|                    | INT8 TensorRT | 10.2         | 98.5          |


---

## Limitations

The reduced model capacity introduces inevitable lower expressive power compared to the original architecture, potentially havingweaker generalization to unseen domains.
