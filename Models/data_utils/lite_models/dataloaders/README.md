# 🚀 Lite Models – Data Parsing

This module contains the complete data parsing and dataset management logic for the lightweight model family:

- **EgoLanesLite**
- **SceneSegLite**
- **Scene3DLite**

The design goal is to provide a **clean, modular, and extensible dataset pipeline** that guarantees:

- Consistent preprocessing across tasks  
- Unified augmentation strategy  
- Fully modular dataset handling  
- Optional pseudo-labeling integration  
- Reproducible training configurations  

---

# 1️⃣ Dataset Abstraction

All datasets inherit from the common base class:

```python
BaseDataset
```

### Core Design Principles

| Feature | Description |
|----------|-------------|
| **Unified Interface** | Every dataset implements the same `__getitem__` contract |
| **Task-Agnostic** | Supports `SEGMENTATION`, `DEPTH`, and `LANE_DETECTION` |
| **Augmentation-Aware** | Augmentations built through a factory pattern |
| **Pseudo-Label Ready** | Supports dynamic pseudo-label generation |
| **Easily Extendable** | New datasets only require `_build_file_list()` implementation |

The abstraction layer ensures that the trainer remains completely independent of dataset-specific logic.

---

# 2️⃣ Dataloader Philosophy

A dedicated dataloader is implemented **per dataset**, including:

- BDD100K  
- Cityscapes  
- CurveLanes  
- TuSimple  
- IDDA  
- ACDC
- MUSES
- MAPILLARY

### Why per-dataset loaders?

This design guarantees:

- No task-specific hacks inside training code  
- Isolation of dataset-specific edge cases  
- Clean split management  
- Full experiment reproducibility  

Each dataset implementation handles:

- File discovery  
- Train/val/test split logic  
- Ground-truth pairing  
- Blacklist handling (if required)  
- Dataset consistency validation  

The training loop remains generic and unaware of dataset internals.

---

# 3️⃣ CurveLanes Adjustments

The original Autoware preprocessing for CurveLanes has been adapted for the Lite architecture.

The crop width was modified to better align with EfficientNet-based backbones (need input dimensions divisible by 32 or 16, depending on the outpout stride).
The processed images are all cropped and resized to 800x400, which is compatible with the Lite models.

Run the following command to apply the crop width change:

```bash
python Models/data_utils/lite_models/preprocess_curvelanes.py --dataset_dir /path/to/curvelanes --output_dir /path/to/output 
```

The general preprocessing structure remains aligned with the original Autoware implementation while being streamlined for Lite models.

---

# 4️⃣ Pseudo-Labeling Support (currently for Depth estimation only)

When `pseudo_labeling=True`, the dataloader:

- Generates placeholder ground-truth tensors  
- Enables on-the-fly inference using large teacher models  
- Supports depth pseudo-labeling (e.g., Depth Anything v2)  

This enables:

- Semi-supervised learning  
- Cross-task supervision  
- Efficient teacher-student training setups  

The mechanism is fully integrated without modifying the training pipeline.

---

# 5️⃣ Augmentation Factory

Augmentations are constructed via:

```python
build_aug(data_type, cfg, mode, pseudo_labeling)
```

### Design Characteristics

- Centralized augmentation definition  
- Mode-aware (`train`, `val`, `test`)  
- Pseudo-label compatible  
- Spatial consistency across tasks  

Lane detection internally reuses segmentation augmentations to maintain geometric alignment.

---

# 6️⃣ Dataset Extensibility

Adding a new dataset requires:

1. Inheriting from `BaseDataset`  
2. Implementing `_build_file_list()`  
3. Defining `self.dataset_name`  
4. Registering augmentation mapping (if required)  

No changes are needed in the trainer or core pipeline.
