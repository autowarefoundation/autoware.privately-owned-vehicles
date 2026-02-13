# 🚀 Lite Models – Data Utilities

This module contains utility components specific to the **Lite framework**, including depth pseudo-labeling integration and supporting infrastructure.

---

# 1️⃣ Depth Anything v2 Integration

Lite models optionally support pseudo-labeling via **Depth Anything v2**.

Depth Anything v2 is **already included inside this repository**.  
It was manually integrated from:

https://github.com/DepthAnything/Depth-Anything-V2.git

It is **not included as a submodule** — the source code has been directly added to ensure:

- Full reproducibility  
- No external dependency on nested repositories  
- Stable version control within this project  

---

## Installation Notes

If additional dependencies are required:

```bash
pip install timm
pip install einops
```

No external cloning is necessary.

---

# 2️⃣ On-the-Fly Pseudo Labeling

Pseudo-labeling can be enabled via dataset configuration:

```yaml
pseudo_labeling: true
```

When enabled:

- Ground-truth depth maps are ignored  
- Depth Anything v2 generates depth predictions dynamically  
- Generated depth maps are used as supervision  

This enables:

- Semi-supervised training  
- Cross-dataset knowledge transfer  
- Teacher–student training setups  

The mechanism is fully integrated into the dataset pipeline and does not require trainer modifications.

---

# 3️⃣ Design Rationale

| Feature | Benefit |
|----------|----------|
| On-the-fly generation | No need to store large precomputed depth maps |
| Modular integration | Can be disabled without changing training logic |
| Teacher-agnostic design | Can be replaced with alternative depth models |
| Repository-integrated | No dependency on external submodules |

---

# 4️⃣ Limitations

- Adds computational overhead during training  
- Requires GPU memory for the teacher model  
- Not recommended for strict real-time training pipelines  

---