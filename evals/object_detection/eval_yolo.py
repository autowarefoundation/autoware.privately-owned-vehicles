#!/usr/bin/env python3
# evals/object_detection/eval_singleimage_object_detection_trt.py

import argparse
from pathlib import Path

import torch
import numpy as np
import cv2

from ultralytics import YOLO  # solo per compatibilità PyTorch
from torchvision.ops import nms as tv_nms

from evals.utils.helpers import ensure, resolve_checkpoint


import numpy as np
import torch
import tensorrt as trt

from cuda import cudart


class TensorRTWrapper:
    """
    Generic TensorRT inference wrapper (TRT >= 9)
    - 1 input
    - N outputs
    - cuda-python (no pycuda)
    - returns tuple(torch.Tensor)
    """

    def __init__(self, engine_path, device="cuda"):
        self.engine_path = str(engine_path)
        self.device = device

        # --------------------------------------------------
        # Ensure Torch CUDA context is initialized first
        # --------------------------------------------------
        if isinstance(device, torch.device) and device.type == "cuda":
            torch.cuda.set_device(device.index or 0)
        elif device == "cuda":
            torch.cuda.set_device(0)

        # --------------------------------------------------
        # Create CUDA stream
        # --------------------------------------------------
        err, self.stream = cudart.cudaStreamCreate()
        if err != 0:
            raise RuntimeError(f"cudaStreamCreate failed (err={err})")

        # --------------------------------------------------
        # Load TensorRT engine
        # --------------------------------------------------
        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")

        self.context = self.engine.create_execution_context()

        # --------------------------------------------------
        # Discover I/O tensors
        # --------------------------------------------------
        self.input_names = []
        self.output_names = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)

            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_names.append(name)

        if len(self.input_names) != 1:
            raise RuntimeError(
                f"Expected exactly 1 input tensor, got {self.input_names}"
            )

        self.input_name = self.input_names[0]

        # --------------------------------------------------
        # Dtypes
        # --------------------------------------------------
        self.input_dtype = trt.nptype(
            self.engine.get_tensor_dtype(self.input_name)
        )

        self.output_dtypes = {
            name: trt.nptype(self.engine.get_tensor_dtype(name))
            for name in self.output_names
        }

        # --------------------------------------------------
        # Initial shapes (may be dynamic)
        # --------------------------------------------------
        self.input_shape = tuple(
            self.context.get_tensor_shape(self.input_name)
        )

        self.output_shapes = {
            name: tuple(self.context.get_tensor_shape(name))
            for name in self.output_names
        }

        print(f"[INFO] TRT input : {self.input_name}  shape={self.input_shape} dtype={self.input_dtype}")
        for name in self.output_names:
            print(f"[INFO] TRT output: {name} shape={self.output_shapes[name]} dtype={self.output_dtypes[name]}")

        # --------------------------------------------------
        # Allocate buffers
        # --------------------------------------------------
        self._allocate_buffers(self.input_shape)

    # ======================================================
    # Memory management
    # ======================================================

    def _allocate_buffers(self, input_shape):
        """
        Allocate (or re-allocate) device + host buffers
        for input and all outputs.
        """

        self.input_shape = tuple(input_shape)

        # -------------------------
        # Free old buffers if any
        # -------------------------
        if hasattr(self, "d_input"):
            cudart.cudaFree(self.d_input)

        if hasattr(self, "d_outputs"):
            for ptr in self.d_outputs.values():
                cudart.cudaFree(ptr)

        # -------------------------
        # Input buffer
        # -------------------------
        in_bytes = int(
            np.prod(self.input_shape) * np.dtype(self.input_dtype).itemsize
        )

        err, self.d_input = cudart.cudaMalloc(in_bytes)
        if err != 0:
            raise RuntimeError(f"cudaMalloc(input) failed (err={err})")

        self.context.set_tensor_address(
            self.input_name, int(self.d_input)
        )

        # -------------------------
        # Output buffers
        # -------------------------
        self.d_outputs = {}
        self.h_outputs = {}

        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = self.output_dtypes[name]

            out_bytes = int(
                np.prod(shape) * np.dtype(dtype).itemsize
            )

            err, dptr = cudart.cudaMalloc(out_bytes)
            if err != 0:
                raise RuntimeError(f"cudaMalloc({name}) failed (err={err})")

            self.context.set_tensor_address(name, int(dptr))

            self.d_outputs[name] = dptr
            self.h_outputs[name] = np.empty(shape, dtype=dtype)

    # ======================================================
    # Inference
    # ======================================================

    def __call__(self, images: torch.Tensor):
        """
        images: torch.Tensor (CUDA) with shape matching engine input
        returns: tuple(torch.Tensor) in output_names order
        """

        assert images.is_cuda, "Input tensor must be CUDA"

        # Torch → NumPy (host)
        inp = (
            images
            .detach()
            .contiguous()
            .cpu()
            .numpy()
            .astype(self.input_dtype)
        )

        # --------------------------------------------------
        # Handle dynamic shapes
        # --------------------------------------------------
        if tuple(inp.shape) != self.input_shape:
            self.context.set_input_shape(self.input_name, inp.shape)
            self._allocate_buffers(inp.shape)

        # --------------------------------------------------
        # H2D input
        # --------------------------------------------------
        err = cudart.cudaMemcpyAsync(
            self.d_input,
            inp.ctypes.data,
            inp.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            self.stream,
        )[0]
        if err != 0:
            raise RuntimeError(f"cudaMemcpyAsync H2D failed (err={err})")

        # --------------------------------------------------
        # Execute
        # --------------------------------------------------
        ok = self.context.execute_async_v3(
            stream_handle=self.stream
        )
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")

        # --------------------------------------------------
        # D2H outputs
        # --------------------------------------------------
        for name in self.output_names:
            h_out = self.h_outputs[name]
            d_out = self.d_outputs[name]

            err = cudart.cudaMemcpyAsync(
                h_out.ctypes.data,
                d_out,
                h_out.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream,
            )[0]
            if err != 0:
                raise RuntimeError(
                    f"cudaMemcpyAsync D2H failed for {name} (err={err})"
                )

        cudart.cudaStreamSynchronize(self.stream)

        # --------------------------------------------------
        # Return torch tensors
        # --------------------------------------------------
        outputs = []
        for name in self.output_names:
            outputs.append(
                torch.from_numpy(self.h_outputs[name]).to(images.device)
            )

        return tuple(outputs)

    def eval(self):
        return self



# ============================================================
# Paths
# ============================================================
EVAL_ROOT = Path("runs/evals/object_detection")


# ============================================================
# Visualization utils
# ============================================================

def yolo_color(cls_id: int):
    """Deterministic YOLO-like color per class."""
    np.random.seed(cls_id)
    return tuple(int(x) for x in np.random.randint(0, 255, size=3))


def draw_predictions(img_bgr, predictions, class_names=None, draw_scores=True):
    """
    predictions: list of [x1,y1,x2,y2,conf,cls]
    img_bgr: uint8 BGR
    """
    out = img_bgr.copy()

    for x1, y1, x2, y2, conf, cls in predictions:
        cls_i = int(cls)
        color = yolo_color(cls_i)

        x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
        cv2.rectangle(out, (x1i, y1i), (x2i, y2i), color, 2)

        if class_names and 0 <= cls_i < len(class_names):
            label = class_names[cls_i]
        else:
            label = f"cls={cls_i}"

        if draw_scores:
            label = f"{label} {conf:.2f}"

        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        y0 = max(0, y1i - th - baseline - 4)
        cv2.rectangle(
            out,
            (x1i, y0),
            (x1i + tw + 6, y0 + th + baseline + 4),
            color,
            -1,
        )
        cv2.putText(
            out,
            label,
            (x1i + 3, y0 + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return out


# ============================================================
# YOLO post-processing (TensorRT, 2 outputs)
# ============================================================

def xywh2xyxy(x):
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def post_process_yolo_trt(
    boxes,      # [1, N, 4] (cx,cy,w,h)
    scores,     # [1, N, C]
    conf_thres=0.25,
    iou_thres=0.45,
):
    """
    Returns:
        Tensor [K,6] -> [x1,y1,x2,y2,conf,cls]
    """

    boxes = boxes.squeeze(0)
    scores = scores.squeeze(0)

    conf, cls = scores.max(dim=1)
    keep = conf > conf_thres

    if keep.sum() == 0:
        return torch.empty(0, 6, device=boxes.device)

    boxes = boxes[keep]
    conf  = conf[keep]
    cls   = cls[keep].float()

    boxes_xyxy = xywh2xyxy(boxes)

    dets = torch.cat(
        [
            boxes_xyxy,
            conf.unsqueeze(1),
            cls.unsqueeze(1),
        ],
        dim=1,
    )

    keep_idx = tv_nms(dets[:, :4], dets[:, 4], iou_thres)
    return dets[keep_idx]


# ============================================================
# Pre / Post helpers
# ============================================================

def normalize_imagenet(img_rgb):
    img = img_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    return img.transpose(2, 0, 1)  # CHW


def scale_boxes_xyxy(boxes, from_hw, to_hw):
    Hf, Wf = from_hw
    Ht, Wt = to_hw
    sx = Wt / max(Wf, 1)
    sy = Ht / max(Hf, 1)

    out = boxes.copy()
    out[:, 0] *= sx
    out[:, 2] *= sx
    out[:, 1] *= sy
    out[:, 3] *= sy
    return out


# ============================================================
# Single image inference
# ============================================================

@torch.no_grad()
def run_single_image_inference(
    model,
    image_path,
    device,
    out_dir,
    conf_thres,
    iou_thres,
    class_names=None,
    width=640,
    height=640,
):
    image_path = Path(image_path)
    assert image_path.exists()

    out_dir.mkdir(parents=True, exist_ok=True)

    img_bgr0 = cv2.imread(str(image_path))
    assert img_bgr0 is not None

    #resize if needed
    if (img_bgr0.shape[1], img_bgr0.shape[0]) != (width, height):
        img_bgr0 = cv2.resize(img_bgr0, (width, height))

    img_rgb0 = cv2.cvtColor(img_bgr0, cv2.COLOR_BGR2RGB)
    H0, W0 = img_rgb0.shape[:2]

    # No letterbox: assume engine was built for fixed input
    img_rgb = img_rgb0.copy()
    H1, W1 = img_rgb.shape[:2]

    chw = normalize_imagenet(img_rgb)
    img_t = torch.from_numpy(chw).unsqueeze(0).to(device)

    model.eval()
    outputs = model(img_t)

    assert isinstance(outputs, (list, tuple)) and len(outputs) == 2, \
        f"Expected 2 outputs, got {type(outputs)}"

    boxes, scores = outputs

    preds = post_process_yolo_trt(
        boxes=boxes,
        scores=scores,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
    )

    preds_cpu = preds.cpu().numpy()

    if preds_cpu.shape[0] > 0 and (H1 != H0 or W1 != W0):
        preds_cpu[:, :4] = scale_boxes_xyxy(
            preds_cpu[:, :4],
            from_hw=(H1, W1),
            to_hw=(H0, W0),
        )

    preds_list = preds_cpu.tolist()
    vis = draw_predictions(img_bgr0, preds_list, class_names)

    stem = image_path.stem
    cv2.imwrite(str(out_dir / f"{stem}_input.png"), img_bgr0)
    cv2.imwrite(str(out_dir / f"{stem}_pred.png"), vis)

    print(f"[OK] Saved results to {out_dir}")
    print(f"[INFO] Detections: {len(preds_list)}")
    for i, (x1,y1,x2,y2,conf,cls) in enumerate(preds_list[:20]):
        print(f"[{i:02d}] cls={int(cls)} conf={conf:.3f} box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser("YOLO TensorRT single image inference")

    ap.add_argument("--run", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=640)

    args = ap.parse_args()
    device = torch.device(args.device)

    run_dir = Path(args.run)

    suffix = "trt"
    out_dir = Path(args.out_dir) if args.out_dir else EVAL_ROOT / f"{run_dir.name}_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Output dir: {out_dir}")
    print(f"[INFO] Using TensorRT engine: {args.engine}")

    model = TensorRTWrapper(args.engine, device=device)

    run_single_image_inference(
        model=model,
        image_path=args.image,
        device=device,
        out_dir=out_dir,
        conf_thres=args.conf,
        iou_thres=args.iou,
        class_names=None,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
