#!/usr/bin/env python3
# evals/segmentation/eval_generic_trt.py

import argparse
from pathlib import Path

import numpy as np
import cv2
import torch

import tensorrt as trt
import pycuda.driver as cuda

from training.segmentation.deeplabv3plus_trainer import DeepLabV3PlusTrainer

from cuda import cudart


# ============================================================
# Paths
# ============================================================
TRAIN_ROOT   = Path("runs/training/segmentation")
EVAL_ROOT    = Path("runs/evals/segmentation")
CONFIG_ROOT  = Path("logs/wandb/latest-run/files/config.yaml")
CHECKPOINT_ROOT = Path("checkpoints")




class TensorRTWrapper:
    """
    TensorRT inference wrapper (TRT >= 9) using cuda-python (no pycuda).
    Safe with PyTorch CUDA context.
    """

    def __init__(self, engine_path, device="cuda"):
        self.engine_path = str(engine_path)
        self.device = device

        # Ensure torch initializes CUDA first
        if isinstance(device, torch.device) and device.type == "cuda":
            torch.cuda.set_device(device.index or 0)
        elif device == "cuda":
            torch.cuda.set_device(0)

        # Create CUDA stream
        err, self.stream = cudart.cudaStreamCreate()
        if err != 0:
            raise RuntimeError(f"cudaStreamCreate failed with error code {err}")

        # Load engine
        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")

        self.context = self.engine.create_execution_context()

        # Discover IO tensors (TRT >= 9 API)
        inputs, outputs = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                inputs.append(name)
            elif mode == trt.TensorIOMode.OUTPUT:
                outputs.append(name)

        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(
                f"Expected 1 input and 1 output tensor, got inputs={inputs}, outputs={outputs}"
            )

        self.input_name  = inputs[0]
        self.output_name = outputs[0]

        self.input_dtype  = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        self.output_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))

        # Initial shapes (can be dynamic)
        self.input_shape  = tuple(self.context.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self.context.get_tensor_shape(self.output_name))

        print(f"[INFO] TRT input : {self.input_name}  shape={self.input_shape}  dtype={self.input_dtype}")
        print(f"[INFO] TRT output: {self.output_name} shape={self.output_shape} dtype={self.output_dtype}")

        # Allocate buffers
        self._alloc_buffers(self.input_shape, self.output_shape)

    # ---------------------------------------------------------
    # Memory management
    # ---------------------------------------------------------

    def _alloc_buffers(self, input_shape, output_shape):
        self.input_shape  = tuple(input_shape)
        self.output_shape = tuple(output_shape)

        in_bytes  = int(np.prod(self.input_shape)  * np.dtype(self.input_dtype).itemsize)
        out_bytes = int(np.prod(self.output_shape) * np.dtype(self.output_dtype).itemsize)

        err, self.d_input = cudart.cudaMalloc(in_bytes)
        if err != 0:
            raise RuntimeError(f"cudaMalloc(input) failed: {err}")

        err, self.d_output = cudart.cudaMalloc(out_bytes)
        if err != 0:
            raise RuntimeError(f"cudaMalloc(output) failed: {err}")

        # Bind addresses
        self.context.set_tensor_address(self.input_name,  int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))

        self.h_output = np.empty(self.output_shape, dtype=self.output_dtype)

    # ---------------------------------------------------------
    # Inference
    # ---------------------------------------------------------

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        assert images.is_cuda, "Input must be CUDA tensor"

        # Torch → numpy (host)
        inp = images.detach().contiguous().cpu().numpy().astype(self.input_dtype)

        # Handle dynamic shapes
        if tuple(inp.shape) != tuple(self.input_shape):
            self.context.set_input_shape(self.input_name, inp.shape)
            new_out_shape = tuple(self.context.get_tensor_shape(self.output_name))

            if new_out_shape != self.output_shape:
                # Free old buffers
                cudart.cudaFree(self.d_input)
                cudart.cudaFree(self.d_output)
                self._alloc_buffers(inp.shape, new_out_shape)

        # H2D
        err = cudart.cudaMemcpyAsync(
            self.d_input,
            inp.ctypes.data,
            inp.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            self.stream,
        )[0]
        if err != 0:
            raise RuntimeError(f"cudaMemcpyAsync H2D failed: {err}")

        # Execute
        ok = self.context.execute_async_v3(stream_handle=self.stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 returned False")

        # D2H
        err = cudart.cudaMemcpyAsync(
            self.h_output.ctypes.data,
            self.d_output,
            self.h_output.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            self.stream,
        )[0]
        if err != 0:
            raise RuntimeError(f"cudaMemcpyAsync D2H failed: {err}")

        cudart.cudaStreamSynchronize(self.stream)

        return torch.from_numpy(self.h_output).to(device=images.device)

    def eval(self):
        return self


