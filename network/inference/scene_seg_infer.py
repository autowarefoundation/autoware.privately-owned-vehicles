import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from network.awf.scene_seg_network import SceneSegNetwork


# ============================================
#             PyTorch Inference
# ============================================
class SceneSegNetworkInfer:
    def __init__(self, checkpoint_path="", scale_norm=True, stat_norm=True):
        self.scale_norm = scale_norm
        self.stat_norm = stat_norm
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SceneSegNetworkInfer] Using {self.device} for inference")

        self.model = SceneSegNetwork()
        if len(checkpoint_path) > 0:
            state = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state, strict=False)
        else:
            raise ValueError("No path to checkpoint file provided.")

        self.model = self.model.to(self.device).eval()

    def inference(self, pil_image: Image.Image):
        # Convert PIL → float32 tensor [0..1], NCHW
        img = np.array(pil_image).astype(np.float32)
        if self.scale_norm:
            img = img / 255.0  # scale normalization

        if self.stat_norm:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std  # statistical normalization

        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        img = np.expand_dims(img, axis=0)   # NCHW
        img_tensor = torch.from_numpy(img).to(self.device)

        output = self.model(img_tensor)

        output = output.squeeze(0).cpu().numpy()
        return output


# ============================================
#             ONNX Inference
# ============================================
class SceneSegOnnxInfer:
    def __init__(self, model_path, scale_norm=True, stat_norm=True):
        self.scale_norm = scale_norm
        self.stat_norm = stat_norm
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def inference(self, pil_image: Image.Image):
        # Convert PIL → float32 tensor [0..1], NCHW
        img = np.array(pil_image).astype(np.float32)
        if self.scale_norm:
            img = img / 255.0  # scale normalization

        if self.stat_norm:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std  # statistical normalization

        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        img = np.expand_dims(img, axis=0)   # NCHW

        outputs = self.session.run([self.output_name], {self.input_name: img})
        mask = np.squeeze(outputs[0], 0).astype(np.uint8)
        return mask


# ============================================
#             TensorRT Inference
# ============================================
class SceneSegTrtInfer:
    def __init__(self, engine_path, scale_norm=True, stat_norm=True):
        self.scale_norm = scale_norm
        self.stat_norm = stat_norm

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)

        self.bindings = []
        self.host_inputs, self.device_inputs = [], []
        self.host_outputs, self.device_outputs = [], []

        for binding in self.engine:
            shape = self.engine.get_tensor_shape(binding)
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
            size = int(np.prod(shape))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                self.host_inputs.append(host_mem)
                self.device_inputs.append(device_mem)
            else:
                self.host_outputs.append(host_mem)
                self.device_outputs.append(device_mem)

    def inference(self, pil_image: Image.Image):
        # Convert PIL → float32 tensor [0..1], NCHW
        img = np.array(pil_image).astype(np.float32)
        if self.scale_norm:
            img = img / 255.0  # scale normalization

        if self.stat_norm:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std  # statistical normalization

        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        img = np.expand_dims(img, axis=0)   # NCHW

        # Copy to GPU
        cuda.memcpy_htod_async(self.device_inputs[0], img.ravel(), self.stream)

        # Bind and execute
        self.context.set_tensor_address(self.input_name, int(self.device_inputs[0]))
        self.context.set_tensor_address(self.output_name, int(self.device_outputs[0]))
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Retrieve results
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.device_outputs[0], self.stream)
        self.stream.synchronize()

        output = np.reshape(self.host_outputs[0], self.engine.get_tensor_shape(self.output_name))
        mask = np.squeeze(output, 0)
        return mask
