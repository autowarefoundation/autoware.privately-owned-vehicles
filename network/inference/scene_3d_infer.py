import torch
from torchvision import transforms
import sys
from network.awf.scene_seg_network import SceneSegNetwork
from network.awf.scene_3d_network import Scene3DNetwork

import onnxruntime as ort
import numpy as np

# TensorRT + PyCUDA
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # initializes CUDA context

from PIL import Image

class Scene3DNetworkInfer():
    def __init__(self, checkpoint_path = ''):

        # Image loader
        self.image_loader = transforms.ToTensor()

        # Checking devices (GPU vs CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using {self.device} for inference')
            
        # Instantiate model, load to device and set to evaluation mode
        sceneSegNetwork = SceneSegNetwork()
        self.model = Scene3DNetwork(sceneSegNetwork)

        if len(checkpoint_path) > 0:
            state = torch.load(checkpoint_path, map_location=self.device)
            # strict=False per ignorare i buffer mancanti
            self.model.load_state_dict(state, strict=False) #no contol on missing keys, since mean and std were added later
        else:
            raise ValueError("No path to checkpoint file provided in class initialization")
        
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

    def inference(self, image):

        width, height = image.size
        if(width != 640 or height != 320):
            raise ValueError('Incorrect input size - input image must have height of 320px and width of 640px')

        image_tensor = self.image_loader(image).unsqueeze(0).to(self.device)  # [1,3,320,640]

        # Run model
        prediction = self.model(image_tensor)

        # Get output, find max class probability and convert to numpy array
        prediction = prediction.squeeze(0).cpu().detach()
        prediction = prediction.permute(1, 2, 0)
        output = prediction.numpy()

        return output
        


# class Scene3DOnnxInfer:
#     def __init__(self, model_path):
#         # Providers: try GPU (CUDA), then fallback to CPU
#         providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
#         self.session = ort.InferenceSession(model_path, providers=providers)

#         # Assume single input/output
#         self.input_name = self.session.get_inputs()[0].name
#         self.output_name = self.session.get_outputs()[0].name
    
#     def normalize_image(self, img):
#         #normalize image with (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) (image - mean) / std
#         mean = np.array([0.485, 0.456, 0.406])
#         std = np.array([0.229, 0.224, 0.225])
#         img = (img - mean) / std
#         return img


#     def preprocess_image(self, pil_image):
#         # Convert PIL → numpy → CHW float32/uint8 depending on model
#         img = np.array(pil_image).astype(np.float32) / 255.0

#         #normalize image with (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) (image - mean) / std
#         img = self.normalize_image(img)


#         img = np.transpose(img, (2, 0, 1))  # HWC → CHW
#         img = np.expand_dims(img, axis=0)   # NCHW
#         return img
    

#     def inference(self, pil_image):
#         # Convert PIL → numpy → CHW float32/uint8 depending on model. input is of shape N C H W

#         img = self.preprocess_image(pil_image).astype(np.float32)

#         # Run inference
#         outputs = self.session.run([self.output_name], {self.input_name: img})
#         #outpout shape is N C H W, with C = num_classes, here 3

#         #each pixel is now a vector of class probabilities
#         # [prob_class0, prob_class1, prob_class2]

#         # B C H W → C H W (C = num_classes)
#         prediction = outputs[0]              # numpy array, shape (1, C, H, W)
#         prediction = np.squeeze(prediction, 0)   # (C, H, W)

#         # Argmax over channels → segmentation mask (H, W), take the highest prob class for each pixel
#         output = np.argmax(prediction, axis=0).astype(np.uint8)

#         return output



class Scene3DOnnxInfer:
    def __init__(self, model_path):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Debug shapes/dtypes once
        print("[ONNX] input:",  self.session.get_inputs()[0].shape,  self.session.get_inputs()[0].type)
        print("[ONNX] output:", self.session.get_outputs()[0].shape, self.session.get_outputs()[0].type)

    def inference(self, pil_image: Image.Image):
        # PIL (RGB) -> float32 [0,1] HWC -> NCHW
        img = np.array(pil_image).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))         # HWC -> CHW
        img = np.expand_dims(img, axis=0)          # -> NCHW

        outputs = self.session.run([self.output_name], {self.input_name: img})
        out = outputs[0]

        # Squeeze all singleton dims
        out = np.squeeze(out)  # could become (H,W) or (C,H,W)
        # If channel-first, convert to (H,W,C)
        if out.ndim == 3:
            # assume C,H,W -> H,W,C
            out = np.transpose(out, (1, 2, 0))
        elif out.ndim == 2:
            # make it (H,W,1) to be consistent with PyTorch branch
            out = out[..., None]

        # keep float32; visualization will scale
        return out.astype(np.float32)

class Scene3DTrtInfer:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Names
        assert self.engine.num_io_tensors == 2, "Expect exactly 1 input and 1 output"
        self.input_name  = [n for n in self.engine if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT][0]
        self.output_name = [n for n in self.engine if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT][0]

        self.stream = cuda.Stream()
        self.input_dptr  = None
        self.output_dptr = None
        self.output_host = None

    def _alloc_if_needed(self, inp_shape):
        # Set runtime input shape (NCHW)
        self.context.set_input_shape(self.input_name, inp_shape)

        # Query runtime shapes
        out_shape = tuple(self.context.get_tensor_shape(self.output_name))

        # Allocate (once) with correct sizes/dtypes
        in_dtype  = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        out_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))

        in_size  = int(np.prod(inp_shape))
        out_size = int(np.prod(out_shape))

        if (self.input_dptr is None) or (self.output_dptr is None):
            self.input_dptr  = cuda.mem_alloc(in_size  * np.dtype(in_dtype).itemsize)
            self.output_dptr = cuda.mem_alloc(out_size * np.dtype(out_dtype).itemsize)
            self.output_host = cuda.pagelocked_empty(out_size, dtype=out_dtype)

        # Bind addresses
        self.context.set_tensor_address(self.input_name,  int(self.input_dptr))
        self.context.set_tensor_address(self.output_name, int(self.output_dptr))

        return out_shape, out_dtype

    def inference(self, pil_image: Image.Image):
        # Input: PIL RGB -> float32 [0,1] -> NCHW
        img = np.array(pil_image).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))            # HWC -> CHW
        img = np.expand_dims(img, axis=0)             # -> NCHW
        img = np.ascontiguousarray(img)               # ensure contiguous

        # Configure context and allocate
        out_shape, out_dtype = self._alloc_if_needed(tuple(img.shape))

        # H2D
        cuda.memcpy_htod_async(self.input_dptr, img, self.stream)

        # Run
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # D2H
        cuda.memcpy_dtoh_async(self.output_host, self.output_dptr, self.stream)
        self.stream.synchronize()

        # Reshape to runtime output shape
        out = np.asarray(self.output_host, dtype=out_dtype).reshape(out_shape)

        # Squeeze batch/channel if needed and return float32 for proper visualization
        out = np.squeeze(out)  # (H,W) or (C,H,W)
        if out.ndim == 3:      # C,H,W -> H,W,C
            out = np.transpose(out, (1, 2, 0))
        elif out.ndim == 2:    # H,W -> H,W,1
            out = out[..., None]

        return out.astype(np.float32)