#!/usr/bin/env python3
"""
ONNX and PyTorch Model Benchmarking Utility with mIoU Calculation

This utility provides a framework for comparing the performance and accuracy of
PyTorch FP32, ONNX FP32, and ONNX INT8 segmentation models.

Key Features:
- Benchmarks inference speed for different model types.
- Calculates model size (for ONNX models).
- Evaluates output accuracy using mean Intersection over Union (mIoU), ensuring
  consistency with the SceneSegTrainer's validation methodology.
  Both overall mIoU and per-class IoU scores are reported.
- Supports multiple datasets (ACDC, IDDAW, MUSES, BDD100K, etc.).
- Model-centric evaluation: each model processes all selected samples before moving
  to the next model, optimizing for model warmup and consistent performance measurement.
- Leverages components from the SceneSeg project (SceneSegTrainer, LoadDataSceneSeg)
  for PyTorch model handling and data loading, promoting code reuse and consistency.
- Saves combined visualizations (Original Image | Ground Truth | Prediction) for a
  configurable number of samples per dataset, aiding in qualitative assessment.

Workflow:
1. Parses command-line arguments for model paths, dataset root, and other parameters.
2. Initializes BenchmarkConfig with these settings.
3. Loads dataset samples using BenchmarkDatasetLoader, which internally uses LoadDataSceneSeg.
4. Initializes model handlers (PyTorchModel, ONNXModel) for each specified model.
   - PyTorchModel uses SceneSegTrainer for loading and its validate() method for IoU.
   - ONNXModel uses onnxruntime and mirrors SceneSegTrainer's IoU calculation logic.
5. The Benchmarker orchestrates the evaluation:
   - Performs warmup runs for each model.
   - For each model, processes all benchmark samples, collecting inference times and IoU scores.
   - If enabled, combined visualizations are saved for a few samples per dataset.
6. Aggregates results (mIoU, per-class IoUs, average inference times) per dataset and overall.
7. Generates and prints a comprehensive report comparing all benchmarked models.

Usage Example:
    python benchmark_onnx_models.py \
        --pytorch_model_path /path/to/your_pytorch_model.pth \
        --fp32_model_path /path/to/your_fp32_model.onnx \
        --int8_model_path /path/to/your_int8_model.onnx \
        --dataset_root /path/to/your/SceneSeg_datasets_root/ \
        --num_samples_per_dataset 10 \
        --visualizations_per_dataset 2 \
        --save_outputs
"""

import os
import time
import logging
import argparse
import numpy as np
import onnxruntime
import platform
import cv2
import random
from datetime import datetime   
import sys
from PIL import Image
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

################################################################################
# Import Progress Bar Utility
################################################################################
try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found. Please install it for progress bars: pip install tqdm", file=sys.stderr)
    def tqdm(iterable, *args, **kwargs): return iterable

################################################################################
# PyTorch and Project Specific Imports
################################################################################
try:
    import torch
    from torchvision import transforms as T # Alias to avoid conflict if any
    # Adjust path to import SceneSegNetwork and LoadDataSceneSeg
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_models = os.path.abspath(os.path.join(script_dir, '..', '..'))
    if project_root_models not in sys.path:
        sys.path.append(project_root_models)
    
    # Ensure project root is in path for sibling directory imports
    from data_utils.load_data_scene_seg import LoadDataSceneSeg 
    from training.scene_seg_trainer import SceneSegTrainer  # For PyTorch model processing
   
    PYTORCH_AVAILABLE = True
except ImportError as e:
    print(f"CRITICAL: PyTorch or project modules (SceneSegNetwork, LoadDataSceneSeg, SceneSegTrainer) not found. Error: {e}. PyTorch benchmarking will be unavailable.", file=sys.stderr)
    PYTORCH_AVAILABLE = False
    SceneSegNetwork = None 
    LoadDataSceneSeg = None
    SceneSegTrainer = None
    torch = None
    T = None

################################################################################
# Logging Configuration
################################################################################
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

################################################################################
# Common Definitions
################################################################################
CLASS_NAMES: List[str] = ["Background", "Foreground", "Road"] # Ensure this order matches model outputs and GT

################################################################################
# Configuration Class
################################################################################
@dataclass
class BenchmarkConfig:
    """Configuration for the benchmarking process.

    Attributes:
        pytorch_model_path: Path to the PyTorch (.pth) model checkpoint.
        fp32_model_path: Path to the FP32 ONNX model.
        int8_model_path: Path to the INT8 ONNX model.
        dataset_root: Root directory of the segmentation datasets.
        num_samples_per_dataset: Maximum number of samples to process from each dataset.
        warmup_runs: Number of warmup inference runs before actual benchmarking.
        save_outputs: Whether to save output visualizations.
        output_dir_base: Base directory for saving benchmark results and visualizations.
        pytorch_target_hw: Target (Height, Width) for PyTorch model preprocessing and GT resizing.
                           If None, SceneSegTrainer's default might be used or a script default.
        onnx_target_hw: Target (Height, Width) for ONNX model preprocessing and GT resizing.
                        If None, it's derived from the FP32 ONNX model's input metadata.
        visualizations_per_dataset: Number of combined visualizations (Image|GT|Prediction)
                                    to save per dataset and per model.
    """
    pytorch_model_path: Optional[str] = None
    fp32_model_path: Optional[str] = None
    int8_model_path: Optional[str] = None
    dataset_root: str = ""
    num_samples_per_dataset: int = 20
    warmup_runs: int = 3
    save_outputs: bool = False
    output_dir_base: str = "./benchmark_results_multimodel"
    pytorch_target_hw: Optional[Tuple[int, int]] = None 
    onnx_target_hw: Optional[Tuple[int, int]] = None
    visualizations_per_dataset: int = 2


################################################################################
# Dataset Loading 
################################################################################
class BenchmarkDatasetLoader:
    """Loads and prepares dataset samples for benchmarking.

    This class scans specified dataset directories, loads image and ground truth data
    using LoadDataSceneSeg, and makes a specified number of samples available for
    the benchmarking process. It stores samples as dictionaries containing paths,
    image data (NumPy array), ground truth lists (from LoadDataSceneSeg.getItemVal),
    the dataset key, and the original sample index.
    """
    def __init__(self, dataset_root_dir: str, num_samples_per_dataset: int):
        """Initializes the BenchmarkDatasetLoader.

        Args:
            dataset_root_dir: The root directory where datasets (ACDC, IDDAW, etc.) are stored.
            num_samples_per_dataset: The maximum number of validation samples to load from each dataset.
        
        Raises:
            RuntimeError: If LoadDataSceneSeg is not available (e.g., due to import issues).
        """
        self.dataset_root_dir = dataset_root_dir
        if not self.dataset_root_dir.endswith('/'):
            self.dataset_root_dir += '/'
        self.num_samples_per_dataset = num_samples_per_dataset
        self.samples = []
        self.dataset_loaders = {}  # Store dataset loaders for later use
        if LoadDataSceneSeg is None:
            logger.error("LoadDataSceneSeg unavailable, cannot initialize BenchmarkDatasetLoader.")
            raise RuntimeError("LoadDataSceneSeg is not available.")
        self._load_all_samples()

    def _load_all_samples(self) -> None:
        """Scans dataset directories, loads samples using LoadDataSceneSeg.
        
        Constructs paths to image and label directories for each configured dataset.
        If a dataset is found, it initializes LoadDataSceneSeg and retrieves a specified
        number of validation samples. These samples (image NumPy array, ground truth list from
        getItemVal, image path, dataset key, sample index) are collected, shuffled, and stored.
        """
        all_raw_samples = []
        
        # Direct path construction like in training script
        datasets_to_load = {
            'ACDC': {
                'images': self.dataset_root_dir + 'ACDC/ACDC/images/',
                'labels': self.dataset_root_dir + 'ACDC/ACDC/gt_masks/'
            },
            'IDDAW': {
                'images': self.dataset_root_dir + 'IDDAW/IDDAW/images/',
                'labels': self.dataset_root_dir + 'IDDAW/IDDAW/gt_masks/'
            },
            'MUSES': {
                'images': self.dataset_root_dir + 'MUSES/MUSES/images/',
                'labels': self.dataset_root_dir + 'MUSES/MUSES/gt_masks/'
            },
            'MAPILLARY': {
                'images': self.dataset_root_dir + 'Mapillary_Vistas/Mapillary_Vistas/images/',
                'labels': self.dataset_root_dir + 'Mapillary_Vistas/Mapillary_Vistas/gt_masks/'
            },
            'COMMA10K': {
                'images': self.dataset_root_dir + 'comma10k/comma10k/images/',
                'labels': self.dataset_root_dir + 'comma10k/comma10k/gt_masks/'
            },
            'BDD100K': {
                'images': self.dataset_root_dir + 'BDD100K/BDD100K/images/',
                'labels': self.dataset_root_dir + 'BDD100K/BDD100K/gt_masks/'
            }
        }
        
        for dataset_key, paths in tqdm(datasets_to_load.items(), desc="Scanning datasets"):
            try:
                images_fp = paths['images']
                labels_fp = paths['labels']
                
                # Check if paths exist
                if not os.path.exists(images_fp) or not os.path.isdir(images_fp) or \
                   not os.path.exists(labels_fp) or not os.path.isdir(labels_fp):
                    continue
                
                # Load dataset using LoadDataSceneSeg
                dataset_loader = LoadDataSceneSeg(labels_filepath=labels_fp, images_filepath=images_fp, dataset=dataset_key)
                self.dataset_loaders[dataset_key] = dataset_loader
                
                _, num_val_available = dataset_loader.getItemCount()
                actual_samples_to_load = min(num_val_available, self.num_samples_per_dataset)
                
                if actual_samples_to_load == 0:
                    continue
                
                logger.info(f"Loading {actual_samples_to_load} samples from {dataset_key}")
                for i in range(actual_samples_to_load):
                    img_np, gt_list, _ = dataset_loader.getItemVal(i)
                    # Store original path, dataset key, and sample index for later reference
                    all_raw_samples.append({
                        'img_path': dataset_loader.val_images[i],
                        'img_np': img_np,
                        'gt_list': gt_list,
                        'dataset_key': dataset_key,
                        'sample_idx': i
                    })
            except Exception as e:
                logger.error(f"Error processing dataset {dataset_key}: {e}", exc_info=True)
        
        if not all_raw_samples:
            logger.error("No samples were loaded from any dataset.")
        else:
            random.shuffle(all_raw_samples)
            self.samples = all_raw_samples
            logger.info(f"Total {len(self.samples)} samples collected for benchmarking.")

    def get_samples(self):
        """Returns the list of loaded and shuffled benchmark samples.

        Each sample is a dictionary with keys: 'img_path', 'img_np', 'gt_list', 
        'dataset_key', 'sample_idx'.
        """
        return self.samples

    def get_dataset_loader(self, dataset_key):
        """Returns the LoadDataSceneSeg instance for a given dataset key.

        Args:
            dataset_key: The key identifying the dataset (e.g., 'ACDC').

        Returns:
            The LoadDataSceneSeg instance for the dataset, or None if not found.
        """
        return self.dataset_loaders.get(dataset_key)

################################################################################
# Model Interface & Implementations
################################################################################
class ModelInterface(ABC):
    """Abstract Base Class for model handlers (ONNX, PyTorch, etc.).

    Defines a common interface for loading models, preprocessing inputs, running inference,
    and obtaining segmentation maps and visualizations. This allows the Benchmarker
    to treat different model types uniformly.
    """
    def __init__(self, model_path: Optional[str], model_name: str, device_str: str = "cpu"):
        """Initializes the ModelInterface.

        Args:
            model_path: Path to the model file.
            model_name: A descriptive name for the model (e.g., "ONNX_FP32", "PyTorch_FP32").
            device_str: The device string for model execution (e.g., "cpu", "cuda"),
                        primarily relevant for PyTorch models.
        """
        self.model_path = model_path
        self.model_name = model_name
        self.device_str = device_str
        self.model: Any = None
        self._target_hw: Optional[Tuple[int, int]] = None # Cache for target_hw
        self._model_info: Dict[str, Any] = {}

    @abstractmethod
    def load_model(self) -> bool:
        pass

    def get_model_info(self) -> Dict[str, Any]:
        if not self._model_info and self.is_loaded(): # Compute if not cached
            self._model_info = self._compute_model_info()
        return self._model_info
    
    @abstractmethod
    def _compute_model_info(self) -> Dict[str, Any]: # Specific to each model type
        pass

    @abstractmethod
    def get_target_hw(self) -> Tuple[int, int]: # H, W for preprocessing and GT resizing
        pass

    @abstractmethod
    def preprocess_input(self, pil_image: Image.Image, target_hw: Tuple[int, int]) -> Any:
        pass # Returns model-ready tensor

    @abstractmethod
    def run_inference(self, processed_input: Any) -> np.ndarray:
        pass # Returns raw output as NCHW float32 numpy array

    @abstractmethod
    def get_segmentation_map(self, raw_output_nchw: np.ndarray) -> np.ndarray:
        pass # Returns HW class index map (uint8)

    def get_visualization_colors(self) -> List[Tuple[int,int,int]]: # BGR colors
        # Use the exact same colors as in LoadDataSceneSeg.createGroundTruth
        # and SceneSegTrainer.make_visualization
        return [(61, 93, 255),  # background_objects_colour
                (255, 28, 145), # foreground_objects_colour
                (0, 255, 220)]  # road_colour

    def get_visualization(self, pred_map_hw: np.ndarray, target_hw_vis: Tuple[int, int]) -> Image.Image:
        colors_bgr = self.get_visualization_colors()
        vis_h, vis_w = target_hw_vis
        
        colored_mask_bgr = np.zeros((vis_h, vis_w, 3), dtype=np.uint8)
        # Resize pred_map_hw if its shape is different from target_hw_vis
        if pred_map_hw.shape != (vis_h, vis_w):
            pred_map_hw_resized = cv2.resize(pred_map_hw.astype(np.uint8), (vis_w, vis_h), interpolation=cv2.INTER_NEAREST)
        else:
            pred_map_hw_resized = pred_map_hw.astype(np.uint8)

        for class_idx, color in enumerate(colors_bgr):
            if class_idx < len(CLASS_NAMES):
                colored_mask_bgr[pred_map_hw_resized == class_idx] = color
        return Image.fromarray(cv2.cvtColor(colored_mask_bgr, cv2.COLOR_BGR2RGB))

    def is_loaded(self) -> bool:
        return self.model is not None

class ONNXModel(ModelInterface):
    """Handles ONNX model loading, inference, and evaluation.

    This class uses onnxruntime to load and run ONNX models. It preprocesses input images,
    runs inference, calculates IoU scores by mirroring SceneSegTrainer's logic, and provides
    segmentation maps for visualization.
    """
    def __init__(self, model_path: str, model_name: str, device_str: str = "cpu", target_hw_override: Optional[Tuple[int,int]] = None):
        """Initializes the ONNXModel handler.

        Args:
            model_path: Path to the .onnx model file.
            model_name: Descriptive name for the model.
            device_str: Device for inference (e.g., "cpu", "cuda"). Currently, onnxruntime
                        providers are hardcoded to ['CUDAExecutionProvider'] but this argument
                        is kept for interface consistency.
            target_hw_override: Optional (Height, Width) to override model's input dimensions.
        """
        super().__init__(model_path, model_name, device_str)
        self.session: Optional[onnxruntime.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.output_names: Optional[List[str]] = None
        self._target_hw_override = target_hw_override
        self.temp_trainer = SceneSegTrainer() if SceneSegTrainer is not None else None

    def load_model(self) -> bool:
        if not self.model_path or not os.path.exists(self.model_path):
            logger.error(f"ONNX model path invalid or not found: {self.model_path} for {self.model_name}")
            return False
        try:
            # TODO: Allow specifying providers (e.g. CUDAExecutionProvider)
            self.session = onnxruntime.InferenceSession(self.model_path, providers=['CUDAExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [out.name for out in self.session.get_outputs()]
            self.model = self.session # For is_loaded()
            logger.info(f"ONNX model {self.model_name} loaded from {self.model_path} with providers {self.session.get_providers()}")
            # Prime target_hw and model_info caches
            self.get_target_hw()
            self.get_model_info()
            return True
        except Exception as e:
            logger.error(f"Failed to load ONNX model {self.model_name} from {self.model_path}: {e}", exc_info=True)
            self.session = None
            self.model = None
            return False

    def _compute_model_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        if not self.model_path or not os.path.exists(self.model_path): return info
        info['size_bytes'] = os.path.getsize(self.model_path)
        info['size_mb'] = info['size_bytes'] / (1024 * 1024)
        info['type'] = 'ONNX'
        info['path'] = self.model_path
        if self.session:
            inputs = self.session.get_inputs()
            info['input_names'] = [inp.name for inp in inputs]
            info['input_shapes'] = [inp.shape for inp in inputs] # List of lists/tuples
            info['output_names'] = [out.name for out in self.session.get_outputs()]
            # H,W from metadata
            h, w = self._get_hw_from_onnx_metadata()
            info['metadata_height'] = h
            info['metadata_width'] = w
        return info
        
    def _get_hw_from_onnx_metadata(self) -> Tuple[Optional[int], Optional[int]]:
        if self.session:
            first_input_shape = self.session.get_inputs()[0].shape
            if len(first_input_shape) == 4 and \
               all(isinstance(dim, int) and dim > 0 for dim in first_input_shape[2:]):
                return first_input_shape[2], first_input_shape[3] # H, W
        return None, None

    def get_target_hw(self) -> Tuple[int, int]:
        if self._target_hw: return self._target_hw
        
        if self._target_hw_override:
            logger.info(f"Using target H,W override for {self.model_name}: {self._target_hw_override}")
            self._target_hw = self._target_hw_override
            return self._target_hw

        h, w = self._get_hw_from_onnx_metadata()
        if h and w:
            self._target_hw = (h,w)
            return self._target_hw
        else:
            logger.error(f"Could not determine target H,W for ONNX model {self.model_name}. Input shape: {self.session.get_inputs()[0].shape if self.session else 'N/A'}. Please provide --onnx_target_hw.")
            raise ValueError(f"Target H,W indeterminable for {self.model_name}")

    def preprocess_input(self, sample_data: Dict[str, Any], target_hw: Tuple[int, int]) -> Dict[str, Any]:
        """Preprocesses the input image from sample_data for ONNX inference.

        Extracts the image NumPy array from sample_data, converts it to a PIL Image,
        resizes it to target_hw, normalizes it, and transposes it to CHW format.
        The processed NumPy array (NCHW, float32) is stored back in the sample_data
        dictionary under the key 'processed_input'.

        Args:
            sample_data: Dictionary containing the raw sample data, including 'img_np'.
            target_hw: Target (Height, Width) for resizing.

        Returns:
            The modified sample_data dictionary with 'processed_input', or None on error.
        """
        try:
            # Process the image for ONNX inference
            img_np = sample_data['img_np']
            pil_image = Image.fromarray(img_np).convert("RGB")
            
            img_resized_pil = pil_image.resize((target_hw[1], target_hw[0]), Image.Resampling.BILINEAR if hasattr(Image, 'Resampling') else Image.BILINEAR)
            img_resized_np = np.array(img_resized_pil)
            norm_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
            norm_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
            img_normalized = (img_resized_np / 255.0 - norm_mean) / norm_std
            img_chw = img_normalized.transpose(2, 0, 1)
            
            # Store processed input in the sample data
            sample_data['processed_input'] = np.expand_dims(img_chw, axis=0).astype(np.float32)
            return sample_data
        except Exception as e:
            logger.error(f"Error preprocessing image for ONNX model {self.model_name}: {e}", exc_info=True)
            return None

    def run_inference(self, sample_data: Dict[str, Any]) -> Optional[List[float]]:
        if not self.session or not self.input_name or not sample_data: 
            return None
            
        try:
            # Get the processed input
            processed_input = sample_data.get('processed_input')
            if processed_input is None:
                logger.error(f"Processed input not found in sample data for {self.model_name}")
                return None
                
            # Run ONNX inference
            outputs = self.session.run(self.output_names, {self.input_name: processed_input})
            raw_output = outputs[0]
            
            # Calculate IoU scores directly
            return self.calc_iou_from_output(raw_output, sample_data)
        except Exception as e:
            logger.error(f"Error during ONNX inference for {self.model_name}: {e}", exc_info=True)
            return None
            
    def calc_iou_from_output(self, raw_output: np.ndarray, sample_data: Dict[str, Any]) -> List[float]:
        """Calculates IoU scores using same approach as SceneSegTrainer.calc_IoU_val"""
        if raw_output is None or sample_data is None or self.temp_trainer is None:
            logger.error("Cannot calculate IoU: missing raw output, sample data, or trainer")
            return [0.0, 0.0, 0.0, 0.0]  # Full, BG, FG, RD
            
        try:
            # Get ground truth data from sample
            gt_list = sample_data.get('gt_list')
            if not gt_list or len(gt_list) < 4:
                logger.error("Invalid ground truth data in sample")
                return [0.0, 0.0, 0.0, 0.0]  # Full, BG, FG, RD
                
            # Process exactly like in calc_IoU_val
            if raw_output.ndim == 4 and raw_output.shape[0] == 1:
                output_val = raw_output.squeeze(0) # Remove batch dim
            else:
                output_val = raw_output
                
            # Convert from CHW to HWC (if needed)
            if output_val.ndim == 3 and output_val.shape[0] <= 3:  # Likely CHW format
                output_val = np.transpose(output_val, (1, 2, 0))
                
            # Now we have HWC format, similar to SceneSegTrainer.calc_IoU_val output
            output_processed = np.zeros_like(output_val)
            
            # Convert to one-hot format exactly like in calc_IoU_val
            for x in range(0, output_val.shape[0]):
                for y in range(0, output_val.shape[1]):
                    bg_prob = output_val[x,y,0]
                    fg_prob = output_val[x,y,1]
                    rd_prob = output_val[x,y,2]

                    if(bg_prob >= fg_prob and bg_prob >= rd_prob):
                        output_processed[x,y,0] = 1
                        output_processed[x,y,1] = 0
                        output_processed[x,y,2] = 0
                    elif(fg_prob >= bg_prob and fg_prob >= rd_prob):
                        output_processed[x,y,0] = 0
                        output_processed[x,y,1] = 1
                        output_processed[x,y,2] = 0
                    elif(rd_prob >= bg_prob and rd_prob >= fg_prob):
                        output_processed[x,y,0] = 0
                        output_processed[x,y,1] = 0
                        output_processed[x,y,2] = 1
            
            # First, prepare GT in the format expected by calc_IoU_val
            # These match the format in SceneSegTrainer.apply_augmentations
            h, w = output_processed.shape[:2]
            bg_mask = gt_list[1].astype(bool)
            fg_mask = gt_list[2].astype(bool)
            rd_mask = gt_list[3].astype(bool)
                
            # Resize masks to match model output
            bg_mask_resized = cv2.resize(bg_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            fg_mask_resized = cv2.resize(fg_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            rd_mask_resized = cv2.resize(rd_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                
            # Stack into HWC
            gt_val_fused = np.stack((bg_mask_resized, fg_mask_resized, rd_mask_resized), axis=2)
            
            # Calculate IoU exactly as in calc_IoU_val
            iou_score_full = self.temp_trainer.IoU(output_processed, gt_val_fused)
            iou_score_bg = self.temp_trainer.IoU(output_processed[:,:,0], gt_val_fused[:,:,0])
            iou_score_fg = self.temp_trainer.IoU(output_processed[:,:,1], gt_val_fused[:,:,1])
            iou_score_rd = self.temp_trainer.IoU(output_processed[:,:,2], gt_val_fused[:,:,2])
            
            # For visualization, save the processed output
            sample_data['output_processed'] = output_processed
            
            # Return all IoU scores with full IoU first, just like in training script
            return [iou_score_full, iou_score_bg, iou_score_fg, iou_score_rd]
        except Exception as e:
            logger.error(f"Error calculating IoU for ONNX model: {e}", exc_info=True)
            return [0.0, 0.0, 0.0, 0.0]  # Full, BG, FG, RD

    def get_segmentation_map(self, sample_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extracts the H,W class index map from the processed output stored in sample_data.

        This map is typically used for generating visualizations. It assumes that
        calc_iou_from_output has already been called and stored the one-hot 'output_processed'
        in the sample_data dictionary.

        Args:
            sample_data: The dictionary containing sample data, including 'output_processed'.

        Returns:
            A NumPy array (H, W) of class indices, or None if an error occurs.
        """
        # This is now mainly for visualization purposes
        if not sample_data or 'output_processed' not in sample_data:
            logger.warning(f"ONNXModel '{self.model_name}': 'output_processed' not found in sample_data for get_segmentation_map. Was calc_iou_from_output run?")
            return None
            
        try:
            # output_processed is in one-hot HWC format, convert to class indices
            output_processed = sample_data['output_processed']
            return np.argmax(output_processed, axis=2).astype(np.uint8)
        except Exception as e:
            logger.error(f"Error in ONNXModel.get_segmentation_map: {e}", exc_info=True)
            return None

class PyTorchModel(ModelInterface):
    """Handles PyTorch model loading, inference, and evaluation using SceneSegTrainer.

    This class leverages the existing SceneSegTrainer for most operations, ensuring consistency
    with the training pipeline. It uses trainer.validate() for IoU calculation and provides
    methods to get segmentation maps for visualization by re-running inference on the
    trainer's internal validation tensor.
    """
    def __init__(self, model_path: str, model_name: str, device_str: str = "cpu", target_hw_override: Optional[Tuple[int, int]] = None):
        """Initializes the PyTorchModel handler.

        Args:
            model_path: Path to the PyTorch (.pth) model checkpoint.
            model_name: Descriptive name for the model.
            device_str: Device to load the model on (e.g., "cpu", "cuda").
            target_hw_override: Optional (Height, Width) to override the default target dimensions.
                                This primarily influences GT resizing if the model's internal
                                target dimensions differ.
        """
        super().__init__(model_path, model_name, device_str)
        if not PYTORCH_AVAILABLE or SceneSegTrainer is None: # Check SceneSegTrainer
            raise RuntimeError("PyTorch or SceneSegTrainer is not available, cannot create PyTorchModel.")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"PyTorchModel '{self.model_name}' will use device: {self.device}")
        
        self._target_hw_override = target_hw_override
        self.trainer_instance: Optional[SceneSegTrainer] = None # Will hold SceneSegTrainer

    def load_model(self) -> bool:
        if not self.model_path or not os.path.exists(self.model_path):
            logger.error(f"PyTorch model path invalid or not found: {self.model_path} for {self.model_name}")
            return False
        try:
            # Instantiate SceneSegTrainer - it loads the model internally if checkpoint_path is given
            self.trainer_instance = SceneSegTrainer(checkpoint_path=self.model_path) # Pass model path to trainer
            self.trainer_instance.model = self.trainer_instance.model.to(self.device) # Ensure model is on correct device
            self.trainer_instance.set_eval_mode() # Calls self.model.eval()
            self.model = self.trainer_instance.model # For ModelInterface.is_loaded()
            
            logger.info(f"PyTorch model '{self.model_name}' (via SceneSegTrainer) loaded from '{self.model_path}' to {self.device}")
            self.get_target_hw()
            self.get_model_info()
            return True
        except Exception as e:
            logger.error(f"Failed to load PyTorch model '{self.model_name}' using SceneSegTrainer: {e}", exc_info=True)
            self.model = None; self.trainer_instance = None
            return False

    def _compute_model_info(self) -> Dict[str, Any]:
        info : Dict[str, Any] = {}
        if self.model_path and os.path.exists(self.model_path):
             info['size_bytes'] = os.path.getsize(self.model_path)
             info['size_mb'] = info['size_bytes'] / (1024 * 1024)
        info['type'] = 'PyTorch'; info['path'] = self.model_path
        if self.model:
            try: info['num_trainable_params'] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            except: pass 
        return info

    def get_target_hw(self) -> Tuple[int, int]:
        if self._target_hw: return self._target_hw
        if self._target_hw_override:
            logger.info(f"Using target H,W override for PyTorch model '{self.model_name}': {self._target_hw_override}")
            self._target_hw = self._target_hw_override
            return self._target_hw
        default_hw = (320, 640) 
        logger.info(f"PyTorch model '{self.model_name}' effective target H,W (from SceneSegTrainer default or override): {default_hw}. This is mainly for GT resizing if trainer doesn't return a map.")
        self._target_hw = default_hw
        return self._target_hw

    def preprocess_input(self, sample_data: Dict[str, Any], target_hw: Tuple[int, int]) -> Dict[str, Any]:
        # No preprocessing needed since we're passing data directly to validate
        return sample_data

    def run_inference(self, sample_data: Dict[str, Any]) -> Optional[List[float]]:
        # For PyTorch, we'll directly use trainer.validate which returns IoU scores
        if not self.trainer_instance or sample_data is None: 
            logger.error(f"Trainer instance or sample_data is None for '{self.model_name}'.")
            return None
            
        try:
            # Directly use validate as shown in the example code
            # validate expects raw numpy image and gt_list returned by getItemVal
            IoU_score_full, IoU_score_bg, IoU_score_fg, IoU_score_rd = self.trainer_instance.validate(
                sample_data['img_np'], sample_data['gt_list']
            )
            
            # Return all IoU scores with full IoU first, just like in training script
            return [IoU_score_full, IoU_score_bg, IoU_score_fg, IoU_score_rd]
        except Exception as e:
            logger.error(f"Error in PyTorchModel.run_inference using validate for '{self.model_name}': {e}", exc_info=True)
            return None
            
    def get_segmentation_map(self) -> Optional[np.ndarray]:
        """Generates a segmentation map (HW class indices) after validate() has been run.
           This is used to get the raw prediction map for visualization.
        """
        if not self.trainer_instance or self.trainer_instance.image_val_tensor is None:
            logger.error("PyTorchModel: Trainer instance or its image_val_tensor is not ready for get_segmentation_map. Ensure run_inference (which calls validate) was called first.")
            return None
        try:
            with torch.no_grad():
                # Ensure model is on the correct device
                self.trainer_instance.model.to(self.device)
                self.trainer_instance.image_val_tensor = self.trainer_instance.image_val_tensor.to(self.device)
                
                raw_output_tensor = self.trainer_instance.model(self.trainer_instance.image_val_tensor)
            
            # Process CHW tensor to HW class index map
            if raw_output_tensor.ndim == 4 and raw_output_tensor.shape[0] == 1:
                output_chw_np = raw_output_tensor.squeeze(0).cpu().numpy()
            elif raw_output_tensor.ndim == 3:
                output_chw_np = raw_output_tensor.cpu().numpy()
            else:
                logger.error(f"PyTorchModel: Unexpected output tensor dim from model: {raw_output_tensor.shape}")
                return None
            
            pred_map_hw = np.argmax(output_chw_np, axis=0).astype(np.uint8)
            return pred_map_hw
        except Exception as e:
            logger.error(f"Error in PyTorchModel.get_segmentation_map: {e}", exc_info=True)
            return None

################################################################################
# Evaluation Utilities
################################################################################
# IoU calculation is done using the same approach as in SceneSegTrainer:
# 1. For PyTorch models: directly use trainer.validate() which returns IoU scores
# 2. For ONNX models: use calc_iou_from_output which mirrors SceneSegTrainer.calc_IoU_val
#    and uses SceneSegTrainer.IoU for the final calculation
# This ensures consistency between benchmarking and training evaluation.

################################################################################
# Benchmarking Orchestration
################################################################################
class Benchmarker:
    """Orchestrates the model benchmarking process.

    This class manages the overall workflow of evaluating multiple models across
    multiple datasets. It handles:
    - Warmup runs for each model.
    - For each model, processes all benchmark samples, collecting inference times and IoU scores.
    - If enabled, combined visualizations are saved for a few samples per dataset.
    - Aggregating results (mIoU, per-class IoUs, average inference times) per dataset and overall.
    """
    def __init__(self, config: BenchmarkConfig, models_to_benchmark: List[ModelInterface], dataset_loader: BenchmarkDatasetLoader):
        """Initializes the Benchmarker.

        Args:
            config: The BenchmarkConfig object with all settings.
            models_to_benchmark: A list of initialized ModelInterface compliant model handlers.
            dataset_loader: An initialized BenchmarkDatasetLoader providing the samples.
        """
        self.config = config
        self.models = [m for m in models_to_benchmark if m.is_loaded()] # Only use successfully loaded models
        self.dataset_loader = dataset_loader
        # Track visualizations per dataset and model
        self.visualizations_count = defaultdict(lambda: defaultdict(int))
        # Create output directory for visualizations
        if config.save_outputs:
            self.visualization_dir = os.path.join(config.output_dir_base, "visualizations")
            os.makedirs(self.visualization_dir, exist_ok=True)

    def _save_pil_combined_visualization(self, sample_data: Dict[str, Any], 
                                      pred_hw_map: np.ndarray, 
                                      model_handler: ModelInterface, 
                                      dataset_name: str, 
                                      target_vis_hw: Tuple[int,int]) -> None:
        """Save a combined PIL image: Original | Ground Truth | Prediction."""
        try:
            if self.visualizations_count[dataset_name][model_handler.model_name] >= self.config.visualizations_per_dataset:
                return

            output_subdir = os.path.join(self.visualization_dir, dataset_name)
            os.makedirs(output_subdir, exist_ok=True)
            img_basename = os.path.splitext(os.path.basename(sample_data['img_path']))[0]

            # Original Image (from sample_data['img_np'])
            original_pil = Image.fromarray(sample_data['img_np']).convert("RGB")
            orig_resized = original_pil.resize((target_vis_hw[1], target_vis_hw[0]), 
                                             Image.Resampling.BILINEAR if hasattr(Image, 'Resampling') else Image.BILINEAR)
            
            # Ground Truth Visualization (from sample_data['gt_list'][0])
            gt_vis_pil = Image.fromarray(sample_data['gt_list'][0]).convert("RGB") # gt_list[0] is the vis from createGroundTruth
            gt_vis_resized = gt_vis_pil.resize((target_vis_hw[1], target_vis_hw[0]), 
                                              Image.Resampling.NEAREST if hasattr(Image, 'Resampling') else Image.NEAREST)

            # Prediction Visualization (using model_handler.get_visualization)
            pred_vis_pil = model_handler.get_visualization(pred_hw_map, target_vis_hw)

            # Combine: Original | GT | Prediction
            total_width = target_vis_hw[1] * 3
            combined_image = Image.new('RGB', (total_width, target_vis_hw[0]))
            combined_image.paste(orig_resized, (0,0))
            combined_image.paste(gt_vis_resized, (target_vis_hw[1], 0))
            combined_image.paste(pred_vis_pil, (target_vis_hw[1]*2, 0))
            
            save_path = os.path.join(output_subdir, f"{img_basename}_{model_handler.model_name}_comparison.png")
            combined_image.save(save_path)
            
            self.visualizations_count[dataset_name][model_handler.model_name] += 1
            logger.info(f"Saved PIL combined visualization for {dataset_name} with {model_handler.model_name} to {save_path}")

        except Exception as e:
            logger.error(f"Error saving PIL combined visualization: {e}", exc_info=True)

    def run_evaluation(self) -> Dict[str, Dict[str, Any]]:
        logger.info(f"Starting evaluation for {len(self.models)} models.")
        all_model_results = defaultdict(lambda: {
            'iou_data': defaultdict(lambda: {'all_sample_ious': [], 'all_inference_times_ms': [], 'num_samples_processed': 0}),
            'model_info': {}
        })

        benchmark_samples = self.dataset_loader.get_samples()
        if not benchmark_samples:
            logger.error("No samples loaded by BenchmarkDatasetLoader. Aborting evaluation.")
            return dict(all_model_results)

        for model_handler in self.models:
            model_name = model_handler.model_name
            logger.info(f"--- Evaluating Model: {model_name} ---")
            all_model_results[model_name]['model_info'] = model_handler.get_model_info()
            
            if benchmark_samples:
                first_sample = benchmark_samples[0]
                target_hw_warmup = model_handler.get_target_hw()
                warmup_input = model_handler.preprocess_input(first_sample, target_hw_warmup)
                if warmup_input is not None:
                    logger.info(f"Performing {self.config.warmup_runs} warmup runs for {model_name}...")
                    for _ in range(self.config.warmup_runs): 
                        _ = model_handler.run_inference(warmup_input)
                else:
                    logger.warning(f"Warmup input could not be preprocessed for {model_name}.")

            for sample in tqdm(benchmark_samples, desc=f"Processing {model_name}"):
                try:
                    img_path = sample['img_path']
                    ds_name = sample['dataset_key']
                    
                    current_target_hw = model_handler.get_target_hw() # H,W for model processing
                    input_for_handler = model_handler.preprocess_input(sample, current_target_hw)
                    if input_for_handler is None:
                        logger.warning(f"Preprocessing failed for {img_path} with model {model_name}. Skipping.")
                        continue

                    start_t = time.perf_counter()
                    iou_scores = model_handler.run_inference(input_for_handler)
                    inference_time_ms = (time.perf_counter() - start_t) * 1000

                    if iou_scores is None:
                        logger.warning(f"Inference failed for {img_path} with model {model_name}. Skipping.")
                        continue
                    
                    iou_score_full = iou_scores[0]
                    per_class_iou = iou_scores[1:4]
                        
                    res_io = all_model_results[model_name]['iou_data']
                    res_io[ds_name]['all_sample_ious'].append(per_class_iou)
                    res_io[ds_name]['all_inference_times_ms'].append(inference_time_ms)
                    res_io[ds_name]['num_samples_processed'] += 1
                    res_io[ds_name]['full_iou_sum'] = res_io[ds_name].get('full_iou_sum', 0.0) + iou_score_full
                    
                    res_io['Overall']['all_sample_ious'].append(per_class_iou)
                    res_io['Overall']['all_inference_times_ms'].append(inference_time_ms)
                    res_io['Overall']['num_samples_processed'] += 1
                    res_io['Overall']['full_iou_sum'] = res_io['Overall'].get('full_iou_sum', 0.0) + iou_score_full

                    if self.config.save_outputs and self.visualizations_count[ds_name][model_name] < self.config.visualizations_per_dataset:
                        pred_map_hw = None
                        if isinstance(model_handler, ONNXModel):
                            pred_map_hw = model_handler.get_segmentation_map(input_for_handler)
                        elif isinstance(model_handler, PyTorchModel):
                            pred_map_hw = model_handler.get_segmentation_map() # Uses internal state set by run_inference
                        
                        if pred_map_hw is not None:
                            self._save_pil_combined_visualization(
                                sample, pred_map_hw, model_handler, ds_name, current_target_hw
                            )
                except Exception as e_sample:
                    logger.error(f"Error processing sample {sample.get('img_path', 'unknown')} with model {model_name}: {e_sample}", exc_info=True)

        # Aggregate final metrics
        for model_name_res in all_model_results.keys():
            for key_ds_overall, data_dict in all_model_results[model_name_res]['iou_data'].items():
                if data_dict['all_sample_ious']:
                    avg_ious = np.mean(np.array(data_dict['all_sample_ious']), axis=0)
                    data_dict['avg_ious_per_class'] = avg_ious.tolist()
                else:
                    data_dict['avg_ious_per_class'] = [0.0] * len(CLASS_NAMES)
                
                num_processed = data_dict['num_samples_processed']
                if num_processed > 0:
                    data_dict['mIoU'] = data_dict.get('full_iou_sum', 0.0) / num_processed
                else:
                    data_dict['mIoU'] = 0.0
                
                data_dict['avg_inference_time_ms'] = float(np.mean(data_dict['all_inference_times_ms'])) if data_dict['all_inference_times_ms'] else 0.0
        
        return dict(all_model_results)

################################################################################
# Reporting Functions
################################################################################
def format_single_model_iou_table(model_name_report: str, results_for_model: Dict[str, Any]) -> str:
    """Formats a string table summarizing mIoU and per-class IoU scores for a single model.

    The table includes scores for the "Overall" category and for each dataset processed.
    It also lists the number of processed samples and average inference time.

    Args:
        model_name_report: The name of the model for the report header.
        results_for_model: The results dictionary for the model, as produced by Benchmarker.

    Returns:
        A formatted string representing the IoU table for the model.
    """
    if 'iou_data' not in results_for_model or not results_for_model['iou_data']:
        return f"--- {model_name_report} mIoU Scores ---\nNo IoU data available.\n"

    header = f"--- {model_name_report} mIoU Scores ---"
    iou_data = results_for_model['iou_data']
    # Ensure 'Overall' is present
    if 'Overall' not in iou_data: iou_data['Overall'] = {'avg_ious_per_class': [0.0]*len(CLASS_NAMES), 'mIoU':0.0, 'num_samples_processed':0, 'avg_inference_time_ms':0.0}
    
    dataset_keys = sorted([k for k in iou_data.keys() if k != 'Overall' and iou_data[k].get('num_samples_processed', 0) > 0])
    columns = ["Metric"] + ["Overall"] + dataset_keys
    
    table_str = f"{header}\n{' | '.join(columns)}\n{'|-' * len(columns)}\n"
    
    # First row is always full mIoU (calculated as sum/count)
    row_values = ["mIoU (full)"]
    # Overall column
    overall_data_dict = iou_data['Overall']
    val = overall_data_dict.get('mIoU', 0.0)
    row_values.append(f"{val:.3f}")
    # Per-dataset columns
    for ds_key in dataset_keys:
        ds_data_dict = iou_data.get(ds_key, {})
        val_ds = ds_data_dict.get('mIoU', 0.0)
        row_values.append(f"{val_ds:.3f}")
    table_str += f"{' | '.join(row_values)}\n"
    
    # Add per-class IoU rows
    class_names = CLASS_NAMES
    for class_idx, class_name in enumerate(class_names):
        row_values = [f"{class_name} IoU"]
        # Overall column
        overall_data_dict = iou_data['Overall']
        val = overall_data_dict.get('avg_ious_per_class', [0.0]*len(class_names))[class_idx]
        row_values.append(f"{val:.3f}")
        # Per-dataset columns
        for ds_key in dataset_keys:
            ds_data_dict = iou_data.get(ds_key, {})
            val_ds = ds_data_dict.get('avg_ious_per_class', [0.0]*len(class_names))[class_idx]
            row_values.append(f"{val_ds:.3f}")
        table_str += f"{' | '.join(row_values)}\n"
    
    num_s_overall = iou_data['Overall'].get('num_samples_processed', 0)
    processed_samples_str = f"Processed Samples (Overall): {num_s_overall}"
    for ds_key in dataset_keys:
        num_s_ds = iou_data[ds_key].get('num_samples_processed', 0)
        if num_s_ds > 0: processed_samples_str += f", {ds_key}: {num_s_ds}"
    table_str += f"\n{processed_samples_str}\n" if num_s_overall > 0 else "\nNo samples processed for IoU.\n"

    avg_time_overall = iou_data['Overall'].get('avg_inference_time_ms', 0.0)
    table_str += f"Avg. Inference Time (Overall): {avg_time_overall:.2f} ms\n"
    
    model_info_dict = results_for_model.get('model_info', {})
    model_size_mb = model_info_dict.get('size_mb', 0.0)
    if model_size_mb > 0 : table_str += f"Model Size: {model_size_mb:.2f} MB\n"
    
    return table_str

def generate_full_comparison_report(all_results: Dict[str, Dict[str, Any]], config: BenchmarkConfig) -> None:
    """Generates and logs a comprehensive comparison report for all benchmarked models.

    This function first logs the detailed IoU table for each model using
    format_single_model_iou_table. Then, it prints a summary table comparing
    key performance indicators (mIoU, average time, model size) across all models.

    Args:
        all_results: A dictionary containing results for all benchmarked models.
        config: The BenchmarkConfig object (currently unused in this function but
                could be used for future enhancements to the report).
    """
    logger.info("\n" + "="*70 + "\nCOMPREHENSIVE MODEL COMPARISON REPORT\n" + "="*70)
    for model_name_key, results_for_one_model in all_results.items():
        logger.info(format_single_model_iou_table(model_name_key, results_for_one_model))
    
    logger.info("\n--- Key Performance Indicators (Overall) ---")
    # Simple comparison for now, can be expanded
    for model_name_key, results_for_one_model in all_results.items():
        overall_data = results_for_one_model.get('iou_data',{}).get('Overall',{})
        miou = overall_data.get('mIoU',0.0)
        avg_time = overall_data.get('avg_inference_time_ms',0.0)
        size_mb = results_for_one_model.get('model_info',{}).get('size_mb')
        logger.info(f"Model: {model_name_key:<15} | mIoU: {miou:.4f} | Avg Time: {avg_time:>7.2f} ms {f'| Size: {size_mb:.2f} MB' if size_mb else ''}")
    logger.info("="*70 + "\n")

################################################################################
# Main Orchestration
################################################################################
def main() -> int:
    """Main function to parse arguments, set up, and run the benchmarking process.

    Parses command-line arguments, initializes configuration and loaders,
    instantiates model handlers, runs the benchmarking through the Benchmarker class,
    and finally generates the report.

    Returns:
        0 on successful completion, 1 on error.
    """
    parser = argparse.ArgumentParser(description="Benchmark PyTorch and ONNX models with mIoU.")
    parser.add_argument("--pytorch_model_path", type=str, help="Path to PyTorch (.pth) model checkpoint.")
    parser.add_argument("--fp32_model_path", type=str, help="Path to FP32 ONNX model.")
    parser.add_argument("--int8_model_path", type=str, help="Path to INT8 ONNX model.")
    parser.add_argument("--dataset_root", type=str, help="Root directory of segmentation datasets.")
    parser.add_argument("--num_samples_per_dataset", type=int, default=20, help="Max samples per dataset (default: 20).")
    parser.add_argument("--warmup_runs", type=int, default=3, help="Warmup runs (default: 3).")
    parser.add_argument("--save_outputs", action="store_true", help="Save output visualizations.")
    parser.add_argument("--output_dir_base", type=str, default="./benchmark_multimodel_results", help="Base dir for outputs.")
    parser.add_argument("--pytorch_target_hw", type=str, help="Target H,W for PyTorch model e.g., '320,640'. If None, uses SceneSegTrainer default.")
    parser.add_argument("--onnx_target_hw", type=str, help="Target H,W for ONNX models e.g., '320,640'. If None, derived from FP32 model.")
    parser.add_argument("--device", type=str, default="cpu", help="Device for PyTorch model ('cpu' or 'cuda:0' etc.)")
    parser.add_argument("--visualizations_per_dataset", type=int, default=2, help="Number of visualizations to save per dataset (default: 2).")

    args = parser.parse_args()
    
    # Parse H,W string arguments
    pytorch_hw = None
    if args.pytorch_target_hw:
        try: pytorch_hw = tuple(map(int, args.pytorch_target_hw.split(',')))
        except: logger.error("Invalid --pytorch_target_hw format. Use H,W (e.g., 320,640).")
    
    onnx_hw_override = None
    if args.onnx_target_hw:
        try: onnx_hw_override = tuple(map(int, args.onnx_target_hw.split(',')))
        except: logger.error("Invalid --onnx_target_hw format.")

    # Create the config directly from args
    config = BenchmarkConfig(
        pytorch_model_path=args.pytorch_model_path,
        fp32_model_path=args.fp32_model_path,
        int8_model_path=args.int8_model_path,
        dataset_root=args.dataset_root,
        num_samples_per_dataset=args.num_samples_per_dataset,
        warmup_runs=args.warmup_runs,
        save_outputs=args.save_outputs,
        output_dir_base=args.output_dir_base,
        pytorch_target_hw=pytorch_hw,
        onnx_target_hw=onnx_hw_override,
        visualizations_per_dataset=args.visualizations_per_dataset
    )

    logger.info(f"Initializing Multi-Model Benchmarking with Config: {config}")
    logger.info(f"System: {platform.system()} {platform.release()}, Python: {platform.python_version()}, ONNXRuntime: {onnxruntime.__version__}, PyTorch: {torch.__version__ if PYTORCH_AVAILABLE else 'N/A'}")

    if not config.pytorch_model_path and not config.fp32_model_path and not config.int8_model_path:
        logger.error("No models specified for benchmarking. Exiting.")
        return 1

    # Setup output directory
    if config.save_outputs:
        try:
            os.makedirs(config.output_dir_base, exist_ok=True)
            # Subdirectory for visualizations
            os.makedirs(os.path.join(config.output_dir_base, "visualizations"), exist_ok=True) 
            logger.info(f"Outputs will be saved under: {config.output_dir_base}")
        except Exception as e:
            logger.error(f"Could not create output directory {config.output_dir_base}: {e}. Disabling output saving.")
            config.save_outputs = False
    
    try:
        dataset_loader = BenchmarkDatasetLoader(config.dataset_root, config.num_samples_per_dataset)
        if not dataset_loader.get_samples(): return 1 # Exit if no data
    except RuntimeError as e: # Handles LoadDataSceneSeg unavailability
        logger.error(f"Dataset loading failed: {e}. Exiting.")
        return 1

    models_to_evaluate: List[ModelInterface] = []
    
    # Determine ONNX target H,W (from FP32 model if available and no override)
    final_onnx_target_hw = config.onnx_target_hw
    if not final_onnx_target_hw and config.fp32_model_path:
        temp_onnx_fp32_handler = ONNXModel(config.fp32_model_path, "temp_fp32_for_shape")
        if temp_onnx_fp32_handler.load_model():
            try:
                final_onnx_target_hw = temp_onnx_fp32_handler.get_target_hw()
                logger.info(f"Derived ONNX target H,W from FP32 model: {final_onnx_target_hw}")
            except ValueError: # Raised if get_target_hw fails
                pass # Keep final_onnx_target_hw as None, will be handled by individual ONNXModel init
        del temp_onnx_fp32_handler

    # Load models based on provided paths
    if config.pytorch_model_path and PYTORCH_AVAILABLE:
        pytorch_handler = PyTorchModel(config.pytorch_model_path, "PyTorch_FP32", args.device, config.pytorch_target_hw)
        if pytorch_handler.load_model(): models_to_evaluate.append(pytorch_handler)
    
    if config.fp32_model_path:
        onnx_fp32_handler = ONNXModel(config.fp32_model_path, "ONNX_FP32", target_hw_override=final_onnx_target_hw)
        if onnx_fp32_handler.load_model(): models_to_evaluate.append(onnx_fp32_handler)

    if config.int8_model_path:
        onnx_int8_handler = ONNXModel(config.int8_model_path, "ONNX_INT8", target_hw_override=final_onnx_target_hw)
        if onnx_int8_handler.load_model(): models_to_evaluate.append(onnx_int8_handler)

    if not models_to_evaluate:
        logger.error("No models were successfully loaded for benchmarking. Exiting.")
        return 1

    benchmarker_orchestrator = Benchmarker(config, models_to_evaluate, dataset_loader)
    all_results = benchmarker_orchestrator.run_evaluation()

    generate_full_comparison_report(all_results, config)
    
    logger.info("Benchmarking process finished.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 