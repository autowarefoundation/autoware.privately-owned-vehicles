import os
import glob
import cv2
import numpy as np
from Models.data_utils.lite_models.dataloaders.BaseDataset import BaseDataset

"""
Tree structure. This script uses for now only the v1.2 version, wihch has been adapted to be cityscapes
compatible (labels_cs).

Run preprocess_mapillary.py to create the label_cs gt labels. this is needed in order to have cityscapes compatible 
labels (original) mapillary is not compatible with cityscapes ids.
.ROOT
├── testing
│   └── images
├── training
│   ├── images
│   ├── v1.2
│   │   ├── instances
│   │   ├── labels
│   │   ├── labels_cs
│   │   ├── depth
│   │   └── panoptic
│   └── v2.0
│       ├── instances
│       ├── labels
│       ├── panoptic
│       └── polygons
└── validation
    ├── images
    ├── v1.2
    │   ├── instances
    │   ├── labels
    │   ├── labels_cs
    │   ├── depth
    │   └── panoptic
    └── v2.0
        ├── instances
        ├── labels
        ├── panoptic
        └── polygons
"""
class MapillaryDataset(BaseDataset):
    def __init__(self, dataset_root: str, aug_cfg: dict = {}, mode="train", data_type="SEGMENTATION", pseudo_labeling=False):
        
        super().__init__(dataset_root, aug_cfg=aug_cfg, mode=mode, data_type=data_type, pseudo_labeling=pseudo_labeling)
        """
        cfg contains:
        - root
        - conditions
        - split ("train"/"val"/"test")
        - augmentations (dict)

        """
        self.root = dataset_root
        self.split = mode  #train, val

        self.pseudo_labeling = pseudo_labeling

        # ---- Build file lists ----
        self.samples = self._build_file_list()

        self.dataset_name = "mapillary"


    def _build_file_list(self):
        samples = []
        print(f"[MapillaryDataset] Building file list for split '{self.split}'..., data_type='{self.data_type}'")

        #building folders and suffixes based on self.data_type
        if self.data_type == "SEGMENTATION":
            labels_folder = "labels_cs"   #we use the cityscapes compatible labels created with preprocess_mapillary.py
            suffix = "*.png"
        elif self.data_type == "DEPTH":
            labels_folder = "depth"
            suffix = "*_depth.npy"
        else:
            raise ValueError(f"[MapillaryDataset] ERROR: unsupported data_type: {self.data_type}")
        
        if self.mode == "train":
            print(f"[MapillaryDataset] Loading TRAINING samples...")
            self.root = os.path.join(self.root, "training") #pair with "training" or "validation"

        else:
            print(f"[MapillaryDataset] Loading VALIDATION samples...")
            self.root = os.path.join(self.root, "validation") #pair with "training" or "validation"


        #code is the same, regardless of train/val for v1.2. only change is inside the training or validation subfolder
        img_root = os.path.join(self.root, "images")
        
        #get all images recursively. example : "/home/sergey/DEV/AI/datasets/mapillary/training/images/__CRyFzoDOXn6unQ6a3DnQ.jpg"
        img_files = glob.glob(os.path.join(img_root, "*.jpg"),recursive=True)   #list of images

        print(f"[MapillaryDataset] Found {len(img_files)} images under: {img_root}")

        #now get the masks inside the v1.2/labels_cs folder (created with preprocess_mapillary.py)

        gt_label = os.path.join(self.root, "v1.2", labels_folder)

        gt_files = glob.glob(os.path.join(gt_label, suffix),recursive=True)

        print(f"[MapillaryDataset] Found {len(gt_files)} GT masks under: {gt_label}")

        #map images to their gt masks
        assert len(img_files) == len(gt_files), "[MapillaryDataset] ERROR: number of images and GT masks do not match!"

        for img_path in img_files:
            # Example filename:
            # /home/sergey/DEV/AI/datasets/mapillary/training/images/__CRyFzoDOXn6unQ6a3DnQ.jpg
            filename = os.path.basename(img_path)
            base = filename.replace(".jpg", "")

            #construct the gt path, depending on the data_type
            gt_filename = base + suffix.replace("*", "")
            gt_path = os.path.join(gt_label, gt_filename)


            #append even if pseudo labeling is used (we will generate fake GT later)
            if os.path.isfile(gt_path):
                samples.append((img_path, gt_path))
            else:
                print(f"[MapillaryDataset] WARNING: no GT for {img_path}")

        #return the list of samples (image_path, gt_path)
        return samples

    def __getitem__(self, idx):
        img_path, gt_path = self.samples[idx]

        # --------------------------------------------------
        # 1) LOAD IMAGE (BGR → RGB)
        # --------------------------------------------------
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        # --------------------------------------------------
        # 2) LOAD / FAKE GT
        # --------------------------------------------------
        if self.pseudo_labeling is False:
            if self.data_type == "SEGMENTATION":
                label = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
            elif self.data_type == "DEPTH":
                label = np.load(gt_path)
            else:
                raise ValueError(
                    f"[BaseDataset] ERROR: unsupported data_type: {self.data_type}"
                )
        else:
            # fake GT (placeholder, will be ignored)
            if self.data_type == "SEGMENTATION":
                label = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            elif self.data_type == "DEPTH":
                label = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        # --------------------------------------------------
        # 3) PRE-RESIZE FOR VERY LARGE IMAGES (SEG + DEPTH)
        # --------------------------------------------------
        scale = 0.5
        h, w = image.shape[:2]

        if h > 768*2 or w > 1024*2:
            new_h = int(h * scale)
            new_w = int(w * scale)

            image = cv2.resize(
                image,
                (new_w, new_h),
                interpolation=cv2.INTER_LINEAR,
            )

            label = cv2.resize(
                label,
                (new_w, new_h),
                interpolation=cv2.INTER_NEAREST,
            )

        # --------------------------------------------------
        # 4) AUGMENTATIONS
        # --------------------------------------------------
        image, label = self.aug.apply_augmentation(image, label, dataset_name=self.dataset_name)

        # --------------------------------------------------
        # 5) FINAL CAST
        # --------------------------------------------------
        image = image.astype(np.float32)
        label = label.astype(np.int64)

        image = np.transpose(image, (2, 0, 1))  # CHW

        # --------------------------------------------------
        # 6) RETURN SAMPLE
        # --------------------------------------------------
        sample = {
            "image": image,
            "gt": label,
        }


        return sample



"""
Cityscapes labels
labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]"""
