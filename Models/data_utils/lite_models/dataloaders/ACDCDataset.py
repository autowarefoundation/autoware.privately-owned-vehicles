import os
import glob

from Models.data_utils.lite_models.dataloaders.BaseDataset import BaseDataset


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


class ACDCDataset(BaseDataset):
    def __init__(self, dataset_root: str, aug_cfg: dict = {}, mode="train", data_type="SEGMENTATION", pseudo_labeling=False):
        super().__init__(dataset_root, aug_cfg=aug_cfg, mode=mode, data_type=data_type, pseudo_labeling=pseudo_labeling)
        """
        ACDC Dataset class

        """
        self.root = dataset_root
        self.conditions = ["fog", "night", "rain", "snow"]

        self.split = mode  #train, val, test

        self.dataset_name = "acdc"
        
        self.pseudo_labeling = pseudo_labeling

        # ---- Build file lists ----
        self.samples = self._build_file_list()


    def _build_file_list(self):
        samples = []
        print(f"[ACDCDataset] Building file list for split '{self.split}'..., data_type='{self.data_type}'")


        #changing inner folder and suffix depending on the data type
        if self.data_type == "SEGMENTATION":
            inner_folder = "gt"
            suffix = "gt_labelTrainIds.png"
        elif self.data_type == "DEPTH":
            #not used, since pseudo labeling is used on the fly
            inner_folder = "depth"
            suffix = "depth.npy"
        else:
            raise ValueError(f"Unknown data_type: {self.data_type}")
        


        for cond in self.conditions:

            img_root = os.path.join(
                self.root,
                "rgb_anon",
                cond,
                self.split
            )

            gt_root = os.path.join(
                self.root,
                "gt_trainval",
                inner_folder,       #depends on the data_type
                cond,
                self.split
            )

            # print(f"[ACDCDataset] Searching images under: {img_root}")
            # print(f"[ACDCDataset] GT root: {gt_root}")

            #get all images recursively
            img_files = glob.glob(
                os.path.join(img_root, "**", "*_rgb_anon.png"),
                recursive=True
            )

            for img_path in img_files:
                # Example filename:
                # GOPR0475_frame_000123_rgb_anon.png
                filename = os.path.basename(img_path)
                parent_seq = os.path.basename(os.path.dirname(img_path))

                base = filename.replace("_rgb_anon.png", "")


                #construct the gt path, by matching 100% every image with its label. this avoids images inside rgb_anon without labels
                gt_filename = base + "_" + suffix      #example : _gt_labelTrainIds.png for segmentation

                gt_path = os.path.join(gt_root, parent_seq, gt_filename)


                #load the gt path even if pseudo-labeling is used, (in that case the path is not used)
                if os.path.isfile(gt_path):
                    #load both image and GT
                    samples.append((img_path, gt_path))
                else:
                    print(f"[ACDCDataset] WARNING: no GT for {img_path}")

        print(f"[ACDCDataset] Loaded {len(samples)} samples for '{self.split}'")
        return samples