import os
import random
from torch.utils.data import Dataset
from PIL import Image
import json
import torchvision.transforms as T


class DDataset(Dataset):
    def __init__(self, root_dir, transform=None, routes=None):
        self.root_dir = root_dir
        self.transform = transform
        self.pairs = []

        if routes is None:
            routes = sorted([s for s in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, s))])

        for route in routes:
            route_path = os.path.join(root_dir, route)
            sequences = sorted([r for r in os.listdir(route_path) if os.path.isdir(os.path.join(route_path, r))])

            for sequence in sequences:
                route_path = os.path.join(route_path, sequence)
                metadata_file = os.path.join(route_path, "metadata.json")
                if not os.path.exists(metadata_file):
                    continue

                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

                frames = sorted(metadata["frames"], key=lambda x: x["timestamp"])

                for i in range(1, len(frames)):
                    img_T_minus_1_path = os.path.join(route_path, f"{frames[i - 1]['timestamp']}.jpg")
                    steering_angle_T_minus_1 = frames[i - 1]["steering_angle_corrected"]
                    img_T_path = os.path.join(route_path, f"{frames[i]['timestamp']}.jpg")
                    steering_angle_T = frames[i]["steering_angle_corrected"]

                    if os.path.exists(img_T_minus_1_path) and os.path.exists(img_T_path):
                        self.pairs.append((img_T_minus_1_path, steering_angle_T_minus_1, img_T_path, steering_angle_T))

        self.pairs = self._filter(self.pairs)

        print(f"Dataset created: {len(self.pairs)} image pairs from {len(routes)} sequences.")

    def _filter(self, pairs):
        intervals = [
            (1761806100565550080, 1761806286756780032),
            (1761806474351460096, 1761806628653689856),
            (1761806853293499904, 1761806988906460160),
            (1761807511835620096, 1761807761244930048)
        ]

        def in_any_interval(t):
            return any(start <= t <= end for start, end in intervals)

        pairs = [pair for pair in pairs if in_any_interval(int(os.path.splitext(os.path.basename(pair[2]))[0]))]
        pairs = [pair for pair in pairs if abs(pair[3]) <= 30]

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_T_minus_1_path, steering_angle_T_minus_1, img_T_path, steering_angle_T = self.pairs[idx]
        img_T_minus_1 = Image.open(img_T_minus_1_path).convert("RGB")
        img_T = Image.open(img_T_path).convert("RGB")

        if self.transform:
            img_T_minus_1 = self.transform(img_T_minus_1)
            img_T = self.transform(img_T)

        return img_T_minus_1, steering_angle_T_minus_1, img_T, steering_angle_T


class LoadDataAutoSteer:
    def __init__(self, root_dir, transform=None, train_split=0.9, seed=42):
        self.root_dir = root_dir
        self.transform = transform

        # --- Split sequences ---
        train_routes = sorted(["train"])
        val_routes = sorted(["val"])

        # --- Create dataset instances ---
        self.train = DDataset(root_dir, transform=transform, routes=train_routes)
        self.val = DDataset(root_dir, transform=transform, routes=val_routes)
