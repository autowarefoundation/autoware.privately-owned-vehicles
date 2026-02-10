#!/usr/bin/env python3
# compute_class_weights_separate.py

import os
import argparse
import numpy as np
import yaml
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from utils.training import (
    build_single_dataset,
    load_yaml,
    set_global_seed,
)


# ============================================================
# UTILITIES
# ============================================================

def compute_histogram(dset, num_classes, ignore_index):
    """
    Conta i pixel per classe di un singolo dataset, usando i gt ORIGINALI
    (nessun resize).
    """
    loader = DataLoader(
        dset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    counts = np.zeros(num_classes, dtype=np.int64)

    for batch in tqdm(loader, desc=f"[{dset.name}] counting", ncols=120):
        gt = batch["gt"].cpu().numpy()  # (1,H,W)
        flat = gt.reshape(-1)
        valid = flat != ignore_index
        lab = flat[valid]
        if lab.size > 0:
            counts += np.bincount(lab, minlength=num_classes)

    return counts


def compute_weight_variants(counts, eps=1e-6):
    """
    Calcola tutte le varianti dei pesi.
    """
    total = counts.sum() + eps
    freq = counts / total

    W = {"freq": freq}

    # inverse
    W["inv_raw"] = 1.0 / (freq + eps)
    W["inv_norm"] = W["inv_raw"] / (W["inv_raw"].mean() + eps)
    W["inv_norm_clamp"] = np.clip(W["inv_norm"], 0.1, 10.0)

    # median frequency balancing
    nz = freq > 0
    if np.any(nz):
        med = np.median(freq[nz])
    else:
        med = 1.0
    W["mfb"] = med / (freq + eps)

    # log-based
    for k in [1.01, 1.02, 1.05, 1.10]:
        W[f"log_k_{k}"] = 1.0 / np.log(k + freq + eps)

    # log normalized
    for k in [1.01, 1.02]:
        arr = 1.0 / np.log(k + freq + eps)
        W[f"log_k_{k}_norm"] = arr / (arr.mean() + eps)

    # sqrt inverse
    W["sqrt_inv"] = 1.0 / np.sqrt(freq + eps)
    W["sqrt_inv_norm"] = W["sqrt_inv"] / (W["sqrt_inv"].mean() + eps)
    W["sqrt_inv_norm_clamp"] = np.clip(W["sqrt_inv_norm"], 0.1, 10.0)

    return W


def write_report(path, ds_name, counts, weights, class_names):
    """
    Scrive un singolo file TXT con il report per UN singolo dataset.
    """
    with open(path, "w") as f:
        f.write("=============================================================\n")
        f.write(f" CLASS WEIGHTS REPORT – {ds_name}\n")
        f.write("=============================================================\n\n")

        total = counts.sum()
        f.write(f"Total labeled pixels: {int(total)}\n\n")

        # class table
        f.write("idx | class_name            | count         | freq\n")
        f.write("-------------------------------------------------------------\n")
        freq = weights["freq"]

        for c in range(len(counts)):
            cname = class_names.get(c, f"class_{c}")
            f.write(f"{c:2d} | {cname:20s} | {counts[c]:12d} | {freq[c]:.8f}\n")

        # weights
        f.write("\n\n----------- WEIGHT ARRAYS -----------\n\n")

        for key, arr in weights.items():
            if key == "freq":
                continue
            arr_str = ", ".join([f"{x:.6f}" for x in arr])
            f.write(f"{key}: [{arr_str}]\n")

        f.write("\nDone.\n")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config",
        default="configs/sceneseg.yaml",
        help="Path to main training YAML"
    )
    args = parser.parse_args()

    # load config
    cfg = load_yaml(args.config)
    set_global_seed(cfg["experiment"].get("seed", 42))

    # dataset info
    ds_cfg = cfg["dataset"]
    training_sets = ds_cfg["training_sets"]

    # class info
    loss_cfg = cfg["loss"]
    num_classes = loss_cfg["num_classes"]
    ignore_index = loss_cfg["ignore_index"]
    class_names = {int(k): v for k, v in loss_cfg["class_names"].items()}

    # output folder
    out_root = os.path.join("utils", "weights")
    os.makedirs(out_root, exist_ok=True)

    # storage for global merging
    global_counts = np.zeros(num_classes, dtype=np.int64)

    # ========================================================
    # PER-DATASET REPORTS
    # ========================================================
    for ds_name in training_sets:

        key = f"{ds_name.lower()}_path"
        if key not in ds_cfg:
            raise ValueError(f"Missing dataset path for {ds_name}")

        dset = build_single_dataset(ds_name, ds_cfg[key], split="train", is_train=True)
        dset.name = ds_name

        print(f"\nProcessing dataset: {ds_name}")

        # compute pixel histogram
        counts = compute_histogram(dset, num_classes, ignore_index)
        global_counts += counts

        # compute weight variants
        weights = compute_weight_variants(counts)

        # write txt
        out_path = os.path.join(out_root, f"{ds_name}_weights.txt")
        write_report(out_path, ds_name, counts, weights, class_names)

        print(f"Saved: {out_path}")

    # ========================================================
    # GLOBAL REPORT (all datasets combined)
    # ========================================================

    print("\nBuilding GLOBAL report…")

    global_weights = compute_weight_variants(global_counts)
    global_path = os.path.join(out_root, "global_weights.txt")

    write_report(global_path, "GLOBAL", global_counts, global_weights, class_names)

    print(f"Saved GLOBAL: {global_path}")
    print("\nDONE ✓")


if __name__ == "__main__":
    main()
