#!/usr/bin/env python3
import argparse
import os
import re
import math
from collections import defaultdict

# ------------------------------------------------------------
# Parsing utilities
# ------------------------------------------------------------

LINE_RE = re.compile(
    r"^\s*(\d+)\s*\|\s*([a-zA-Z0-9_ ]+?)\s*\|\s*(\d+)\s*\|\s*([0-9.eE+-]+)"
)

def parse_dataset_file(path):
    """
    Parses a *_weights.txt or *_dataset.txt file.

    Returns:
      class_names: dict[idx] -> name
      counts: dict[idx] -> count
      total_pixels: int
    """
    class_names = {}
    counts = defaultdict(int)

    with open(path, "r") as f:
        for line in f:
            if "|" not in line:
                continue

            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 4:
                continue

            idx_str, name, count_str, freq_str = parts[:4]

            if not idx_str.isdigit():
                continue

            try:
                idx = int(idx_str)
                count = int(count_str)
            except ValueError:
                continue

            class_names[idx] = name
            counts[idx] += count

    total_pixels = sum(counts.values())

    if total_pixels == 0:
        raise RuntimeError(
            f"[ERROR] Parsed zero pixels from {path}. "
            f"Check file format."
        )

    return class_names, counts, total_pixels


# ------------------------------------------------------------
# Weight computation
# ------------------------------------------------------------

def compute_weights(freqs, clamp_min=0.1):
    eps = 1e-12

    inv_raw = [1.0 / max(f, eps) for f in freqs]
    max_inv = max(inv_raw)

    inv_norm = [w / max_inv for w in inv_raw]
    inv_norm_clamp = [max(w, clamp_min) for w in inv_norm]

    sqrt_inv = [math.sqrt(w) for w in inv_raw]
    max_sqrt = max(sqrt_inv)
    sqrt_inv_norm = [w / max_sqrt for w in sqrt_inv]
    sqrt_inv_norm_clamp = [max(w, clamp_min) for w in sqrt_inv_norm]

    def log_k(k):
        return [math.log(1.0 + k / max(f, eps)) for f in freqs]

    def norm(arr):
        m = max(arr)
        return [v / m for v in arr]

    log_101 = log_k(0.01)
    log_102 = log_k(0.02)
    log_105 = log_k(0.05)
    log_11  = log_k(0.10)

    return {
        "inv_raw": inv_raw,
        "inv_norm": inv_norm,
        "inv_norm_clamp": inv_norm_clamp,
        "sqrt_inv": sqrt_inv,
        "sqrt_inv_norm": sqrt_inv_norm,
        "sqrt_inv_norm_clamp": sqrt_inv_norm_clamp,
        "log_k_1.01": log_101,
        "log_k_1.02": log_102,
        "log_k_1.05": log_105,
        "log_k_1.1":  log_11,
        "log_k_1.01_norm": norm(log_101),
        "log_k_1.02_norm": norm(log_102),
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--weights_dir", default="utils/weights/")
    parser.add_argument("--out", default="utils/weights/global_weights.txt")
    args = parser.parse_args()

    global_counts = defaultdict(int)
    class_names = {}
    total_pixels = 0

    # -------------------------------
    # Merge counts
    # -------------------------------
    for ds in args.datasets:
        path = os.path.join(args.weights_dir, f"{ds}_weights.txt")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

        print(f"[INFO] Reading {path}")
        names, counts, tot = parse_dataset_file(path)

        for idx, name in names.items():
            class_names[idx] = name

        for idx, c in counts.items():
            global_counts[idx] += c

        total_pixels += tot

    num_classes = max(global_counts.keys()) + 1

    freqs = [
        global_counts[i] / total_pixels
        for i in range(num_classes)
    ]

    weights = compute_weights(freqs)

    # -------------------------------
    # Write output
    # -------------------------------
    with open(args.out, "w") as f:
        f.write("=============================================================\n")
        f.write(f" CLASS WEIGHTS REPORT – GLOBAL ({' '.join(args.datasets)})\n")
        f.write("=============================================================\n\n")

        f.write(f"Total labeled pixels: {total_pixels}\n\n")
        f.write("idx | class_name            | count         | freq\n")
        f.write("-------------------------------------------------------------\n")

        for i in range(num_classes):
            f.write(
                f"{i:2d} | {class_names[i]:20s} | "
                f"{global_counts[i]:12d} | {freqs[i]:.8f}\n"
            )

        f.write("\n\n----------- WEIGHT ARRAYS -----------\n\n")

        for k, arr in weights.items():
            f.write(
                f"{k}: ["
                + ", ".join(f"{v:.6f}" for v in arr)
                + "]\n"
            )

        f.write("\nDone.\n")

    print(f"\n✔ Global class weights written to: {args.out}")


if __name__ == "__main__":
    main()
