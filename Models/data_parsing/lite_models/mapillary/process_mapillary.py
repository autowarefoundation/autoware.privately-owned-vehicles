#!/usr/bin/env python3
"""
Convert Mapillary Vistas RGB masks → Cityscapes 19-class single-channel masks.

INPUT:
  - Mapillary JSON (v1.2 or v2.0)
  - Folder with RGB masks (Mapillary format)
  - Mapping dict (Mapillary-name → Cityscapes train ID)

OUTPUT:
  - Single-channel label images (uint8, Cityscapes train IDs)
  - A mapping summary file (mapillary_to_cityscapes_summary.txt)
  - Multi-processing with tqdm progress bar

"""

import os
import sys
import json
import argparse
import multiprocessing as mp
from functools import partial

from PIL import Image
import numpy as np
from tqdm import tqdm


# -----------------------------------------------------------
# Small logging helpers
# -----------------------------------------------------------

def log(msg: str):
    print(msg, flush=True)


def log_err(msg: str):
    print("[ERROR] " + msg, file=sys.stderr, flush=True)


# -----------------------------------------------------------
# LOAD JSON → GET MAPILLARY LABEL INFO
# -----------------------------------------------------------

def load_mapillary_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["labels"]


# -----------------------------------------------------------
# BUILD RGB→ID and NAME→ID MAPPINGS
# -----------------------------------------------------------

def build_rgb_and_name_mappings(labels):
    """
    Returns:
        rgb_to_id[(r,g,b)] = mapillary_id
        id_to_name[map_id] = "construction--flat--road"
    """
    rgb_to_id = {}
    id_to_name = {}

    for map_id, entry in enumerate(labels):
        r, g, b = entry["color"]
        name = entry["name"]
        rgb_to_id[(r, g, b)] = map_id
        id_to_name[map_id] = name

    return rgb_to_id, id_to_name


# -----------------------------------------------------------
# CONVERTER FOR A SINGLE MASK
# -----------------------------------------------------------

def convert_mask(in_path, out_path, rgb_to_id, id_to_city, rgb_encoded_to_id):
    """
    Convert one Mapillary RGB (or paletted) mask to a single-channel
    Cityscapes train-ID mask and save it to out_path.
    """

    try:
        img = Image.open(in_path)

        # Ensure RGB (some masks are paletted 'P')
        if img.mode != "RGB":
            img = img.convert("RGB")

        mask_rgb = np.array(img)
        H, W = mask_rgb.shape[:2]
        out = np.zeros((H, W), dtype=np.uint8)

        # Encode RGB → integer code
        codes = (
            mask_rgb[:, :, 0].astype(int) * 256 * 256 +
            mask_rgb[:, :, 1].astype(int) * 256 +
            mask_rgb[:, :, 2].astype(int)
        )

        unique_codes = np.unique(codes)

        # Assign Cityscapes IDs based on color codes
        for code in unique_codes:
            if code in rgb_encoded_to_id:
                mid = rgb_encoded_to_id[code]          # Mapillary class id
                cid = id_to_city.get(mid, 255)         # Cityscapes train-id or 255
                out[codes == code] = cid
            else:
                # Unknown color -> ignore
                out[codes == code] = 255

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        Image.fromarray(out).save(out_path)

    except Exception as e:
        log_err(f"Failed to process file: {in_path}\n{e}")


# -----------------------------------------------------------
# MULTIPROCESSING DRIVER
# -----------------------------------------------------------

def process_folder(input_dir, output_dir, rgb_to_id, id_to_city, num_workers=8):
    log(f"[SCAN] Input folder:  {input_dir}")
    log(f"[SCAN] Output folder: {output_dir}")

    # Gather all mask paths
    paths = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(".png"):
                in_path = os.path.join(root, f)
                rel = os.path.relpath(in_path, input_dir)
                out_path = os.path.join(output_dir, rel)
                paths.append((in_path, out_path))

    log(f"[INFO] Found {len(paths)} masks to convert.")

    # Precompute RGB-code → MapillaryID
    rgb_encoded_to_id = {
        (r * 256 * 256 + g * 256 + b): mid
        for (r, g, b), mid in rgb_to_id.items()
    }

    worker = partial(
        convert_mask,
        rgb_to_id=rgb_to_id,
        id_to_city=id_to_city,
        rgb_encoded_to_id=rgb_encoded_to_id,
    )

    log(f"[MP] Using {num_workers} workers...")

    chunk_size = 64  # faster multiprocessing

    with mp.Pool(num_workers) as pool:
        for _ in tqdm(
            pool.starmap(worker, paths, chunksize=chunk_size),
            total=len(paths),
            desc="Converting masks",
        ):
            pass

    log("[DONE] Completed processing all masks.")



# -----------------------------------------------------------
# SAVE SUMMARY FILE
# -----------------------------------------------------------

def save_mapping_summary(output_path, id_to_name, id_to_city):
    with open(output_path, "w") as f:
        for mid, name in id_to_name.items():
            cs = id_to_city.get(mid, 255)
            f.write(f"{mid:3d}  {name:50s} → {cs}\n")


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Mapillary JSON file (labels)")
    parser.add_argument("--input", required=True, help="Folder with RGB label masks")
    parser.add_argument("--output", required=True, help="Folder for converted masks")
    parser.add_argument("--mapping", required=True, help="Python file with MAPILLARY_TO_CITYSCAPES dict")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    # 1) Load JSON
    log("Loading Mapillary JSON...")
    labels = load_mapillary_json(args.json)

    # 2) Build color/name tables
    log("Building RGB and name lookup tables...")
    rgb_to_id, id_to_name = build_rgb_and_name_mappings(labels)

    # 3) Load mapping dict
    log("Loading class mapping file...")
    mapping_globals = {}
    with open(args.mapping, "r") as f:
        exec(f.read(), mapping_globals)

    if "MAPILLARY_TO_CITYSCAPES" not in mapping_globals:
        raise KeyError("Mapping file must define MAPILLARY_TO_CITYSCAPES dict")

    MAP = mapping_globals["MAPILLARY_TO_CITYSCAPES"]

    # Map Mapillary ID -> Cityscapes train ID
    id_to_city = {mid: MAP.get(name, 255) for mid, name in id_to_name.items()}

    # 4) Save summary file
    os.makedirs(args.output, exist_ok=True)
    summary_path = os.path.join(args.output, "mapillary_to_cityscapes_summary.txt")
    log(f"Saving mapping summary to: {summary_path}")
    save_mapping_summary(summary_path, id_to_name, id_to_city)

    # 5) Convert masks with multiprocessing + tqdm
    log("Starting mask conversion...")
    process_folder(args.input, args.output, rgb_to_id, id_to_city, args.workers)

    log("All done.")


if __name__ == "__main__":
    main()
