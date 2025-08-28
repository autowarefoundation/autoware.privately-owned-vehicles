#! /usr/bin/env python3

import os
import cv2
import math
import json
import argparse
import warnings
import numpy as np
from PIL import Image, ImageDraw

# Custom warning format cuz the default one is wayyyyyy too verbose
def custom_warning_format(message, category, filename, lineno, line=None):
    return f"WARNING : {message}\n"

warnings.formatwarning = custom_warning_format

PointCoords = tuple[float, float]
ImagePointCoords = tuple[int, int]

# Skipped frames
skipped_dict = {}

# ============================== Helper functions ============================== #


def log_skipped(frame_id, reason):
    skipped_dict[frame_id] = reason


def roundLineFloats(line, ndigits = 4):
    line = list(line)
    for i in range(len(line)):
        line[i] = [
            round(line[i][0], ndigits),
            round(line[i][1], ndigits)
        ]
    line = tuple(line)
    return line


def normalizeCoords(line, width, height):
    """
    Normalize the coords of line points.
    """
    return [
        (x / width, y / height) 
        for x, y in line
    ]


def renormalizeCoords(line, width, height):
    """
    Renormalize the coords of line points.
    """
    return [
        (x * width, y * height) 
        for x, y in line
    ]


def interpLine(line: list, points_quota: int):
    """
    Interpolates a line of (x, y) points to have at least `point_quota` points.
    This helps with CurveLanes since most of its lines have so few points, 2~3.
    """
    if len(line) >= points_quota:
        return line

    # Extract x, y separately then parse to interp
    x = np.array([pt[0] for pt in line])
    y = np.array([pt[1] for pt in line])
    interp_x = np.interp
    interp_y = np.interp

    # Here I try to interp more points along the line, based on
    # distance between each subsequent original points. 

    # 1) Use distance along line as param (t)
    # This is Euclidian distance between each point and the one before it
    distances = np.cumsum(np.sqrt(
        np.diff(x, prepend = x[0])**2 + \
        np.diff(y, prepend = y[0])**2
    ))
    # Force first t as zero
    distances[0] = 0

    # 2) Generate new t evenly spaced along original line
    evenly_t = np.linspace(distances[0], distances[-1], points_quota)

    # 3) Interp x, y coordinates based on evenly t
    x_new = interp_x(evenly_t, distances, x)
    y_new = interp_y(evenly_t, distances, y)

    return list(zip(x_new, y_new))


def getLineAnchor(line):
    """
    Determine "anchor" point of a lane.

    """
    (x2, y2) = line[0]
    (x1, y1) = line[1]

    for i in range(len(line) - 2, 0, -1):
        if (line[i][0] != x2):
            (x1, y1) = line[i]
            break

    if (x1 == x2) or (y1 == y2):
        if (x1 == x2):
            error_lane = "Vertical"
        elif (y1 == y2):
            error_lane = "Horizontal"
        warnings.warn(f"{error_lane} line detected: {line}, with these 2 anchors: ({x1}, {y1}), ({x2}, {y2}).")
        return (x1, None, None)
    
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    x0 = (H - b) / a
    
    return (x0, a, b)


def drawLine(
    img: np.ndarray, 
    line: list,
    color: tuple,
    thickness: int = 2
):
    for i in range(1, len(line)):
        pt1 = (
            int(line[i - 1][0]), 
            int(line[i - 1][1])
        )
        pt2 = (
            int(line[i][0]), 
            int(line[i][1])
        )
        cv2.line(
            img, 
            pt1, pt2, 
            color = color, 
            thickness = thickness
        )


def annotateGT(
    img: np.ndarray,
    orig_img: np.ndarray,
    frame_id: str,
    bev_egopath: list,
    reproj_egopath: list,
    bev_egoleft: list,
    reproj_egoleft: list,
    bev_egoright: list,
    reproj_egoright: list,
    raw_dir: str, 
    visualization_dir: str,
    normalized: bool
):
    """
    Annotates and saves an image with:
        - Raw image, in "output_dir/image".
        - Annotated image with all lanes, in "output_dir/visualization".
    """

    # =========================== RAW IMAGE =========================== #

    # Save raw img in raw dir, as PNG
    cv2.imwrite(
        os.path.join(
            raw_dir,
            f"{frame_id}.png"
        ),
        img
    )

    # =========================== BEV VIS =========================== #

    img_bev_vis = img.copy()
    h, w, _ = img_bev_vis.shape

    # Draw egopath
    if (normalized):
        renormed_bev_egopath = [
            (x * w, y * h) 
            for x, y in bev_egopath
        ]
    else:
        renormed_bev_egopath = bev_egopath
    drawLine(
        img = img_bev_vis,
        line = renormed_bev_egopath,
        color = COLOR_EGOPATH
    )
    
    # Draw egoleft
    if (normalized):
        renormed_bev_egoleft = [
            (x * w, y * h) 
            for x, y in bev_egoleft
        ]
    else:
        renormed_bev_egoleft = bev_egoleft
    drawLine(
        img = img_bev_vis,
        line = renormed_bev_egoleft,
        color = COLOR_EGOLEFT
    )

    # Draw egoright
    if (normalized):
        renormed_bev_egoright = [
            (x * w, y * h) 
            for x, y in bev_egoright
        ]
    else:
        renormed_bev_egoright = bev_egoright
    drawLine(
        img = img_bev_vis,
        line = renormed_bev_egoright,
        color = COLOR_EGORIGHT
    )

    # Save visualization img in vis dir, as JPG (saving storage space)
    cv2.imwrite(
        os.path.join(
            visualization_dir,
            f"{frame_id}.jpg"
        ),
        img_bev_vis
    )

    # =========================== ORIGINAL VIS =========================== #

    # Draw reprojected egopath
    if (normalized):
        renormed_reproj_egopath = [
            (x * w, y * h) 
            for x, y in reproj_egopath
        ]
    else:
        renormed_reproj_egopath = reproj_egopath
    drawLine(
        img = orig_img,
        line = renormed_reproj_egopath,
        color = COLOR_EGOPATH
    )
    
    # Draw reprojected egoleft
    if (normalized):
        renormed_reproj_egoleft = [
            (x * w, y * h) 
            for x, y in reproj_egoleft
        ]
    else:
        renormed_reproj_egoleft = reproj_egoleft
    drawLine(
        img = orig_img,
        line = renormed_reproj_egoleft,
        color = COLOR_EGOLEFT
    )

    # Draw reprojected egoright
    if (normalized):
        renormed_reproj_egoright = [
            (x * w, y * h) 
            for x, y in reproj_egoright
        ]
    else:
        renormed_reproj_egoright = reproj_egoright
    drawLine(
        img = orig_img,
        line = renormed_reproj_egoright,
        color = COLOR_EGORIGHT
    )

    # Save it
    cv2.imwrite(
        os.path.join(
            visualization_dir,
            f"{frame_id}_orig.jpg"
        ),
        orig_img
    )


def interpX(line, y):
    """
    Interpolate x-value of a point on a line, given y-value
    """
    points = np.array(line)
    list_x = points[:, 0]
    list_y = points[:, 1]

    if not np.all(np.diff(list_y) > 0):
        sort_idx = np.argsort(list_y)
        list_y = list_y[sort_idx]
        list_x = list_x[sort_idx]

    return float(np.interp(y, list_y, list_x))


def polyfit_BEV(
    bev_line: list,
    order: int,
    y_step: int,
    y_limit: int
):
    valid_line = [
        point for point in bev_line
        if (0 <= point[0] < BEV_W) and (0 <= point[1] < BEV_H)
    ]
    x = [point[0] for point in valid_line]
    y = [point[1] for point in valid_line]
    z = np.polyfit(y, x, order)
    f = np.poly1d(z)
    y_new = np.linspace(
        0, y_limit, 
        int(y_limit / y_step) + 1
    )
    x_new = f(y_new)

    # Sort by decreasing y
    fitted_bev_line = sorted(
        tuple(zip(x_new, y_new)),
        key = lambda x: x[1],
        reverse = True
    )

    flag_list = [0] * len(fitted_bev_line)
    for i in range(len(fitted_bev_line)):
        if (not 0 <= fitted_bev_line[i][0] <= BEV_W):
            flag_list[i - 1] = 1
            break
    if (not 1 in flag_list):
        flag_list[-1] = 1

    validity_list = [1] * len(fitted_bev_line)
    last_valid_index = flag_list.index(1)
    for i in range(last_valid_index + 1, len(validity_list)):
        validity_list[i] = 0
    
    return fitted_bev_line, flag_list, validity_list


def imagePointTuplize(point: PointCoords) -> ImagePointCoords:
    """
    Parse all coords of an (x, y) point to int, making it
    suitable for image operations.
    """
    return (int(point[0]), int(point[1]))


def findSourcePointsBEV(
    h: int,
    w: int,
    egoleft: list,
    egoright: list,
) -> dict:
    """
    Find 4 source points for the BEV homography transform.
    """

    # Renorm 2 egolines
    egoleft = [
        [p[0] * w, p[1] * h]
        for p in egoleft
    ]
    egoright = [
        [p[0] * w, p[1] * h]
        for p in egoright
    ]

    ego_height = max(egoleft[-1][1], egoright[-1][1]) * EGO_HEIGHT_RATIO

    sps = {
        "LS" : egoleft[0],
        "RS" : egoright[0],
        "LE" : egoleft[-1],
        "RE" : egoright[-1]
    }

    # Tuplize 4 corners
    for i, pt in sps.items():
        sps[i] = imagePointTuplize(pt)

    # Log the ego_height too
    sps["ego_h"] = ego_height

    return sps


def transformBEV(
    img: np.ndarray,
    line: list,
    sps: dict
):
    h, w, _ = img.shape

    # Renorm/tuplize drivable path
    line = [
        (point[0] * w, point[1] * h) for point in line
        if (point[1] * h >= sps["ego_h"])
    ]
    if (not line):
        return (None, None, None, None, None, None, False)

    # Interp more points for original line
    line = interpLine(line, MIN_POINTS)

    # Get transformation matrix
    mat, _ = cv2.findHomography(
        srcPoints = np.array([
            sps["LS"],
            sps["RS"],
            sps["LE"],
            sps["RE"]
        ]),
        dstPoints = np.array([
            BEV_PTS["LS"],
            BEV_PTS["RS"],
            BEV_PTS["LE"],
            BEV_PTS["RE"],
        ])
    )

    # Transform image
    im_dst = cv2.warpPerspective(
        img, mat,
        np.array([BEV_W, BEV_H])
    )

    # Transform egopath
    bev_line = np.array(
        line,
        dtype = np.float32
    ).reshape(-1, 1, 2)
    bev_line = cv2.perspectiveTransform(bev_line, mat)
    bev_line = [
        tuple(map(int, point[0])) 
        for point in bev_line
    ]

    # Polyfit BEV egopath to get 33-coords format with flags
    bev_line, flag_list, validity_list = polyfit_BEV(
        bev_line = bev_line,
        order = POLYFIT_ORDER,
        y_step = BEV_Y_STEP,
        y_limit = BEV_H
    )

    # Now reproject it back to orig space
    inv_mat = np.linalg.inv(mat)
    reproj_line = np.array(
        bev_line,
        dtype = np.float32
    ).reshape(-1, 1, 2)
    reproj_line = cv2.perspectiveTransform(reproj_line, inv_mat)
    reproj_line = [
        tuple(map(int, point[0])) 
        for point in reproj_line
    ]

    return (im_dst, bev_line, reproj_line, flag_list, validity_list, mat, True)


def calTransformedDistance(
    a: ImagePointCoords | PointCoords,
    b: ImagePointCoords | PointCoords,
    homotrans_mat: list[list[float]]
) -> float:

    pts = np.array([a, b], dtype = np.float32).reshape(-1, 1, 2)
    pts_bev = cv2.perspectiveTransform(pts, homotrans_mat)
    a_bev = pts_bev[0][0]
    b_bev = pts_bev[1][0]

    distance_bev = np.linalg.norm(a_bev - b_bev)

    return distance_bev


def calEgoSide(
    bev_egopath: list[ImagePointCoords],
    anchor_offset: float,
    homotrans_mat: list[list[float]]
) -> list[tuple[
        int,    # x-coord, int
        int,    # y-coord, int
        int,    # flag, int
        int     # validity, int
]]:

    # BEV-egoside
    bev_egoside = []
    for point in bev_egopath:
        x, y = point
        bev_egoside.append((
            int(x + anchor_offset), 
            int(y)
        ))

    # Original egoside
    inv_mat = np.linalg.inv(homotrans_mat)
    orig_egoside = np.array(
        bev_egoside,
        dtype = np.float32
    ).reshape(-1, 1, 2)
    orig_egoside = cv2.perspectiveTransform(orig_egoside, inv_mat)
    orig_egoside = [
        tuple(map(int, point[0])) 
        for point in orig_egoside
    ]

    # Flag list
    egoside_flag_list = [0] * len(bev_egoside)
    for i in range(len(bev_egoside)):
        if (not 0 <= bev_egoside[i][0] <= BEV_W):
            egoside_flag_list[i - 1] = 1
            break
    if (not 1 in egoside_flag_list):
        egoside_flag_list[-1] = 1

    # Validity list
    egoside_validity_list = [1] * len(bev_egoside)
    last_valid_index = egoside_flag_list.index(1)
    for i in range(last_valid_index + 1, len(egoside_validity_list)):
        egoside_validity_list[i] = 0

    return (
        bev_egoside, 
        orig_egoside, 
        egoside_flag_list, 
        egoside_validity_list
    )


# ============================== Main run ============================== #


if __name__ == "__main__":

    # DIRECTORY STRUCTURE

    IMG_DIR = "image"
    JSON_PATH = "drivable_path.json"

    BEV_IMG_DIR = "image_bev"
    BEV_VIS_DIR = "visualization_bev"
    BEV_JSON_PATH = "drivable_path_bev.json"
    BEV_SKIPPED_JSON_PATH = "skipped_frames.json"

    # OTHER PARAMS

    W = 1280
    H = 720

    # BEV-related
    MIN_POINTS = 30
    BEV_PTS = {
        "LS" : [240, 1280],         # Left start
        "RS" : [400, 1280],         # Right start
        "LE" : [240, 0],            # Left end
        "RE" : [400, 0]             # Right end
    }
    BEV_W = 640
    BEV_H = 1280
    EGO_HEIGHT_RATIO = 1.05
    BEV_Y_STEP = 128
    POLYFIT_ORDER = 2

    # Visualization (colors in BGR)
    COLOR_EGOPATH = (0, 255, 255)   # Yellow
    COLOR_EGOLEFT = (0, 128, 0)     # Green
    COLOR_EGORIGHT = (255, 255, 0)  # Cyan
    # COLOR_STARTS = (255, 0, 0)      # Blue
    # COLOR_ENDS = (153, 0, 153)      # Kinda purple
    # COLOR_HEIGHT = (0, 165, 255)    # Orange
    # POINT_SIZE = 8
    THICKNESS = -1

    # PARSING ARGS

    parser = argparse.ArgumentParser(
        description = "Generating BEV from TuSimple processed datasets"
    )
    parser.add_argument(
        "--dataset_dir", 
        type = str, 
        help = "Processed TuSimple directory",
        required = True
    )
    # For debugging only
    parser.add_argument(
        "--early_stopping",
        type = int,
        help = "Num. frames you wanna limit, instead of whole set.",
        required = False
    )
    args = parser.parse_args()

    # Parse dataset dir
    dataset_dir = args.dataset_dir
    IMG_DIR = os.path.join(dataset_dir, IMG_DIR)
    JSON_PATH = os.path.join(dataset_dir, JSON_PATH)
    BEV_JSON_PATH = os.path.join(dataset_dir, BEV_JSON_PATH)
    BEV_SKIPPED_JSON_PATH = os.path.join(dataset_dir, BEV_SKIPPED_JSON_PATH)

    # Parse early stopping
    if (args.early_stopping):
        print(f"Early stopping set, stopping after {args.early_stopping} files.")
        early_stopping = args.early_stopping
    else:
        early_stopping = None

    # Generate new dirs and paths
    BEV_IMG_DIR = os.path.join(dataset_dir, BEV_IMG_DIR)
    BEV_VIS_DIR = os.path.join(dataset_dir, BEV_VIS_DIR)

    if not (os.path.exists(BEV_IMG_DIR)):
        os.makedirs(BEV_IMG_DIR)
    if not (os.path.exists(BEV_VIS_DIR)):
        os.makedirs(BEV_VIS_DIR)

    # Preparing data
    with open(JSON_PATH, "r") as f:
        json_data = json.load(f)
    data_master = {}    # Dumped later

    # Get source points for transform
    STANDARD_FRAME = "000022"
    STANDARD_JSON = json_data[STANDARD_FRAME]
    STANDARD_SPS = findSourcePointsBEV(
        h = H,
        w = W,
        egoleft = STANDARD_JSON["egoleft_lane"],
        egoright = STANDARD_JSON["egoright_lane"]
    )
    matrix_registered = False

    # MAIN GENERATION LOOP

    counter = 0
    for frame_id, frame_content in json_data.items():

        counter += 1

        try:
            
            # Acquire frame
            frame_img_path = os.path.join(
                IMG_DIR,
                f"{frame_id}.png"
            )
            img = cv2.imread(frame_img_path)

            # Acquire frame data
            this_frame_data = json_data[frame_id]

            # MAIN ALGORITHM

            # Transform to BEV space            

            # Egopath
            (
                im_dst, 
                bev_egopath, orig_bev_egopath, 
                egopath_flag_list, egopath_validity_list, 
                mat, success
            ) = transformBEV(
                img = img,
                line = this_frame_data["drivable_path"],
                sps = STANDARD_SPS
            )

            # Register standard homography matrix for the first time
            if (not matrix_registered):
                data_master["standard_homomatrix"] = mat.tolist()
                print("Registered standard homography matrix!")
                matrix_registered = True

            anchor_egopath = getLineAnchor(
                renormalizeCoords(
                    this_frame_data["drivable_path"],
                    W, H
                )
            )
            anchor_egoleft = getLineAnchor(
                renormalizeCoords(
                    this_frame_data["egoleft_lane"],
                    W, H
                )
            )
            anchor_egoright = getLineAnchor(
                renormalizeCoords(
                    this_frame_data["egoright_lane"],
                    W, H
                )
            )


            # Egoleft
            (
                bev_egoleft, 
                orig_bev_egoleft, 
                egoleft_flag_list, 
                egoleft_validity_list, 
            ) = calEgoSide(
                bev_egopath = bev_egopath,
                anchor_offset = - calTransformedDistance(
                    [anchor_egoleft[0], H],
                    [anchor_egopath[0], H],
                    mat
                ),
                homotrans_mat = mat
            )

            # Egoright
            (
                bev_egoright, 
                orig_bev_egoright, 
                egoright_flag_list, 
                egoright_validity_list, 
            ) = calEgoSide(
                bev_egopath = bev_egopath,
                anchor_offset = calTransformedDistance(
                    [anchor_egoright[0], H],
                    [anchor_egopath[0], H],
                    mat
                ),
                homotrans_mat = mat
            )
            
            # Skip if invalid frame (due to too high ego_height value)
            if (success == False):
                log_skipped(
                    frame_id,
                    "Null EgoPath from BEV transformation algorithm."
                )
                continue

        except Exception as e:
            print(f"Unexpected error at frame {frame_id}: {e}")
            log_skipped(frame_id, str(e))
            continue

        # Save stuffs
        annotateGT(
            img = im_dst,
            orig_img = img,
            frame_id = frame_id,
            bev_egopath = bev_egopath,
            reproj_egopath = orig_bev_egopath,
            bev_egoleft = bev_egoleft,
            reproj_egoleft = orig_bev_egoleft,
            bev_egoright = bev_egoright,
            reproj_egoright = orig_bev_egoright,
            raw_dir = BEV_IMG_DIR,
            visualization_dir = BEV_VIS_DIR,
            normalized = False
        )

        # Register this frame GT to master JSON
        # Each point has tuple format (x, y, flag, valid)
        data_master[frame_id] = {
            "bev_egopath" : [
                (point[0], point[1], flag, valid)
                for point, flag, valid in list(zip(
                    roundLineFloats(
                        normalizeCoords(
                            bev_egopath,
                            width = BEV_W,
                            height = BEV_H
                        )
                    ), 
                    egopath_flag_list, 
                    egopath_validity_list
                ))
            ],
            "reproj_egopath" : [
                (point[0], point[1], flag, valid)
                for point, flag, valid in list(zip(
                    roundLineFloats(
                        normalizeCoords(
                            orig_bev_egopath,
                            width = W,
                            height = H
                        )
                    ), 
                    egopath_flag_list, 
                    egopath_validity_list
                ))
            ],
            "bev_egoleft" : [
                (point[0], point[1], flag, valid)
                for point, flag, valid in list(zip(
                    roundLineFloats(
                        normalizeCoords(
                            bev_egoleft,
                            width = BEV_W,
                            height = BEV_H
                        )
                    ), 
                    egoleft_flag_list, 
                    egoleft_validity_list
                ))
            ],
            "reproj_egoleft" : [
                (point[0], point[1], flag, valid)
                for point, flag, valid in list(zip(
                    roundLineFloats(
                        normalizeCoords(
                            orig_bev_egoleft,
                            width = W,
                            height = H
                        )
                    ), 
                    egoleft_flag_list, 
                    egoleft_validity_list
                ))
            ],
            "bev_egoright" : [
                (point[0], point[1], flag, valid)
                for point, flag, valid in list(zip(
                    roundLineFloats(
                        normalizeCoords(
                            bev_egoright,
                            width = BEV_W,
                            height = BEV_H
                        )
                    ), 
                    egoright_flag_list, 
                    egoright_validity_list
                ))
            ],
            "reproj_egoright" : [
                (point[0], point[1], flag, valid)
                for point, flag, valid in list(zip(
                    roundLineFloats(
                        normalizeCoords(
                            orig_bev_egoright,
                            width = W,
                            height = H
                        )
                    ), 
                    egoright_flag_list, 
                    egoright_validity_list
                ))
            ],
        }

        # Break if early_stopping reached
        if (early_stopping is not None):
            if (counter >= early_stopping):
                break

    # Save master data
    with open(BEV_JSON_PATH, "w") as f:
        json.dump(data_master, f, indent = 4)

    # Save skipped frames
    with open(BEV_SKIPPED_JSON_PATH, "w") as f:
        json.dump(skipped_dict, f, indent = 4)