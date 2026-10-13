"""Analytic ground homography (full-frame pixel -> ground, metres from the front bumper).

Output convention matches VisionPilot's config/H.yaml: x forward, y left.
The camera pose is read from the rig JSON, which uses the ROS frame (x fwd, y left, z up).
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

DEFAULT_RIG = Path(__file__).resolve().parents[2] / "config" / "carla916.json"


def load_camera(rig_path, sensor_id):
    rig = json.loads(Path(rig_path).read_text())
    for sensor in rig.get("sensors", []):
        if sensor.get("id") == sensor_id:
            return sensor["spawn_point"], sensor["attributes"]
    raise SystemExit(f"sensor '{sensor_id}' not found in {rig_path}")


def rotation(roll_deg, pitch_deg, yaw_deg):
    r, p, y = np.radians([roll_deg, pitch_deg, yaw_deg])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    return rz @ ry @ rx


def ground_homography(pose, width, height, fov_deg, bumper_x):
    f = (width / 2) / np.tan(np.radians(fov_deg / 2))
    cx, cy = width / 2, height / 2
    rot = rotation(pose["roll"], pose["pitch"], pose["yaw"])

    uv, xy = [], []
    for u in np.linspace(100, width - 100, 12):
        for v in np.linspace(cy + 60, height - 50, 12):
            ray = rot @ np.array([1.0, -(u - cx) / f, -(v - cy) / f])
            if ray[2] >= 0:
                continue
            t = -pose["z"] / ray[2]
            uv.append([u, v])
            xy.append([pose["x"] + t * ray[0] - bumper_x, pose["y"] + t * ray[1]])

    h, _ = cv2.findHomography(np.array(uv, np.float32), np.array(xy, np.float32), 0)
    return h


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rig", type=Path, default=DEFAULT_RIG, help="rig JSON (default: %(default)s)"
    )
    parser.add_argument(
        "--sensor-id", default="main_cam", help="camera id in the rig (default: %(default)s)"
    )
    parser.add_argument(
        "--bumper-x",
        type=float,
        default=2.45,
        help="front bumper x from the vehicle origin, m (default: %(default)s, lincoln mkz_2020)",
    )
    parser.add_argument("--output", type=Path, required=True, help="H.yaml to write")
    args = parser.parse_args()

    pose, attrs = load_camera(args.rig, args.sensor_id)
    h = ground_homography(
        pose,
        int(attrs["image_size_x"]),
        int(attrs["image_size_y"]),
        float(attrs["fov"]),
        args.bumper_x,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fs = cv2.FileStorage(str(args.output), cv2.FILE_STORAGE_WRITE)
    fs.write("H", h)
    fs.release()
    print(f"wrote {args.output}")
    cx = int(attrs["image_size_x"]) // 2
    for u, v in [(cx, 700), (cx, 900), (cx - 460, 900), (cx + 440, 900)]:
        q = h @ np.array([u, v, 1.0])
        print(f"  pixel ({u}, {v}) -> ground {np.round(q[:2] / q[2], 2)}")


if __name__ == "__main__":
    main()
