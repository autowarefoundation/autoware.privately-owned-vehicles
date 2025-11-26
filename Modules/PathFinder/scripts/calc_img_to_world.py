import cv2
import json
# import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
import yaml
import glob
import os

# Format floats with 4 decimal places
def float_representer(dumper, value):
    return dumper.represent_scalar('tag:yaml.org,2002:float', f"{value:.4f}")

# Force [x, y] to appear in flow style
def point_representer(dumper, data):
    if isinstance(data, list) and len(data) == 2 and all(isinstance(i, float) for i in data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
    return dumper.represent_list(data)

# Register custom representers
# yaml.add_representer(float, float_representer)
# yaml.add_representer(list, point_representer)

# def read_predicted_file(json_path):
#     with open(json_path) as f:
#         lane_info = json.load(f)
#     lanes_pd = lane_info['lanes']
#     lanes = list()
#     for lane in lanes_pd:
#         lanes.append(lane['points'])
#     return lanes

def read_img_lane_from_json(json_path):
    """
    Input: json_path of 3D lane labels
    Output: lanes_2d, lanes_3d
    """
    with open(json_path) as f:
        lane_info = json.load(f)
    lanes_3d = lane_info['lanes']
    # lanes_3d = interploate_3d_tool(lanes_3d)
    camera_intri = lane_info['calibration']
    camera_intri_transpose = np.array(camera_intri).T.tolist()
    lanes_2d = []
    for lane_spec_3d in lanes_3d:
        lane_spec_3d = np.array(lane_spec_3d, dtype=np.float32)
        points_xyz = lane_spec_3d[:, :3]
        pcl_camera_homo = np.hstack([points_xyz, np.ones(points_xyz.shape[0], dtype=np.float32).reshape((-1, 1))])
        pcl_img = np.dot(pcl_camera_homo, camera_intri_transpose)
        pcl_img = pcl_img / pcl_img[:, [2]]
        lane_2d_spec = pcl_img[:, :2]
        lanes_2d.append(lane_2d_spec.tolist())
    return lanes_2d, lanes_3d, camera_intri


def draw_lanes2d_on_img_points(lanes_2d, img_path, save_path):
    image = cv2.imread(img_path)
    for lane_spec in lanes_2d:
        for point in lane_spec:
            try:
                image = cv2.circle(image, (int(point[0]), int(point[1])), radius=3, color=(0, 0, 255), thickness=-1)
            except:
                print('project error:', int(point[0]), int(point[1]))
    cv2.imwrite(save_path, image)


def draw_lanes2d_on_img_line(lanes_2d, img_path, save_path):
    image = cv2.imread(img_path)
    for lane_spec in lanes_2d:
        lane_spec = np.array(lane_spec).astype(np.int32)
        lane_spec = lane_spec.reshape((-1, 1, 2))
        cv2.polylines(image, [lane_spec], False, color=(0, 0, 255), thickness=2)
    cv2.imwrite(save_path, image)

def write_to_yaml(lanes2d, lanes3d, camera_intri, yaml_path):
    """
    Function to write the 2D lane points in Image Pixel to a YAML file.
    """
    print(yaml_path)
    with open(yaml_path, 'w') as f:
        yaml.dump({"lanes2d": lanes2d, "lanes3d": lanes3d, "camera_intri": camera_intri}, f, default_flow_style=False)

# if __name__ == '__main__':
#     run = "000004"
#     id = "1616007252900"
#     json_path = f"../ONCE_3DLanes/train/{run}/cam01/{id}.json"
#     img_path = f"../ONCE_3DLanes/data/{run}/cam01/{id}.jpg"
#     lanes_2d, lanes_3d, camera_intri = read_img_lane_from_json(json_path=json_path)
#     print(lanes_2d, lanes_3d, camera_intri)
#     write_to_yaml(lanes_2d, lanes_3d, camera_intri, yaml_path="../test/test.yaml")
#     # draw_lanes2d_on_img_line(lanes_2d=lanes_2d, img_path=img_path, save_path=f"/home/je/{id}.jpg")
#     draw_lanes2d_on_img_points(lanes_2d=lanes_2d, img_path=img_path, save_path=f"/home/je/{id}.jpg")

if __name__ == '__main__':
    run = "000001"
    json_dir = f"../ONCE_3DLanes/train/{run}/cam01/"
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    start_id = 1616005402200
    end_id = 1616005420200

    for json_path in json_files:
        id = os.path.splitext(os.path.basename(json_path))[0]
        if start_id <= int(id) <= end_id:
            img_path = f"../ONCE_3DLanes/data/{run}/cam01/{id}.jpg"
            print(img_path)
            lanes_2d, lanes_3d, camera_intri = read_img_lane_from_json(json_path=json_path)
            # print(lanes_2d, lanes_3d, camera_intri)
            write_to_yaml(lanes_2d, lanes_3d, camera_intri, yaml_path=f"../test/{run}/{id}.yaml")
            draw_lanes2d_on_img_points(lanes_2d=lanes_2d, img_path=img_path, save_path=f"../test/{run}/img/{id}.jpg")