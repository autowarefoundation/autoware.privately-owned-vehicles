import os
import tensorflow.compat.v1 as tf
import numpy as np
import argparse

import matplotlib.pyplot as plt

import json

tf.enable_eager_execution()

from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

from utils import *


def main(args):

    # Load precomputed homography matrix
    H = np.load(args.homography_matrix)
    print(H)

    # directory for data and cipo labels.
    dir = args.data_dir
    cipo_dir = args.cipo_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    avg_l2_error = 0
    record_errors = {}

    pose1,pose2 = -1, -1
    objpose1, objpose2 = -1, -1
    id1, id2 = "-1", "-1"


    for record_name in os.listdir(dir):
        print("on record", record_name)
        FILEPATH = os.path.join(dir, record_name)
        dataset = tf.data.TFRecordDataset(FILEPATH, compression_type='')

        cipo_labels_dir = os.path.join(cipo_dir, record_name[28:-9])
        cipo_labels_list = sorted(os.listdir(cipo_labels_dir))

        record_error = 0

        x_init,y_init = 0,0
        
        # Intialize Kalman Filter for ego vehicle using first frame
        for index, data in enumerate(dataset):
            frame = open_dataset.Frame()
            frame.ParseFromString(data.numpy())
            pose = np.array(frame.pose.transform).reshape(4,4)
            x_init, y_init = pose[0,3], pose[1,3]
            break
        kf = VehicleVelocityKalmanFilter(dt=0.1, x=x_init, y=y_init)

        # Intialize Kalman Filter for object vehicle.
        kf_obj = VehicleVelocityKalmanFilter(dt=0.1)
        
        smooth_ego_speed, smooth_obj_speed = -1, -1
        inst_ego_speed_lst, obj_speed_lst = [], []
        smooth_ego_speed_lst, smooth_obj_speed_lst = [], []

        long_error = []

        for index, data in enumerate(dataset):
            frame = open_dataset.Frame()
            frame.ParseFromString(data.numpy())

            with open(os.path.join(cipo_labels_dir, cipo_labels_list[index]), "r") as f:
                cipo_label = json.load(f)

            # print(frame.pose.transform[0])
            # Get pose of the vehicle in world frame
            pose = np.array(frame.pose.transform).reshape(4,4)
            R_vg = (np.linalg.inv(pose))[:3, :3]
            
            # For calcualting ground truth cooridnates and plotting
            (range_images, camera_projections,_, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
            points, cp_points = frame_utils.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose)


            images = sorted(frame.images, key=lambda i:i.name)

            points_all = np.concatenate(points, axis=0)
            # # camera projection corresponding to each point.
            cp_points_all = np.concatenate(cp_points, axis=0)

            id1 = id2

            # Get cipo object id and pixel coordinates
            # id2, uv = get_camera_label(images[0], frame.camera_labels, [3, 3, 1])
            id2, uv = get_camera_label_cipo(cipo_label)


            # If there is no object detected, skip this frame
            if uv == None:
                continue

            # Find xy cords from uv using precomputed homography matrix
            clicked_uv = np.array([uv], dtype=np.float32)

            xy_coords = project_image_to_world(clicked_uv, H)

            # Find corresponding (x, y) ground truth on the road
            xy, matched_uv, pixel_error = find_closest_xy_from_uv(
                clicked_uv[0], cp_points_all, points_all, camera_id=1)
            
            # Compute error
            error = np.linalg.norm(xy[0]-xy_coords[0][0])
            record_error += error
            long_error.append(error)

            pose1 = pose2
            pose2 = pose

            objpose1 = objpose2
            objpose2 = xy_coords[0]

            if (id1 == id2):
                inst_ego_vel = compute_speed_inst(pose1, pose2, R_vg)
                obj_vel = compute_obj_speed_inst(inst_ego_vel, objpose1, objpose2)

                # Ego velocity estimation (Might be a better idea to get this from vechicle CAN data if available)
                kf.update(np.array([pose[0, 3], pose[1, 3]]))
                smooth_ego_vel = R_vg[:2,:2] @ kf.get_velocity(np.array([0,0]))
                smooth_ego_speed = np.linalg.norm(smooth_ego_vel) * 3.6
                print("Kalman ego velocity: ", smooth_ego_vel)
                print("Kalman ego speed: ", smooth_ego_speed)

                # obstacle velocity estiamtion
                kf_obj.update(np.array([xy_coords[0][0], xy_coords[0][1]]))
                smooth_obj_vel = kf_obj.get_velocity(smooth_ego_vel)
                smooth_obj_speed = np.linalg.norm(smooth_obj_vel) * 3.6
                print("Kalman OBJ velocity: ", smooth_obj_vel)
                print("Kalman OBJ speed: ", smooth_obj_speed)

                d = compute_safe_dist(smooth_ego_speed/3.6, smooth_obj_speed/3.6, 1,1,1,1)

                if np.linalg.norm(xy_coords[0]) < d:
                    print("CAUTION: UNSAFE DISTANCE")
                        # send the caution signal to the vehicle here
                        
                inst_ego_speed_lst.append(inst_ego_speed)
                obj_speed_lst.append(obj_speed)
                smooth_ego_speed_lst.append(smooth_ego_speed)
                smooth_obj_speed_lst.append(smooth_obj_speed)


            else:
                print("Chnaged tracked object. Resetting object kalman filter")
                inst_ego_vel = -1
                obj_vel = -1
                kf_obj = VehicleVelocityKalmanFilter(dt=0.1, x=xy_coords[0][0], y=xy_coords[0][1])
            

            # woudl it be beter to transform it into glob coordinates and calculate velocity independently (from vel of ego vehicle)
            inst_ego_speed = np.linalg.norm(inst_ego_vel) * 3.6
            obj_speed = np.linalg.norm(obj_vel) * 3.6

            print("Ego Speed: ", inst_ego_speed)
            print("Obj Speed: ", obj_speed)
            print("Longitudnal error (L2):", error)
            print()

            if args.visualize:
                plot_points_on_image_with_annotations(xy, xy_coords[0], smooth_ego_speed, smooth_obj_speed, clicked_uv[0], matched_uv, images[0], index, record_name, output_dir)

        record_error/= index
        record_errors[record_name] = record_error
        plt.figure()
        plt.plot(long_error)
        plt.xlabel("Frame")
        plt.ylabel("L2 error (m)")
        plt.title(f"Error over frames for {record_name}")
        plt.savefig(f"{output_dir}/error_{record_name}.png")

        np.savez(f"{output_dir}/error_{record_name}.npz",
            long_error=long_error,
            xlabel="Frame",
            ylabel="L2 error (m)",
            title=f"Error over frames for {record_name}")
        
        # Make a plot with two subfigures. one plotting inst_ego_speed_lst and smooth_ego_speed_lst, other plotting obj_speed_lst and smooth_obj_speed_lst
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(inst_ego_speed_lst, label='Instantaneous Ego Speed', alpha=0.5)
        plt.plot(smooth_ego_speed_lst, label='Smoothed Ego Speed', alpha=0.8)
        plt.xlabel('Frame')
        plt.ylabel('Speed (km/h)')
        plt.title('Ego Vehicle Speed')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(obj_speed_lst, label='Instantaneous Object Speed', alpha=0.5)
        plt.plot(smooth_obj_speed_lst, label='Smoothed Object Speed', alpha=0.8)
        plt.xlabel('Frame')
        plt.ylabel('Speed (km/h)')
        plt.title('Object Vehicle Speed')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{output_dir}/speed_{record_name}.png")
        plt.close()

    avg_l2_error = sum(record_errors.values()) / len(record_errors)

    print(avg_l2_error)
    print(record_errors)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute distance and velocity.")
    parser.add_argument("homography_matrix", type=str, help="Path to the homography matrix file (e.g., 'output/vehicle_to_pixel_la.npy')")
    parser.add_argument("data_dir", type=str, help="Directory containing the input dataset")
    parser.add_argument("cipo_dir", type=str, help="Directory containing the CIPO labels")
    parser.add_argument("output_dir", type=str, help="Directory to save the results")
    parser.add_argument("--visualize", action="store_true", help="Whether to visualize and save images with annotations")

    args = parser.parse_args()

    main(args)