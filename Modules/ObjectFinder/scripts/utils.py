import tensorflow.compat.v1 as tf
import numpy as np

import matplotlib.pyplot as plt

tf.enable_eager_execution()

from filterpy.kalman import KalmanFilter

##### HELPER FUNCTIONS ######

def get_camera_label_cipo(cipo_labels):
    objects = cipo_labels["result"]
    if len(objects) < 1:
       return None, None
   
    try:
        objects = sorted(objects, key = lambda x: x["id"])
        cipo = objects[0]
        return cipo["trackid"], [int(cipo["x"]+ (cipo["width"]/2)), int(cipo["y"] + cipo["height"])]
    except Exception as e:
        print(f"Error in get_camera_label_cipo: {e}")
        return None, None

def plot_image(camera_image):
  """Plot a cmaera image."""
  plt.figure(figsize=(20, 12))
  plt.imshow(tf.image.decode_jpeg(camera_image.image))
  plt.grid("on")

def project_image_to_world(uv_points, H_inv):
    """
    uv_points: Nx2 array of pixel coordinates (u, v)
    H_inv: 3x3 inverse homography matrix
    Returns: Nx2 array of real-world (x, y) coordinates on the road plane
    """
    uv_hom = np.hstack([uv_points, np.ones((uv_points.shape[0], 1))])  # Nx3
    xy_hom = uv_hom @ H_inv.T  # Nx3
    xy = xy_hom[:, :2] / xy_hom[:, 2:]  # normalize by third coordinate
    return xy

def find_closest_xy_from_uv(uv_query, cp_points_on_road, points_on_road, camera_id=0):
    """
    uv_query: (2,) pixel coordinate you clicked, e.g., (u, v)
    cp_points_on_road: (N, 6) array, first 3 cols are (camera_id, u, v)
    points_on_road: (N, 3) array of corresponding 3D points
    camera_id: which camera to match against (default = 0)

    Returns: (x, y), and index of the closest match
    """
    # Filter to only points from the specified camera
    mask = cp_points_on_road[:, 0] == camera_id
    cp_uv = cp_points_on_road[mask][:, 1:3]  # (M, 2)
    points_xy = points_on_road[mask][:, :2]  # (M, 2)

    # Compute L2 distance in pixel space
    dists = np.linalg.norm(cp_uv - uv_query, axis=1)
    min_idx = np.argmin(dists)

    if dists[min_idx] > 50:
       return ((-1, -1), cp_uv[min_idx], dists[min_idx])

    # Return the corresponding (x, y) and the index (within the filtered set)
    return points_xy[min_idx], cp_uv[min_idx], dists[min_idx]

def plot_points_on_image_with_annotations(real_points, predicted_points, ego_speed, obj_speed, camera_projection, closest_matched_pixel, camera_image, iter=0, filename="", output_dir="."):
    """
    Plots points on a camera image and annotates real and projected coordinates.

    Args:
        projected_points: [N, 3] numpy array. Each row is [camera_x, camera_y, range].
        real_points: [N, 2] or [N, 3] numpy array. Each row is [real_x, real_y] or [real_x, real_y, _].
        camera_image: jpeg encoded camera image (bytes).
        rgba_func: function that generates an RGBA color from a range value.
        point_size: float. Size of plotted points.
    """
    # Decode and show the image
    plot_image(camera_image)


        # Get the corresponding real point
    if real_points is not None:
        real_x, real_y = real_points[0], real_points[1]
        predicted_x, predicted_y = predicted_points[0], predicted_points[1]
        # Annotate with both real and projected coordinates
        annotation = f"P:({round(predicted_x,3)},{round(predicted_y, 3)})\nR:({round(real_x,3)},{round(real_y,3)} \n Ego Speed: {int(ego_speed)} km/h \nSpeed: {int(obj_speed)} km/h)"
        plt.text(camera_projection[0] + 5, camera_projection[1] + 5, annotation, fontsize=8, color='white',
                  bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))

        plt.scatter([closest_matched_pixel[0]], [closest_matched_pixel[1]], color = "red", s=40)
        plt.scatter([camera_projection[0]], [camera_projection[1]], color = "green", s=40)

    plt.savefig(f"{output_dir}/{filename[:-9]}_{iter}.png")
    plt.close()
    # plt.show()

# INST SPEED ESTIMATION

def compute_speed_inst(pose1, pose2, R_vg):
    # Extract x, y, z positions
    if type(pose1) == int or type(pose2) == int:
      return -1
      
    p1 = np.array([pose1[0, 3], pose1[1,3], pose1[2, 3]])
    p2 = np.array([pose2[0,3], pose2[1,3], pose2[2,3]])

    # Time difference in seconds
    dt = 0.1
    ego_vel_v = R_vg @ (p2-p1)/dt

    return ego_vel_v



def compute_obj_speed_inst(ego_vel, pose1, pose2):
    if type(pose1) == int or type(pose2)== int:
      return -1
    

    p1 = np.array([pose1[0], pose1[1]])
    p2 = np.array([pose2[0], pose2[1]])

    # Time difference in seconds

    dt = 0.1

    obj_vel_v = (p2 - p1 )/dt
    # Euclidean distance
    # distance = np.linalg.norm(p2 - p1)

    # Speed = distance / time
    return ego_vel[:2] + obj_vel_v  # in meters per second


def compute_safe_dist(v_r, v_f, rho, a_max_accel, a_min_brake, a_max_brake):
  """
    Calculates the minimum safe longitudinal distance between two vehicles
    based on the RSS (Responsibility-Sensitive Safety) model.

    Parameters:
    - v_r (float): Speed of rear (following) vehicle [m/s]
    - v_f (float): Speed of front (leading) vehicle [m/s]
    - rho (float): Response time of rear vehicle [s]
    - a_max_accel (float): Max acceleration of rear vehicle [m/s^2]
    - a_min_brake (float): Min (guaranteed) braking of rear vehicle [m/s^2]
    - a_max_brake (float): Max braking of front vehicle [m/s^2]

    Returns:
    - d_min (float): Minimum safe distance [meters]
    """
  return v_r*rho + 0.5*(a_max_accel* rho * rho) + (v_r + rho * a_max_accel)*(v_r + rho * a_max_accel) / (2*a_min_brake) - (v_f*v_f)/(2*a_max_brake)

#### KALMAN FILTER


class VehicleVelocityKalmanFilter:
    def __init__(self, dt,x=0,y=0, process_noise_std=1.0, measurement_noise_std=0.5):
        self.dt = dt

        # Initialize Kalman Filter
        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        # State vector: [x_rel, y_rel, vx_rel, vy_rel]
        # self.kf.x = np.zeros(4)
        self.kf.x = np.array([[x], [y], [0], [0]])

        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Measurement function: we only observe relative position
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Initial state covariance
        self.kf.P *= 1000.0

        # Process noise covariance
        q = process_noise_std

        self.kf.Q = q * np.array([
        [dt**4/4,     0, dt**3/2,     0],
        [0,     dt**4/4,     0, dt**3/2],
        [dt**3/2,     0, dt**2,     0],
        [0,     dt**3/2,     0, dt**2]
        ])

        # Measurement noise covariance
        r = measurement_noise_std
        self.kf.R = np.array([
            [r, 0],
            [0, r]
        ])

    def update(self, rel_position_meas):
        """
        rel_position_meas: np.array([x_rel, y_rel]) - measured relative position (ego frame)
        ego_velocity: np.array([vx_ego, vy_ego]) - ego velocity in ego frame

        Returns:
            front_velocity_estimate: np.array([vx_front, vy_front])
        """
        # Predict
        self.kf.predict()

        # Update with measured relative position
        self.kf.update(rel_position_meas)
        return True
        
    def get_velocity(self, ego_velocity):
        # Extract relative velocity estimate from state
        vx_rel = self.kf.x[2]
        vy_rel = self.kf.x[3]

        # Compute front vehicle velocity in ego frame
        vx_front = ego_velocity[0] + vx_rel
        vy_front = ego_velocity[1] + vy_rel

        return np.array([vx_front, vy_front])

#############################################################
