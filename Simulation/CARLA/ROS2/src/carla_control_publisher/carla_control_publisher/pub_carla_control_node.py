import rclpy
from rclpy.node import Node

import numpy as np
import carla
from carla_msgs.msg import CarlaEgoVehicleControl
from std_msgs.msg import Float32


class CarlaControlPublisher(Node):
    def __init__(self):
        super().__init__('carla_control_publisher')
        self.steering_sub_ = self.create_subscription(Float32, '/vehicle/steering_cmd', self.steering_callback, 1)
        self.throttle_sub_ = self.create_subscription(Float32, '/vehicle/throttle_cmd', self.throttle_callback, 1)
        # self.control_pub_ = self.create_publisher(CarlaEgoVehicleControl, '/carla/hero/vehicle_control_cmd', 1)
        # -------------------------------------------------------------------------------------------------------
        vehicle_control_cmd_topic = "/carla/hero/vehicle_control_cmd"
        # Connect to CARLA to identify which actor is 'hero'
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)
        world = client.get_world()
        hero_actor = None
        for actor in world.get_actors().filter('vehicle.*'):
            if actor.attributes.get('role_name') == 'hero':
                hero_actor = actor
                actor_id = hero_actor.id
                vehicle_control_cmd_topic = f"/carla/actor{actor_id}/vehicle_control_cmd"
                break
        # -------------------------------------------------------------------------------------------------------
        self.control_pub_ = self.create_publisher(CarlaEgoVehicleControl, vehicle_control_cmd_topic, 1)
        self.timer = self.create_timer(0.01, self.timer_callback)
        self.steering_angle_cmd = 0.0
        self.throttle_cmd = 0.0

    def timer_callback(self):
        msg = CarlaEgoVehicleControl()
        msg.throttle = self.throttle_cmd if self.throttle_cmd > 0 else 0.0
        msg.steer = np.clip((180.0 / np.pi) * (self.steering_angle_cmd / 70.0), -1.0, 1.0)  # vehicle specific
        msg.brake = - self.throttle_cmd if self.throttle_cmd < 0 else 0.0
        msg.hand_brake = False
        msg.reverse = False
        msg.manual_gear_shift = False
        # msg.gear = 0
        msg.gear = 1
        self.control_pub_.publish(msg)

    def steering_callback(self, msg):
        self.get_logger().info(f'Steering command received: {msg.data}')
        self.steering_angle_cmd = msg.data

    def throttle_callback(self, msg):
        self.get_logger().info(f'Throttle command received: {msg.data}')
        self.throttle_cmd = msg.data


def main(args=None):
    rclpy.init(args=args)
    node = CarlaControlPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
