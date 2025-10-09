import rclpy
from rclpy.node import Node

from carla_msgs.msg import CarlaEgoVehicleControl
from std_msgs.msg import Float32

class CarlaControlPublisher(Node):
    def __init__(self):
        super().__init__('carla_control_publisher')
        self.steering_sub_ = self.create_subscription(Float32, '/vehicle/steering_cmd', self.steering_callback, 1)
        self.throttle_sub_ = self.create_subscription(Float32, '/vehicle/throttle_cmd', self.throttle_callback, 1)
        self.control_pub_ = self.create_publisher(CarlaEgoVehicleControl, '/carla/hero/vehicle_control_cmd', 1)
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        msg = CarlaEgoVehicleControl()
        msg.throttle = 0.5
        msg.steer = 0.0
        msg.brake = 0.0
        msg.hand_brake = False
        msg.reverse = False
        msg.manual_gear_shift = False
        msg.gear = 0
        self.control_pub_.publish(msg)
    
    def steering_callback(self, msg):
        self.get_logger().info(f'Steering command received: {msg.data}')
        
    def throttle_callback(self, msg):
        self.get_logger().info(f'Throttle command received: {msg.data}')

def main(args=None):
    rclpy.init(args=args)
    node = CarlaControlPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
