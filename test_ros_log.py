import rclpy
from rclpy.node import Node
import time
rclpy.init()
n = Node('test_node')
n.get_logger().info("This is an info message")
n.get_logger().warn("This is a warning message")
n.get_logger().error("This is an error message")
rclpy.shutdown()
