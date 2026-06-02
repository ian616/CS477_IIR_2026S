import rclpy
from rclpy.node import Node
class TestNode(Node):
    def __init__(self):
        super().__init__('my_node')
        self.declare_parameter("model_path", "default.pt")
        print("model_path:", self.get_parameter("model_path").value)
rclpy.init(args=["--ros-args", "-p", "model_path:=best.pt"])
n = TestNode()
rclpy.shutdown()
