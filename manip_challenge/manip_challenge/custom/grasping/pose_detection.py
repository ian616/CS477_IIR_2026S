#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import time
import json

# Import the custom services provided in your workspace
from riro_srvs.srv import StringPose
from geometry_msgs.msg import Pose
from std_msgs.msg import String

class ObjectLocatorClient(Node):
    def __init__(self):
        super().__init__('object_locator_client')
        # Create the client targeting Gazebo's internal backend
        self.cli = self.create_client(StringPose, '/get_object_pose')
        
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /get_object_pose service to become available...')

    def get_target_pose(self, object_name: str) -> Pose:
        """Sends a request to get the exact 3D pose of an item by name."""
        req = StringPose.Request()
        req.data = object_name
        
        self.get_logger().info(f"Requesting coordinates for item: '{object_name}'")
        future = self.cli.call_async(req)
        
        # Spin until the service returns the data
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if future.done():
                try:
                    response = future.result()
                    self.get_logger().info(f"Successfully retrieved pose for {object_name}")
                    return response.pose
                except Exception as e:
                    self.get_logger().error(f"Service call failed: {e}")
                    return None
            time.sleep(0.1)

    def get_all_pickable_objects(self) -> list:
        """Subscribes to /world_model and returns a list of all pickable object names."""
        object_names = []
        msg_received = False

        def world_model_callback(msg):
            nonlocal object_names, msg_received
            try:
                data = json.loads(msg.data)
                
                # List of static scene objects to ignore
                static_objects = {
                    'ur5', 'ur5_base', 'cafe_table_left', 'cafe_table_right', 
                    'cafe_table', 'storage_left', 'storage_right', 
                    'workspace_basket', 'bookshelf', 'sun', 'ground_plane'
                }
                
                for obj in data.get('world', []):
                    name = obj.get('name', '')
                    if name and name not in static_objects:
                        object_names.append(name)
                
                msg_received = True
            except Exception as e:
                self.get_logger().error(f"Failed to parse world_model message: {e}")

        sub = self.create_subscription(String, '/world_model', world_model_callback, 10)
        self.get_logger().info('Waiting for /world_model to get the list of objects...')
        
        while rclpy.ok() and not msg_received:
            rclpy.spin_once(self, timeout_sec=0.1)
            
        self.destroy_subscription(sub)
        return object_names

def main(args=None):
    rclpy.init(args=args)
    locator = ObjectLocatorClient()
    
    # Get all available pickable objects
    target_items = locator.get_all_pickable_objects()
    
    if not target_items:
        print("\n--- No pickable objects found in the world model ---")
    else:
        print(f"\n--- Found {len(target_items)} pickable objects ---")
        for item in target_items:
            pose = locator.get_target_pose(item)
            if pose:
                print(f"\n--- Object Location Found: {item} ---")
                print(f"Position -> X: {pose.position.x:.3f}, Y: {pose.position.y:.3f}, Z: {pose.position.z:.3f}")
                print(f"Rotation -> X: {pose.orientation.x:.3f}, Y: {pose.orientation.y:.3f}, Z: {pose.orientation.z:.3f}, W: {pose.orientation.w:.3f}\n")
    
    locator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()