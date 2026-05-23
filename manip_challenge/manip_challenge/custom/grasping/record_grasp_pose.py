#!/usr/bin/env python3

import sys
import os
import json
import rclpy
from rclpy.node import Node
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException

import numpy as np
if not hasattr(np, 'float'):
    np.float = float
import tf_transformations

class GraspRecorder(Node):
    def __init__(self, object_name, save_key):
        super().__init__('grasp_recorder_node')
        
        # We also need sim time to correctly evaluate Gazebo TFs
        from rclpy.parameter import Parameter
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        
        self.object_name = object_name
        self.save_key = save_key
        self.target_frame = f'workpiece_{object_name}'
        self.source_frame = 'tool_center_point'
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Database file path
        # Save it inside the config directory of the workspace for persistence
        self.db_path = os.path.join(
            os.path.expanduser('~'), 
            'ros_ws/cs477_ws/src/CS477_IIR_2026S/manip_challenge/config/grasp_database.json'
        )
        
        self.timer = self.create_timer(0.5, self.record_transform)
        self.recorded = False

    def record_transform(self):
        if self.recorded:
            return
            
        try:
            # We want the transform that maps points in 'tool_center_point' (source)
            # into the 'workpiece_X' (target) coordinate frame.
            # This is mathematically equivalent to the pose of the tool in the workpiece frame.
            t = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.source_frame,
                rclpy.time.Time()
            )
            
            x = t.transform.translation.x
            y = t.transform.translation.y
            z = t.transform.translation.z
            
            qx = t.transform.rotation.x
            qy = t.transform.rotation.y
            qz = t.transform.rotation.z
            qw = t.transform.rotation.w
            
            roll, pitch, yaw = tf_transformations.euler_from_quaternion([qx, qy, qz, qw])
            
            self.get_logger().info(f"Successfully captured transform for TF {self.target_frame}!")
            self.get_logger().info(f"Translation: X={x:.4f}, Y={y:.4f}, Z={z:.4f}")
            self.get_logger().info(f"Rotation: Roll={roll:.4f}, Pitch={pitch:.4f}, Yaw={yaw:.4f}")
            
            self.save_to_database(x, y, z, roll, pitch, yaw)
            
            self.recorded = True
            
        except TransformException as ex:
            self.get_logger().info(f"Waiting for TF between {self.target_frame} and {self.source_frame}...")

    def save_to_database(self, x, y, z, r, p, yaw):
        db_data = {}
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    db_data = json.load(f)
            except json.JSONDecodeError:
                self.get_logger().error("Database file is corrupted. Creating a new one.")
                
        # Update or add the new object grasp pose using the specific save key
        db_data[self.save_key] = {
            'x': x,
            'y': y,
            'z': z,
            'roll': r,
            'pitch': p,
            'yaw': yaw,
            'grasp_value': 0
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with open(self.db_path, 'w') as f:
            json.dump(db_data, f, indent=4)
            
        self.get_logger().info(f"Saved grasp pose for TF '{self.object_name}' under dictionary key '{self.save_key}' to {self.db_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: ros2 run manip_challenge record_grasp_pose <gazebo_object_name> [optional_database_key]")
        print("Example 1: ros2 run manip_challenge record_grasp_pose coke_can")
        print("Example 2: ros2 run manip_challenge record_grasp_pose meat_can meat_can_standing")
        sys.exit(1)
        
    object_name = sys.argv[1]
    
    # Use the second argument as the dictionary key if provided, else default to the object name
    save_key = sys.argv[2] if len(sys.argv) > 2 else object_name
    
    rclpy.init()
    node = GraspRecorder(object_name, save_key)
    
    try:
        # Spin until the transform is successfully recorded
        while rclpy.ok() and not node.recorded:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
