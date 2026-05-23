#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import json
import math
import builtin_interfaces.msg

from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import numpy as np
# Fix for tf_transformations failing on newer versions of numpy
if not hasattr(np, 'float'):
    np.float = float
import tf_transformations

# Bounding box dimensions (X, Y, Z in meters) extracted from SDF/DAE files
DIMENSIONS = {
    'coke_can': (0.0670, 0.0670, 0.1239),
    'strawberry': (0.0452, 0.0453, 0.0457),
    'meat_can': (0.1021, 0.0601, 0.0835),
    'hammer': (0.1335, 0.3348, 0.0508),
    'banana': (0.0819, 0.1432, 0.0685),
    'eraser': (0.1350, 0.0600, 0.0500),
    'mustard_bottle': (0.0972, 0.0666, 0.1913),
    'glue': (0.0540, 0.0320, 0.1330),
    'snacks': (0.1650, 0.0600, 0.2350),
    'book': (0.1300, 0.0300, 0.2060),
    'soap': (0.1400, 0.0650, 0.1000),
    'soap2': (0.0650, 0.0400, 0.1050),
    'biscuits': (0.1900, 0.0600, 0.1500)
}

class RVizVisualizer(Node):
    def __init__(self):
        super().__init__('rviz_visualizer_node')
        
        # VERY IMPORTANT: Tell this node to use the Gazebo simulated clock!
        # Without this, it uses the physical computer's clock (e.g. year 2026), 
        # while RViz TF is using the simulation clock (e.g. 120 seconds).
        # This causes the "Extrapolation needed into the future" error!
        from rclpy.parameter import Parameter
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        
        # Keep track of active marker IDs to delete them smoothly without flickering
        self.active_marker_ids = set()
        
        # Subscribe to world model JSON
        self.create_subscription(String, '/world_model', self.world_model_callback, 10)
        
        # Publisher for markers
        self.marker_pub = self.create_publisher(MarkerArray, '/gazebo_objects_markers', 10)
        
        # TF Broadcaster to show workpiece origins
        self.tf_broadcaster = TransformBroadcaster(self)
        
        self.cleared_old_markers = False
        
        self.get_logger().info("RViz Visualizer running. Publishing markers and Workpiece TFs.")

    def world_model_callback(self, msg):
        try:
            data = json.loads(msg.data)
            world_objects = data.get('world', [])
        except Exception as e:
            self.get_logger().error(f"Failed to parse /world_model JSON: {e}")
            return
            
        marker_array = MarkerArray()
        
        # 1. Clear any ghost markers leftover from previous script runs
        if not self.cleared_old_markers:
            del_all = Marker()
            del_all.action = 3 # Marker.DELETEALL
            marker_array.markers.append(del_all)
            self.cleared_old_markers = True
            
        current_marker_ids = set()
        
        for idx, obj in enumerate(world_objects):
            name = obj.get('name', '')
            pose = obj.get('pose', [0, 0, 0, 0, 0, 0])
            
            if not name or len(pose) < 6:
                continue
                
            # Parse pose: [x, y, z, roll, pitch, yaw]
            x, y, z, roll, pitch, yaw = pose[:6]
            
            # Identify base model name (e.g., coke_can_1234_0 -> coke_can)
            base_model_name = name
            found = False
            for prefix in DIMENSIONS.keys():
                if name.startswith(prefix):
                    base_model_name = prefix
                    found = True
                    break
            
            # 2. Ignore any object that isn't a known pickable item (e.g., robot links, tables, ground plane)
            if not found:
                continue
                
            dim = DIMENSIONS[base_model_name]
            
            # Publish a CUBE marker for the collision bounding box
            marker = Marker()
            
            # Use 'base_link' as the frame_id since Gazebo coordinates perfectly align with the robot base
            marker.header.frame_id = 'base_link' 
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "gazebo_collisions"
            
            # Lock the marker to the frame so RViz constantly updates it with the latest TF,
            # which perfectly bypasses any timestamp synchronization flickering!
            marker.frame_locked = True
            
            # Use a stable ID based on the object's unique name to prevent ID swapping
            stable_id = hash(name) % 2147483647
            marker.id = stable_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            current_marker_ids.add(stable_id)
            
            if not hasattr(self, 'last_poses'):
                self.last_poses = {}
                
            # Deadband filter: Prevent visual physics jitter by freezing the coordinate if it only vibrated slightly.
            # We must still publish the marker every frame to keep RViz happy, we just publish the *same* frozen coordinate.
            last_pose = self.last_poses.get(stable_id)
            if last_pose is not None:
                dx, dy, dz, droll, dpitch, dyaw = [abs(a - b) for a, b in zip((x, y, z, roll, pitch, yaw), last_pose)]
                if dx < 0.005 and dy < 0.005 and dz < 0.005 and droll < 0.05 and dpitch < 0.05 and dyaw < 0.05:
                    x, y, z, roll, pitch, yaw = last_pose # Use the cached frozen coordinates
                else:
                    self.last_poses[stable_id] = (x, y, z, roll, pitch, yaw) # Update the cache because it actually moved
            else:
                self.last_poses[stable_id] = (x, y, z, roll, pitch, yaw)
            
            # Broadcast TF Frame for the workpiece so it appears as a coordinate axes in RViz
            # We use the filtered coordinates to keep the TF perfectly stable.
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'base_link'
            t.child_frame_id = f"workpiece_{name}"
            
            t.transform.translation.x = float(x)
            t.transform.translation.y = float(y)
            # Apply the 0.6m Gazebo table offset. 
            # Note: We do NOT add the half-height here because the TF frame should represent the bottom origin of the object!
            t.transform.translation.z = float(z) - 0.6 
            
            quat = tf_transformations.quaternion_from_euler(roll, pitch, yaw)
            t.transform.rotation.x = quat[0]
            t.transform.rotation.y = quat[1]
            t.transform.rotation.z = quat[2]
            t.transform.rotation.w = quat[3]
            
            self.tf_broadcaster.sendTransform(t)
            
            # Apply the pose to the marker.
            # Here, we offset Z by half the height since the collision box must be centered vertically.
            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.pose.position.z = float(z) - 0.6 + float(dim[2] / 2.0)
            
            marker.pose.orientation.x = quat[0]
            marker.pose.orientation.y = quat[1]
            marker.pose.orientation.z = quat[2]
            marker.pose.orientation.w = quat[3]
            
            # Set scale to the collision dimensions
            marker.scale.x = float(dim[0])
            marker.scale.y = float(dim[1])
            marker.scale.z = float(dim[2])
            
            # Semi-transparent green for collision boxes
            marker.color.a = 0.5
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            
            marker_array.markers.append(marker)
            
        # Clean up any markers that are no longer present, WITH A GRACE PERIOD.
        # Gazebo's link_states sometimes randomly drops objects for a split second.
        # We wait for 4 frames (~2 seconds) before actually deleting the marker to prevent flickering!
        if not hasattr(self, 'missing_counts'):
            self.missing_counts = {}
            
        missing_ids = self.active_marker_ids - current_marker_ids
        
        for old_id in missing_ids:
            self.missing_counts[old_id] = self.missing_counts.get(old_id, 0) + 1
            if self.missing_counts[old_id] > 4: # Object has been missing for 4 consecutive frames
                del_marker = Marker()
                del_marker.header.frame_id = 'base_link'
                del_marker.ns = "gazebo_collisions"
                del_marker.id = old_id
                del_marker.action = Marker.DELETE
                marker_array.markers.append(del_marker)
                
                # Truly remove it from our active tracker
                self.active_marker_ids.remove(old_id)
                del self.missing_counts[old_id]
                if old_id in getattr(self, 'last_poses', {}):
                    del self.last_poses[old_id]
                
        # Reset the missing count for objects that are currently present
        for current_id in current_marker_ids:
            if hasattr(self, 'missing_counts') and current_id in self.missing_counts:
                del self.missing_counts[current_id]
            self.active_marker_ids.add(current_id)
            
        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = RVizVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
