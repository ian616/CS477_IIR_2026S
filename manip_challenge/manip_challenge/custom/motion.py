#!/usr/bin/env python3
import time

import rclpy
import tf2_geometry_msgs  # noqa: F401 — registers PoseStamped with tf2
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, ConnectivityException, ExtrapolationException, LookupException


def transform_pose(node, tf_buffer, pose, source_frame, target_frame):
    """Transform a Pose from source_frame to target_frame. Returns None on failure."""
    try:
        t_pose = PoseStamped()
        t_pose.header.frame_id = source_frame
        t_pose.pose = pose
        transformed = tf_buffer.transform(
            t_pose, target_frame, timeout=rclpy.duration.Duration(seconds=1.0))
        node.get_logger().info(f"Transformed to {target_frame}: {transformed.pose.position}")
        return transformed.pose
    except (LookupException, ConnectivityException, ExtrapolationException) as e:
        node.get_logger().error(f"TF transform failed: {e}")
        return None


def wait_for_tf(node, tf_buffer, source_frame, target_frame):
    """Block until the TF transform between two frames becomes available."""
    while rclpy.ok():
        if tf_buffer.can_transform(
                target_frame, source_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)):
            break
        time.sleep(0.1)
