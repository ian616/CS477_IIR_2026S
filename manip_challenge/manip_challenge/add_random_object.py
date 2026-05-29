#!/usr/bin/env python3
"""
Random Bin Spawner (ROS 2 + Gazebo Classic)

- Randomly selects a model from `MODEL_LIST`
- Finds the model under GAZEBO_MODEL_PATH (or ~/.gazebo/models fallback)
- Spawns it at a random (x, y) within bounds
- Drops from 2–3 cm above a base height so objects stack stably

Requirements (typical):
  - ROS 2 + gazebo_ros_pkgs
  - Gazebo Classic running with gazebo_ros factory plugin (spawn_entity service)

Run:
  ros2 run <your_pkg> random_bin_spawner.py
or:
  python3 random_bin_spawner.py  (if ROS environment sourced)

Tune:
  X_MIN/X_MAX, Y_MIN/Y_MAX, Z_BASE, DROP_H_MIN/MAX, SETTLE_TIME
"""

import os, sys
import time
import random
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose

import numpy as np
from ament_index_python.packages import get_package_share_directory
import manip_challenge.misc as misc
import math

from rclpy.utilities import remove_ros_args

# sdf_path = os.path.join(get_package_share_directory('manip_challenge'), 'models')

# -----------------------------
# User config
# -----------------------------
MODEL_LIST = {
    # Gazebo model folder names and the max number of objects
    #"eraser": 3,
    "coke_can": 1, # 3
    "strawberry": 1, # 2
    "meat_can": 1, # 2
    #"mustard_bottle": 1,
    "hammer": 1, # 1
    "banana": 1, # 1
}

# Random spawn bounds (meters) in world/bin frame
X_MIN, X_MAX =  0.4, 0.65
Y_MIN, Y_MAX = -0.37, 0.37

# Base height (meters): set this to your bin floor height in the world frame,
# or top-of-pile estimate if you want to drop onto an existing stack.
Z_BASE = 0.6

# Drop from 2–3 cm above Z_BASE
DROP_H_MIN, DROP_H_MAX = 0.02, 0.03

# Random yaw rotation
YAW_MIN, YAW_MAX = -3.14159, 3.14159

# How many objects to spawn
NUM_SPAWNS = 5

# Time to wait after each spawn (seconds) to let physics settle
SETTLE_TIME = 0.6


class RandomBinSpawner(Node):
    def __init__(self, args):
        super().__init__("random_bin_spawner")
        self.model_list = args.model_list
        self.num_spawns = args.num_spawns        
        self.xmin, self.xmax = args.xmin, args.xmax
        self.ymin, self.ymax = args.ymin, args.ymax        
        self.z_base = args.z_base
        self.drop_h_min, self.drop_h_max = args.drop_h_min, args.drop_h_max
        self.yaw_min, self.yaw_max = args.yaw_min, args.yaw_max
        self.settle_time = args.settle_time
        self.reference_frame = args.reference_frame

        self.get_logger().info(
            'Creating Service client to connect to `/spawn_entity`')
        self.client = self.create_client(SpawnEntity, "/spawn_entity")

        self.get_logger().info("Connecting to `/spawn_entity` service...")
        if not self.client.service_is_ready():
            self.client.wait_for_service()
            self.get_logger().info("...connected!")

    def random_pose(self) -> Pose:
        pose = Pose()
        pose.position.x = random.uniform(self.xmin, self.xmax)
        pose.position.y = random.uniform(self.ymin, self.ymax)
        pose.position.z = self.z_base + random.uniform(self.drop_h_min, self.drop_h_max)
        
        # yaw: random
        yaw = random.uniform(self.yaw_min, self.yaw_max)
        
        # roll or pitch: choose axis and angle (0 or 90 deg)
        if random.choice([True, False]):
            roll = random.choice([0.0, math.pi / 2])
            pitch = 0.0
        else:
            roll = 0.0
            pitch = random.choice([0.0, math.pi / 2])
            
            # Euler (roll, pitch, yaw) → Quaternion
            cy = math.cos(yaw * 0.5)
            sy = math.sin(yaw * 0.5)
            cr = math.cos(roll * 0.5)
            sr = math.sin(roll * 0.5)
            cp = math.cos(pitch * 0.5)
            sp = math.sin(pitch * 0.5)
            
            pose.orientation.w = cr * cp * cy + sr * sp * sy
            pose.orientation.x = sr * cp * cy - cr * sp * sy
            pose.orientation.y = cr * sp * cy + sr * cp * sy
            pose.orientation.z = cr * cp * sy - sr * sp * cy        
            
        return pose

    def spawn_one_object(self, model, pose, entity_name=None):
        """ Spawn an object """
        
        # Set data for request
        sdf_file_path = os.path.join(
            get_package_share_directory("manip_challenge"), "data", "models",
            model, "model.sdf")
        request = SpawnEntity.Request()
        request.name = entity_name or model
        request.xml = open(sdf_file_path, 'r').read()
        request.robot_namespace = model
        request.initial_pose = pose
        
        self.get_logger().info("Sending service request to `/spawn_entity`")
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            print('response: %r' % future.result())
            return True
        else:
            raise RuntimeError(
                'exception while calling service: %r' % future.exception())
            return False
    
    def run(self):
        # random.seed(3)

        spawn_list = []
        for i in range(self.num_spawns):
            # Select a random object 
            while rclpy.ok():
                model, max_count = random.choice(list(self.model_list.items()))
                self.get_logger().info("{} <= {}".format(spawn_list.count(model), max_count))

                if spawn_list.count(model) < max_count:
                    break
                
            entity_name = f"{model}_{int(time.time()*1000)}_{i}"
            pose = self.random_pose()
            ok = self.spawn_one_object(model, pose, entity_name)
            spawn_list.append(model)
            
            # Let physics settle a bit for stable stacking
            if ok and self.settle_time > 0:
                time.sleep(self.settle_time)

    
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-list", nargs="+", default=MODEL_LIST)
    p.add_argument("--num-spawns", type=int, default=NUM_SPAWNS)
    
    p.add_argument("--xmin", type=float, default=X_MIN)
    p.add_argument("--xmax", type=float, default=X_MAX)
    p.add_argument("--ymin", type=float, default=Y_MIN)
    p.add_argument("--ymax", type=float, default=Y_MAX)

    p.add_argument("--z-base", type=float, default=Z_BASE)
    p.add_argument("--drop-h-min", type=float, default=DROP_H_MIN)
    p.add_argument("--drop-h-max", type=float, default=DROP_H_MAX)

    p.add_argument("--yaw-min", type=float, default=YAW_MIN)
    p.add_argument("--yaw-max", type=float, default=YAW_MAX)

    p.add_argument("--settle-time", type=float, default=SETTLE_TIME)

    p.add_argument("--reference-frame", type=str, default="world")

    non_ros_argv = remove_ros_args(sys.argv)
    return p.parse_args(non_ros_argv[1:])  # 프로그램 이름 제외

def main():
    args = parse_args()
    rclpy.init()
    node = RandomBinSpawner(args)
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
