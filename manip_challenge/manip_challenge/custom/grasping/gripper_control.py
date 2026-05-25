#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import sys
import time

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

JOINT_NAME = 'robotiq_85_left_knuckle_joint'

class GripperClient(Node):
    def __init__(self):
        super().__init__('gripper_client_node')
        
        # Setup Action Client for the Robotiq Gripper
        self.gripper_client = ActionClient(self, FollowJointTrajectory, '/gripper_controller/follow_joint_trajectory')
        self.get_logger().info("Waiting for gripper controller (/gripper_controller/follow_joint_trajectory)...")
        self.gripper_client.wait_for_server()
        
        self.get_logger().info("Gripper Controller connected.")

    def command_gripper(self, position, duration=1.0):
        """
        Sends a single joint angle to the gripper.
        Robotiq 85: 0.0 is fully open, ~0.8 is fully closed.
        """
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = [JOINT_NAME]
        
        point = JointTrajectoryPoint(
            positions=[float(position)],
            velocities=[0.0],
            time_from_start=Duration(sec=int(duration), nanosec=int((duration - int(duration)) * 1e9))
        )
        goal.trajectory.points.append(point)
        
        state_str = "CLOSING" if position > 0.4 else "OPENING"
        self.get_logger().info(f"Executing {state_str} trajectory (position: {position})...")
        
        future = self.gripper_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Gripper goal rejected.")
            return False
            
        self.get_logger().info("Goal accepted, waiting for execution...")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        self.get_logger().info(f"Gripper {state_str} motion complete.")
        return True

def main(args=None):
    rclpy.init(args=args)
    node = GripperClient()
    
    command = "test"
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
    try:
        if command in ["on", "close"]:
            node.command_gripper(0.8) # 0.8 is closed
        elif command in ["off", "open"]:
            node.command_gripper(0.0) # 0.0 is open
        else:
            # Default test sequence if no argument is passed
            node.get_logger().info("No argument passed. Running Test Sequence: Close then Open...")
            node.command_gripper(0.8, 1.5)
            time.sleep(1.0)
            node.command_gripper(0.0, 1.5)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
