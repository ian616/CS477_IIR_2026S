#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import math

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

class HomeRobotClient(Node):
    def __init__(self):
        super().__init__('home_robot_client')
        
        # Setup Action Client for the low-level arm motor trajectory controller
        self.arm_client = ActionClient(self, FollowJointTrajectory, '/ur5_controller/follow_joint_trajectory')
        self.get_logger().info("Waiting for joint trajectory controller (/ur5_controller/follow_joint_trajectory)...")
        self.arm_client.wait_for_server()
        
        self.get_logger().info("Controller connected. Ready to home.")

    def command_arm_motion(self, joint_angles, duration=3.0):
        """Sends joint angles to the physical simulation controllers and blocks until completion."""
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = JOINT_NAMES
        
        point = JointTrajectoryPoint(
            positions=joint_angles,
            velocities=[0.0] * 6,
            time_from_start=Duration(sec=int(duration), nanosec=int((duration - int(duration)) * 1e9))
        )
        goal.trajectory.points.append(point)
        
        self.get_logger().info("Executing homing trajectory...")
        future = self.arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Homing goal rejected by controller.")
            return

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        self.get_logger().info("Robot is now at HOME.")

def main(args=None):
    rclpy.init(args=args)
    homer = HomeRobotClient()
    
    # Home joint angles matching init_joints.py: [0., -pi/2, 1., -pi/3, -pi/2, 0.]
    home_joints = [0.0, -math.pi/2.0, 1.0, -math.pi/3.0, -math.pi/2.0, 0.0]
    
    # Send the arm to the home position over 3 seconds
    homer.command_arm_motion(home_joints, duration=3.0)
    
    homer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
