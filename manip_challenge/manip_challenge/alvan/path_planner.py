#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import time

from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from manip_challenge import move_gripper

JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

class DirectCartesianPlanner(Node):
    def __init__(self):
        super().__init__('direct_cartesian_planner_node')
        
        # 1. Setup Client for MoveIt's IK solver service
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.get_logger().info("Waiting for MoveIt IK service (/compute_ik)...")
        self.ik_client.wait_for_service()
        
        # 2. Setup Action Client for the low-level arm motor trajectory controller
        self.arm_client = ActionClient(self, FollowJointTrajectory, '/ur5_controller/follow_joint_trajectory')
        self.get_logger().info("Waiting for joint trajectory controller...")
        self.arm_client.wait_for_server()
        
        self.get_logger().info("Path Planner MVP ready and online!")

    def request_joint_angles(self, x, y, z):
        """Calls MoveIt's IK service to convert X, Y, Z into 6 joint angles."""
        req = GetPositionIK.Request()
        req.ik_request.group_name = "ur5_arm"
        req.ik_request.pose_stamped.header.frame_id = "base_link"
        
        # Set target coordinates
        req.ik_request.pose_stamped.pose.position.x = x
        req.ik_request.pose_stamped.pose.position.y = y
        req.ik_request.pose_stamped.pose.position.z = z
        
        # Orientation quaternion pointing the gripper straight down towards the table
        req.ik_request.pose_stamped.pose.orientation.x = 0.0
        req.ik_request.pose_stamped.pose.orientation.y = 1.0
        req.ik_request.pose_stamped.pose.orientation.z = 0.0
        req.ik_request.pose_stamped.pose.orientation.w = 0.0
        
        future = self.ik_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        
        response = future.result()
        if response.error_code.val == 1: # SUCCESS
            # Extract the joint positions from the output state
            joint_positions = response.solution.joint_state.position[:6]
            return joint_positions
        else:
            self.get_logger().error(f"IK Solver failed to find a valid arm configuration. Error code: {response.error_code.val}")
            return None

    def command_arm_motion(self, joint_angles, duration=4.0):
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
        
        self.get_logger().info("Executing trajectory path segment...")
        future = self.arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        
        goal_handle = future.result()
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

    def run_pick_pipeline(self, target_x, target_y, target_z):
        """Executes the full pick sequence."""
        self.get_logger().info("--- STARTING PICK SEQUENCE ---")
        
        # Ensure gripper starts open
        move_gripper.gripper_open(self)
        time.sleep(1.0)
        
        # Step 1: Pre-Grasp (Hover 15cm above target object)
        hover_z = target_z + 0.15
        self.get_logger().info(f"Step 1: Calculating hover path to Z={hover_z:.2f}")
        hover_joints = self.request_joint_angles(target_x, target_y, hover_z)
        if hover_joints:
            self.command_arm_motion(hover_joints)
            
        # Step 2: Grasp (Lower down to the book grip zone)
        grip_z = target_z + 0.06
        self.get_logger().info(f"Step 2: Descending to grip height Z={grip_z:.2f}")
        grip_joints = self.request_joint_angles(target_x, target_y, grip_z)
        if grip_joints:
            self.command_arm_motion(grip_joints, duration=2.5)
            
        # Step 3: Actuate Gripper Close
        self.get_logger().info("Step 3: Closing gripper fingers...")
        move_gripper.gripper_close(self)
        time.sleep(1.0)
        
        # Step 4: Post-Grasp Retreat (Lift back up with the object)
        self.get_logger().info("Step 4: Retreating straight upwards...")
        if hover_joints:
            self.command_arm_motion(hover_joints, duration=3.0)
            
        self.get_logger().info("--- PICK SEQUENCE SEQUENCE COMPLETION ---")

def main(args=None):
    
    rclpy.init(args=args)
    planner = DirectCartesianPlanner()
    
    # Standard location coordinates for the spawned book model scene
    book_x = 0.12
    book_y = -0.50
    book_z = 0.15 # 0.05
    
    planner.run_pick_pipeline(book_x, book_y, book_z)
    
    planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()