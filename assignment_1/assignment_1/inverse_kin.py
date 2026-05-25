#!/usr/bin/env python3
"""
Copyright 2020 Daehyung Park

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import rclpy
import rclpy.node

from geometry_msgs.msg import PoseStamped, Point, Quaternion, PoseArray, Pose
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration

import numpy as np
from assignment_1 import move_joint

JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']


# --- Helper Functions for IK ---
def get_forward_kinematics(theta):
    c1, s1 = np.cos(theta[0]), np.sin(theta[0])
    c2, s2 = np.cos(theta[1]), np.sin(theta[1])
    q23 = theta[1] + theta[2]
    q234 = theta[1] + theta[2] + theta[3]
    c23, s23 = np.cos(q23), np.sin(q23)
    c234, s234 = np.cos(q234), np.sin(q234)
    c5, s5 = np.cos(theta[4]), np.sin(theta[4])

    px = c1*(0.425*c2 + 0.392*c23 - 0.095*s234 + 0.25*s5*c234) - (0.109 + 0.25*c5)*s1
    py = s1*(0.425*c2 + 0.392*c23 - 0.095*s234 + 0.25*s5*c234) + (0.109 + 0.25*c5)*c1
    pz = 0.089 - 0.425*s2 - 0.392*s23 - 0.095*c234 - 0.25*s5*s234
    return np.array([px, py, pz])

def move_position(node, goal_pose, init_joint=None):
    """ 
    A function to send a list of joint angles to the robot. 
    This function waits for completing the commanded motion.

    Parameters
    ----------
    node:  
        a ROS2 node handle
    goal_pose : Pose
        a Pose message from geometry_msgs
    init_joint : list
        a initial/current angle
    """

    #------------------------------------------------------------
    # ADD YOUR CODE
    #------------------------------------------------------------
    # 1) construct forward kinematics (problem 1)

    
    # 2) Get the start position from the init_joint via forward kinematics
    start_pos = get_forward_kinematics(init_joint)
    
    # 3) Get the goal position from the goal_pose
    goal_pos = np.array([goal_pose.position.x, goal_pose.position.y, goal_pose.position.z])
    
    # 4) Get a pose/position trajectory from start to goal positions
    pose_traj = np.linspace(start_pos, goal_pos, 100)

    
    # construct a goal message
    g = FollowJointTrajectory.Goal()
    g.trajectory = JointTrajectory()
    g.trajectory.joint_names = JOINT_NAMES
        
    # Get a sequence of joint angles that track the position trajecotory
    q = np.array(init_joint, dtype=float)
    dt = 0.05
    time_from_start = 0

    g.trajectory.points.append(
        JointTrajectoryPoint(positions=q.tolist(), velocities=[0.0]*6,
                             time_from_start=Duration(sec=0, nanosec=0))
    )

    check_points = [0, 24, 49, 74, 99]

    for i in range(1, len(pose_traj)):
        
        # Find a Jacobian
        c1, s1 = np.cos(q[0]), np.sin(q[0])
        c2, s2 = np.cos(q[1]), np.sin(q[1])
        q23 = q[1] + q[2]
        q234 = q[1] + q[2] + q[3]
        c23, s23 = np.cos(q23), np.sin(q23)
        c234, s234 = np.cos(q234), np.sin(q234)
        c5, s5 = np.cos(q[4]), np.sin(q[4])

        px = c1*(0.425*c2 + 0.392*c23 - 0.095*s234 + 0.25*s5*c234) - (0.109 + 0.25*c5)*s1
        py = s1*(0.425*c2 + 0.392*c23 - 0.095*s234 + 0.25*s5*c234) + (0.109 + 0.25*c5)*c1
        pz = 0.089 - 0.425*s2 - 0.392*s23 - 0.095*c234 - 0.25*s5*s234

        j1 = np.array([-py, px, 0, 0, 0, 1])
        j2 = np.array([c1*(pz-0.089), s1*(pz-0.089), -c1*px-s1*py, -s1, c1, 0])
        j3 = np.array([c1*(pz-0.089+0.425*s2), s1*(pz-0.089+0.425*s2), -c1*px-s1*py+0.425*c2, -s1, c1, 0])
        j4 = np.array([c1*(pz-0.089+0.425*s2+0.392*s23), s1*(pz-0.089+0.425*s2+0.392*s23), -c1*px-s1*py+0.425*c2+0.392*c23, -s1, c1, 0])
        j5 = np.array([0.25*(c1*c234*c5 + s1*s5), 0.25*(s1*c234*c5 - c1*s5), -0.25*s234*c5, -c1*s234, -s1*s234, -c234])
        j6 = np.array([0, 0, 0, c1*s234*s5-s1*c5, s1*s234*s5+c1*c5, c234*s5])
        
        J = np.stack([j1, j2, j3, j4, j5, j6], axis=1)
        J_p = J[:3]

        # Take Pseudo inverse
        J_inv = np.linalg.pinv(J_p)

        # Compute a delta position
        current_pos = get_forward_kinematics(q)
        dx = pose_traj[i] - current_pos

        # Compute a delta theta        
        dtheta = J_inv @ dx

        # Compute a desired theta
        q = q + dtheta
        time_from_start = time_from_start + dt
        g.trajectory.points.append(
            JointTrajectoryPoint(positions=q.tolist(), velocities=[0.0]*6,
                                 time_from_start=Duration(sec=int(time_from_start),
            nanosec=int((time_from_start-int(time_from_start))*1e9)))
            )
        

        if i in check_points:
            percent = int((i+1)/100 * 100)
            node.get_logger().info(f"{percent}% Joint Angles: {np.round(q, 4).tolist()}")
    #------------------------------------------------------------
    
    move_joint.move_joint(node, g)




def main(args=None):
    
    rclpy.init() 
    
    node = rclpy.create_node("arm_client")    
    rclpy.spin_once(node, timeout_sec=1)
    node.get_logger().info("init_ur5: direct control mode")

    # Problem 2
    # Move to the initial joint configuration
    theta    = [-0.8, -1.5708, 1.5708, -1.5708, -1.5708, -0.8]
    duration = 4
    
    g = FollowJointTrajectory.Goal()
    g.trajectory = JointTrajectory()
    g.trajectory.joint_names = JOINT_NAMES
    g.trajectory.points.append(
        JointTrajectoryPoint(positions=theta, velocities=[0]*6,
                                 time_from_start=Duration(sec=int(duration),
            nanosec=int((duration-int(duration))*1e9)))
        )        
    move_joint.move_joint(node, g)

    # Move following the linear position trajectory
    goal = Pose()
    goal.position.x = 0.418
    goal.position.y = 0.273
    goal.position.z = 0.264
    move_position(node, goal, theta)
    
    node.destroy_node()
    rclpy.shutdown()

    
if __name__ == '__main__':
    main()
        



