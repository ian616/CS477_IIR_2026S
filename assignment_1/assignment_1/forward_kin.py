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
import numpy as np

from geometry_msgs.msg import Point, Quaternion, PoseArray, Pose, Wrench
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from assignment_1 import move_joint

JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']



def your_forward_kinematics(theta):
    """ 
    Compute the end-effector pose given a set of joint angles (radians) 
    Based on the handwritten equations for UR5 Modified DH
    """
    import numpy as np
    from geometry_msgs.msg import Pose
    
    c1, s1 = np.cos(theta[0]), np.sin(theta[0])
    c2, s2 = np.cos(theta[1]), np.sin(theta[1])
    c5, s5 = np.cos(theta[4]), np.sin(theta[4])
    c6, s6 = np.cos(theta[5]), np.sin(theta[5])
    
    q23 = theta[1] + theta[2]
    q234 = theta[1] + theta[2] + theta[3]
    c23, s23 = np.cos(q23), np.sin(q23)
    c234, s234 = np.cos(q234), np.sin(q234)

    px = c1*(0.425*c2 + 0.392*c23 - 0.095*s234 + 0.25*s5*c234) - (0.109 + 0.25*c5)*s1
    py = s1*(0.425*c2 + 0.392*c23 - 0.095*s234 + 0.25*s5*c234) + (0.109 + 0.25*c5)*c1
    pz = 0.089 - 0.425*s2 - 0.392*s23 - 0.095*c234 - 0.25*s5*s234

    r11 = c6*(c1*c5*c234 + s1*s5) - s6*c1*s234
    r12 = -s6*(c1*c5*c234 + s1*s5) - c6*c1*s234
    r13 = c1*s5*c234 - s1*c5
    
    r21 = c6*(s1*c5*c234 - c1*s5) - s6*s1*s234
    r22 = -s6*(s1*c5*c234 - c1*s5) - c6*s1*s234
    r23 = s1*s5*c234 + c1*c5
    
    r31 = -c5*c6*s234 - s6*c234
    r32 = c5*s6*s234 - c6*c234
    r33 = -s5*s234

    tr = r11 + r22 + r33
    
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2 
        qw = 0.25 * S
        qx = (r32 - r23) / S
        qy = (r13 - r31) / S
        qz = (r21 - r12) / S
    elif (r11 > r22) and (r11 > r33):
        S = np.sqrt(1.0 + r11 - r22 - r33) * 2
        qw = (r32 - r23) / S
        qx = 0.25 * S
        qy = (r12 + r21) / S
        qz = (r13 + r31) / S
    elif r22 > r33:
        S = np.sqrt(1.0 + r22 - r11 - r33) * 2
        qw = (r13 - r31) / S
        qx = (r12 + r21) / S
        qy = 0.25 * S
        qz = (r23 + r32) / S
    else:
        S = np.sqrt(1.0 + r33 - r11 - r22) * 2
        qw = (r21 - r12) / S
        qx = (r13 + r31) / S
        qy = (r23 + r32) / S
        qz = 0.25 * S

    ps = Pose()
    ps.position.x = float(px)
    ps.position.y = float(py)
    ps.position.z = float(pz)
    ps.orientation.x = float(qx)
    ps.orientation.y = float(qy)
    ps.orientation.z = float(qz)
    ps.orientation.w = float(qw)
    
    return ps

def send_command(node, theta, duration=3):
    
    # construct a goal message
    g = FollowJointTrajectory.Goal()
    g.trajectory = JointTrajectory()
    g.trajectory.joint_names = JOINT_NAMES
    g.trajectory.points.append(
        JointTrajectoryPoint(positions=theta, velocities=[0]*6,
                                 time_from_start=Duration(sec=int(duration),
            nanosec=int((duration-int(duration))*1e9)))
        )    
    move_joint.move_joint(node, g)


def main(args=None):
    
    rclpy.init() 
    
    node = rclpy.create_node("arm_client")    
    rclpy.spin_once(node, timeout_sec=1)
    node.get_logger().info("init_ur5: direct control mode")
    
    #------------------------------------------------------------
    # ADD YOUR CODE
    #------------------------------------------------------------
    # Place your desired joint angles! 
    
    # Problem 1.C (i)
    # theta = np.deg2rad([0, -90, 90, -90, -90, 0]).tolist()
    # node.get_logger().info("{}".format(your_forward_kinematics(theta)))
    # send_command(node, theta)


    # Problem 1.C (ii)
    theta = np.deg2rad([-45, -45, 90, 0, 0, 0]).tolist()
    node.get_logger().info("{}".format(your_forward_kinematics(theta)))
    send_command(node, theta)
    #------------------------------------------------------------
    
    node.destroy_node()
    rclpy.shutdown()

    
if __name__ == '__main__':
    main()
        



