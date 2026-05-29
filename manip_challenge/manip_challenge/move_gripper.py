#!/usr/bin/env python3
"""
Copyright 2020 Daehyung Park

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import time
import rclpy
import rclpy.node

from rclpy.action import ActionClient
from rosidl_runtime_py import message_to_yaml
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState

def gripper_open(node, force=1, timeout=1, gripper_open_pos=0.0):
    """ Open the parallel jaw gripper """
    #assert speed>=0.013, "Minimum speed is 0.013, but the speed is set to {}".format(speed)
    #assert force>=5, "Minimum force is 5, but the force is set to {}".format(force)
    return gripperGotoPos(node, gripper_open_pos, force=force, timeout=timeout)


def gripper_close(node, force=1, timeout=3, gripper_close_pos=0.8):
    """ Close the parallel jaw gripper """
    #assert speed>=0.013, "Minimum speed is 0.013, but the speed is set to {}".format(speed)
    # assert force>=5, "Minimum force is 5, but the force is set to {}".format(force)
    return gripperGotoPos(node, gripper_close_pos, force=force, timeout=timeout)


def gripperGotoPos(node, pos, force=1., timeout=3,
                       check_contact=False,
                       enable_wait=True,
                       enable_spin=True,
                       tolerance=0.05,
                       max_retries=10,
                       uuid=None, **kwargs):
    """
    Move the gripper finger to the designated position

    Parameters
    ----------
    node:  
        a ROS2 node handle
    pos : 
        a fingertip position where
        pos=0 indicates the status of OPEN      (radian) 0    ->  0       degrees
        pos=0.8 indicates the status of CLOSE   (radian) 0.8  ->  46.07   degrees
    """
    node.get_logger().info('run_gripper: gripperGotoPos')
    
    # Setup joint state subscriber if it doesn't exist
    if not hasattr(node, '_gripper_pos'):
        node._gripper_pos = None
        def js_callback(msg):
            if 'robotiq_85_left_knuckle_joint' in msg.name:
                idx = msg.name.index('robotiq_85_left_knuckle_joint')
                node._gripper_pos = msg.position[idx]
        
        node._js_sub = node.create_subscription(JointState, '/joint_states', js_callback, 10)

    try:
        client = ActionClient(node, FollowJointTrajectory,
                              '/gripper_controller/follow_joint_trajectory')
        client.wait_for_server(5)
        node.get_logger().info("Connected to gripper server")

        for attempt in range(max_retries):
            # Joint trajectory control
            g = FollowJointTrajectory.Goal()
            g.trajectory = JointTrajectory()
            g.trajectory.joint_names = ['robotiq_85_left_knuckle_joint']
            g.trajectory.points = [
                JointTrajectoryPoint(positions=[float(pos)], velocities=[0.0],
                                     time_from_start=Duration(sec=int(timeout), nanosec=int((timeout - int(timeout)) * 1e9))),
            ]

            node.get_logger().info(f'Attempt {attempt + 1}: Executing trajectory to pos {pos}')
            goal_future = client.send_goal_async(g)
            # Return result when the command is delivered
            rclpy.spin_until_future_complete(node, goal_future)

            goal_handle = goal_future.result()
            node.get_logger().info('goal_handle:\n {}'.format(goal_handle))
        
            # Wait until the execution ends and return the result
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(node, result_future)
        
            result = result_future.result()
            if result is None:
                raise RuntimeError(
                    'Exception while getting result: {!r}'.format(result_future.exception()))
            node.get_logger().info('Result:\n    {}'.format(message_to_yaml(result.result)))

            # Spin to process joint states
            for _ in range(10):
                rclpy.spin_once(node, timeout_sec=0.01)
                
            if node._gripper_pos is not None:
                error = abs(node._gripper_pos - pos)
                if error <= tolerance:
                    node.get_logger().info(f"Gripper reached target {pos} within tolerance (Error: {error:.4f}).")
                    return True
                else:
                    node.get_logger().warn(f"Tolerance not met. Target: {pos}, Actual: {node._gripper_pos:.4f}, Error: {error:.4f} > {tolerance}")
            else:
                node.get_logger().warn("Current position unknown (no joint states received).")
                
            time.sleep(0.5)

    except KeyboardInterrupt:
        rclpy.shutdown("KeyboardInterrupt")
        raise
    
    node.get_logger().error(f"Failed to reach target {pos} within tolerance after {max_retries} attempts.")
    rclpy.spin_once(node, timeout_sec=1)
    return False


def main(args=None):
    """ a main function """
    
    rclpy.init() 
    
    node = rclpy.create_node("gripper_client")    
    rclpy.spin_once(node, timeout_sec=1)
    node.get_logger().info("gripper_client: direct control mode")

    gripper_open(node)
    gripper_close(node)

    #node.destroy_node()

    rclpy.shutdown()

    
if __name__ == '__main__':
    main()
        


