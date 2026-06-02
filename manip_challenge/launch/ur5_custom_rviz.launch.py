#!/usr/bin/python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import xacro
import yaml

# LOAD FILE:
def load_file(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path, 'r') as file:
            return file.read()
    except EnvironmentError:
        return None

# LOAD YAML:
def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path, 'r') as file:
            return yaml.safe_load(file)
    except EnvironmentError:
        return None

def generate_launch_description():

    cell_layout_1 = "true"
    cell_layout_2 = "false"
    EE_no = "true"

    # ***** ROBOT DESCRIPTION ***** #
    ur5_description_path = os.path.join(get_package_share_directory('ur5_ros2_gazebo'))
    xacro_file = os.path.join(ur5_description_path, 'urdf', 'ur5.urdf.xacro')
    
    doc = xacro.parse(open(xacro_file))
    xacro.process_doc(doc, mappings={
        "cell_layout_1": cell_layout_1,
        "cell_layout_2": cell_layout_2,
        'hardware_interface': "PositionJointInterface",
        'camera_enabled': "false",
        "EE_no": EE_no,
    })
    robot_description_config = doc.toxml()
    robot_description = {'robot_description': robot_description_config}

    use_sim_time = LaunchConfiguration("use_sim_time", default="true")

    # *** PLANNING CONTEXT *** #
    robot_description_semantic_config = load_file("ur5_ros2_moveit2", "config/ur5.srdf")
    robot_description_semantic = {"robot_description_semantic": robot_description_semantic_config}
    
    kinematics_yaml = load_yaml("ur5_ros2_moveit2", "config/kinematics.yaml")

    ompl_planning_pipeline_config = {
        "move_group": {
            "planning_plugin": "ompl_interface/OMPLPlanner",
            "request_adapters": """default_planner_request_adapters/AddTimeOptimalParameterization default_planner_request_adapters/FixWorkspaceBounds default_planner_request_adapters/FixStartStateBounds default_planner_request_adapters/FixStartStateCollision default_planner_request_adapters/FixStartStatePathConstraints""",
            "start_state_max_bounds_error": 0.1,
        }
    }
    ompl_planning_yaml = load_yaml("ur5_ros2_moveit2", "config/ompl_planning.yaml")
    ompl_planning_pipeline_config["move_group"].update(ompl_planning_yaml)

    # INCLUDE ORIGINAL MOVEIT LAUNCH (Disable its built-in RViz)
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('ur5_ros2_moveit2'), 'launch'),
            '/ur5_moveit.launch.py'
        ]),
        launch_arguments={'rviz_file': 'True', 'use_sim_time': use_sim_time}.items(),
    )

    # RVIZ (From manip_challenge)
    rviz_base = os.path.join(get_package_share_directory("manip_challenge"), "config")
    rviz_full_config = os.path.join(rviz_base, "manip_ur5.rviz")
    rviz_node_full = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_full_config],
        parameters=[
            robot_description,
            robot_description_semantic,
            ompl_planning_pipeline_config,
            kinematics_yaml,
            {"use_sim_time": use_sim_time},
        ],
    )

    rviz_visualizer_collider = Node(
        package="manip_challenge",
        executable="rviz_visualizer_collider",
        name="rviz_visualizer_collider",
        output="screen",
    )

    return LaunchDescription([
        moveit_launch,
        rviz_node_full,
        rviz_visualizer_collider,
    ])
