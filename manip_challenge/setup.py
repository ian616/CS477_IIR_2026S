#!/usr/bin/env python3
import os
from glob import glob
from setuptools import find_packages, setup
from pathlib import Path
    
package_name = 'manip_challenge'


data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ]

def package_files(data_files, directory_list):

    paths_dict = {}
    for directory in directory_list:
        for (path, directories, filenames) in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(path, filename)
                install_path = os.path.join('share', package_name, path)
                if install_path in paths_dict.keys():
                    paths_dict[install_path].append(file_path)
                else:
                    paths_dict[install_path] = [file_path]

    for key in paths_dict.keys():
        data_files.append((key, paths_dict[key]))

    return data_files

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(include=[package_name, package_name + '.*']),
    package_data={
        'manip_challenge.custom.perception': ['model/best.pt'],
    },
    data_files=package_files(data_files, ['data/models/', 'launch/', 'data/worlds/', 'config']),
    install_requires=[
        'setuptools',
        'google-genai',
        'Pillow',
    ],
    zip_safe=True,
    author='Daehyung Park',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'example1_grasping         = manip_challenge.examples.example1_grasping:main',            
            'example2_gazebo_pose      = manip_challenge.examples.example2_gazebo_pose:main',
            'example3_task_command_pub = manip_challenge.examples.example3_task_command_pub:main',
            'example3_task_command_sub = manip_challenge.examples.example3_task_command_sub:main',
            'example4_detection_server = manip_challenge.examples.example4_detection_server:main',
            'example4_detection_client = manip_challenge.examples.example4_detection_client:main',
            'example5_grasp_server     = manip_challenge.examples.example5_grasp_server:main',
            'example5_grasp_client     = manip_challenge.examples.example5_grasp_client:main',
            'add_object_set1   = manip_challenge.add_object_set1:main',
            'add_object_set2   = manip_challenge.add_object_set2:main',
            'add_random_object = manip_challenge.add_random_object:main',
            'init_joints  = manip_challenge.init_joints:main',
            'world_model_gazebo = manip_challenge.world_model_gazebo:main',
            'get_joint    = manip_challenge.get_joint:main',            
            'get_pose     = manip_challenge.get_pose:main',            
            'move_gripper = manip_challenge.move_gripper:main',
            'move_joint   = manip_challenge.move_joint:main',            
            'custom_rgbd_crop_server = manip_challenge.custom.perception.rgbd_crop_server:main',
            'custom_rgbd_crop_client = manip_challenge.custom.perception.rgbd_crop_client:main',
            ],
    },
)
