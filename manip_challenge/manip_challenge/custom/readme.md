1. Launch GAZEBO (Terminal 1)
ros2 launch manip_challenge ur5_setup.launch.py

2. Launch Server (Terminal 2)
python two_view_grasp_server.py

3. Send command (Terminal 3)
python two_view_grasp_client.py hammer left
python two_view_grasp_client.py meat_can right