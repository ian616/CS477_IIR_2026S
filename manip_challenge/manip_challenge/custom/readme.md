1. Launch GAZEBO (Terminal 1)
ros2 launch manip_challenge ur5_setup.launch.py
ros2 launch manip_challenge ur5_setup_random_picking.launch.py

2. Launch Server (Terminal 2)
python3 two_view_grasp_server.py

3. Send command (Terminal 3)
python3 two_view_grasp_client.py hammer left
python3 two_view_grasp_client.py meat_can right
python3 two_view_grasp_client.py "Move a banana to left storage."
python3 two_view_grasp_client.py "Move a banana, a coke can, and a hammer to left storage."
python3 two_view_grasp_client.py "Move a coke can to left. Move a hammer to right storage."

# Or publish directly via topic
ros2 topic pub --once /task_commands std_msgs/msg/String "data: 'Move a coke can to left. Move a hammer to right storage.'"
