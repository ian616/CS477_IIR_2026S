# Grasping Submodule (`manip_challenge.grasping`)

This submodule contains the core components for the manipulation and grasping pipeline of the CS477 challenge. It integrates Ground Truth pose detection, YOLO/RGBD perception calibration, and RViz visualization for the UR5e arm.

## Core Components

### 1. Workspace Testing & Configuration
- **`ur5_moveit2.rviz`**: Pre-configured RViz layout used for testing and monitoring the manipulation workspace.

### 2. Calibration & Coordinate Transformation
- **`transform_calibration.ipynb`**: The central interactive Jupyter Notebook used to calibrate object transformations. It computes offsets between Gazebo Ground Truth (the object mesh origin), the TCP Grasp Frame, and Camera Perception (the YOLO RGBD "Red Dot" frame). It automatically loads and exports data using:
  - `config/calib_database.json`: Dynamic $(X, Y, Z, Roll, Pitch, Yaw)$ offsets.
  - `config/grasp_database.json`: Standard TCP grasping poses.
  - `config/seg_grasp_database.json`: The computed, combined translation matrices for runtime.

### 3. Perception & State Detection
- **`pose_detection.py`**: A robust ROS 2 client that interrogates the Gazebo `world_model` service. It dynamically resolves time-stamped object names (e.g. `banana_12345_0`) and returns highly accurate Base Link frame coordinates for the Ground Truth.

### 4. Visualization
- **`rviz_visualizer.py` & `rviz_visualizer_collider.py`**: Publishes TF markers and collision objects to RViz, allowing real-time visualization of the calculated grasp frames against the UR5e model.
