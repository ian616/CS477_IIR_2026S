# Two-View Grasp Pipeline

This folder contains the new two-view grasp pipeline. Step 1 uses the fixed
top-view camera to save a YOLO segmentation RGB-D crop for a requested object.
It reuses the existing `yolov11_seg.pt` model and crop utilities.

## Step 1: Top-View Segmentation Crop

Start the simulator, then run the top-view segmentation server from a sourced
ROS workspace:

```bash
cd /home/lhs/CS477_IIR_2026S
source install/setup.bash
python3 manip_challenge/manip_challenge/custom/perception/two-view/top_view_seg_server.py
```

In another terminal:

```bash
cd /home/lhs/CS477_IIR_2026S
source install/setup.bash
python3 manip_challenge/manip_challenge/custom/perception/two-view/top_view_seg_client.py banana
```

Default top-view topics:

```text
RGB image:   /camera/camera/color/image_raw
Depth image: /camera/camera/depth/color/image_raw
Point cloud: /camera/camera/depth/color/points
Service:     /detect_object_top_rgbd_seg_crop
```

Default outputs are saved under:

```text
manip_challenge/manip_challenge/custom/perception/two-view/results/top_seg/
```

Useful debug topics:

```text
/two_view/top_seg/detection_image
/two_view/top_seg/mask
/two_view/top_seg/info
/two_view/top_seg/points
```

If the simulator exposes different top camera topics, override them:

```bash
python3 manip_challenge/manip_challenge/custom/perception/two-view/top_view_seg_server.py --ros-args \
  -p image_topic:=/camera/camera/color/image_raw \
  -p depth_topic:=/camera/camera/depth/color/image_raw \
  -p points_topic:=/camera/camera/depth/color/points
```
