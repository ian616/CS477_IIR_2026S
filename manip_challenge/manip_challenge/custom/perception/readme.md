# Custom Perception Pipeline

This folder contains the custom perception code for detecting one target object,
saving its RGB-D crop, and preparing the next step for 6D pose estimation with
LINE-MOD.

## Current Status

Done:

- Added a YOLO RGB-D crop server and client.
- The server uses `model/best.pt` to detect one of:
  - `coke_can`
  - `strawberry`
  - `meat_can`
  - `hammer`
  - `banana`
- The server subscribes to the wrist camera RGB, depth, and point cloud topics.
- The server saves the target object's RGB-D crop and metadata.
- Copied the five Gazebo object models into `gazebo_object/`.

Not done yet:

- LINE-MOD template generation from the Gazebo models.
- Converting the saved RGB-D crop into the exact input format expected by
  LINE-MOD.
- Running LINE-MOD detection and returning final 6D pose.
- Publishing the 6D pose as a ROS message or marker.

## Folder Layout

```text
perception/
  rgbd_crop_server.py
  rgbd_crop_client.py
  model/
    best.pt
  rgbd_crops/
    <timestamp>_<object>/
      rgb.png
      depth.npy
      depth_vis.png
      mask.png
      mask_on_rgb_size.png
      cloud.npy
      foreground_points.npy
      annotated.png
      metadata.json
  gazebo_object/
    banana/
    coke_can/
    hammer/
    meat_can/
    strawberry/
```

## Build

From the workspace root:

```bash
cd /home/lhs/CS477_IIR_2026S
colcon build --packages-select manip_challenge
source install/setup.bash
```

## Run RGB-D Crop Server

Start the perception server:

```bash
ros2 run manip_challenge custom_rgbd_crop_server
```

By default, saved RGB-D data goes to:

```text
/home/lhs/CS477_IIR_2026S/manip_challenge/manip_challenge/custom/perception/rgbd_crops/
```

The default ROS topics are:

```text
RGB image:   /wrist_camera/wrist_camera/color/image_raw
Depth image: /wrist_camera/wrist_camera/depth/color/image_raw
Point cloud: /wrist_camera/wrist_camera/depth/color/points
Service:     /detect_object_rgbd_crop
```

To override the save directory:

```bash
ros2 run manip_challenge custom_rgbd_crop_server --ros-args \
  -p save_dir:=/home/lhs/CS477_IIR_2026S/manip_challenge/manip_challenge/custom/perception/rgbd_crops
```

## Run Client

In another terminal:

```bash
cd /home/lhs/CS477_IIR_2026S
source install/setup.bash
```

Then request an object:

```bash
ros2 run manip_challenge custom_rgbd_crop_client coke_can
ros2 run manip_challenge custom_rgbd_crop_client strawberry
ros2 run manip_challenge custom_rgbd_crop_client meat_can
ros2 run manip_challenge custom_rgbd_crop_client hammer
ros2 run manip_challenge custom_rgbd_crop_client banana
```

Free-form commands also work if they contain one of the object names:

```bash
ros2 run manip_challenge custom_rgbd_crop_client "pick up the coke can"
```

The client prints JSON with:

- target label
- YOLO bbox
- padded bbox
- ROI centroid in the wrist camera frame
- saved directory
- saved file paths

## Saved RGB-D Data

Each request creates one folder:

```text
rgbd_crops/<timestamp>_<object>/
```

Files:

- `rgb.png`: cropped BGR/RGB image saved as PNG.
- `depth.npy`: raw cropped depth image. This preserves the original ROS depth
  dtype.
- `depth_vis.png`: colorized depth preview for debugging.
- `mask.png`: foreground object mask in the depth/point-cloud crop frame.
- `mask_on_rgb_size.png`: object mask resized to the RGB crop size.
- `cloud.npy`: organized XYZ point cloud crop.
- `foreground_points.npy`: foreground object points only.
- `annotated.png`: full image with bbox and ROI overlay.
- `metadata.json`: bbox, detected label, ROI stats, and centroid.

## Gazebo Model Data

The five object models are copied into:

```text
gazebo_object/
```

Important mesh files:

```text
banana/meshes/Banana.dae
coke_can/meshes/coke_can.dae
hammer/meshes/hammer.dae
meat_can/textured.obj
meat_can/textured.dae
strawberry/textured.obj
strawberry/textured.dae
```

These models are intended to be converted or adapted for LINE-MOD template
generation.

## Planned LINE-MOD Flow

The target pipeline should be:

1. Detect the requested object with YOLO.
2. Save RGB-D crop and metadata.
3. Prepare LINE-MOD model assets from `gazebo_object/`.
4. Generate LINE-MOD templates for the five objects.
5. Convert saved RGB-D data into LINE-MOD detection input.
6. Run LINE-MOD detection for the requested object.
7. Read the detected 6D pose:
   - translation
   - rotation quaternion
   - detection bbox
8. Convert translation from millimeters to meters if needed.
9. Publish or return the pose in the wrist camera frame.

## LINE-MOD Input and Output

The referenced LINE-MOD pipeline expects:

```text
input images:
  color image: cv::Mat
  depth image: cv::Mat, usually 16-bit single-channel depth

input model:
  class/model name matching the generated template class

input settings:
  linemod_settings.yml with camera intrinsics and template settings
```

The main output is an `ObjectPose`:

```text
translation: object position in camera coordinates
quaternions: object orientation
boundingBox: matched bbox
```

Important note: our current crop data is saved as `depth.npy`, but the reference
LINE-MOD code commonly reads 16-bit depth PNG with OpenCV. We should either:

- save an additional `depth.png` as `uint16`, or
- add `.npy` loading support to the LINE-MOD wrapper.

## Important Camera-Intrinsics Issue

LINE-MOD pose calculation depends on:

```text
fx, fy, cx, cy, image width, image height
```

If we run LINE-MOD on a crop, the principal point changes:

```text
crop_cx = original_cx - crop_left
crop_cy = original_cy - crop_top
```

Safer first implementation:

1. Keep full image size.
2. Place the RGB-D crop back into a black full-size canvas.
3. Keep the original camera intrinsics.
4. Run LINE-MOD on the full-size masked image.

This avoids modifying LINE-MOD camera settings per request.

## Next Tasks

1. Add `depth.png` saving in `rgbd_crop_server.py`.
2. Create a small conversion script for Gazebo meshes if LINE-MOD cannot read
   the current `.dae` or `.obj` files directly.
3. Build or install `aelmiger/LINE-MOD-Pipeline`.
4. Create `linemod_settings.yml` for the wrist camera.
5. Generate templates for:
   - `coke_can`
   - `strawberry`
   - `meat_can`
   - `hammer`
   - `banana`
6. Write a wrapper that takes one `rgbd_crops/<timestamp>_<object>/` folder and
   calls LINE-MOD.
7. Return the 6D pose through a ROS service.
