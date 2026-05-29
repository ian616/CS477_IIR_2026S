#!/usr/bin/env python3
"""Top-view YOLO segmentation RGB-D crop server.

This is the first two-view pipeline step: use the fixed overhead D435 to find a
target object and save its segmentation mask, RGB-D crop, organized point-cloud
crop, and foreground points. It intentionally does not run ICP.
"""

from pathlib import Path
import sys


THIS_FILE = Path(__file__).resolve()
PERCEPTION_DIR = THIS_FILE.parents[1]
PACKAGE_ROOT = THIS_FILE.parents[4]

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from manip_challenge.custom.perception.icp.rgbd_seg_crop_server import RgbdSegCropServiceNode  # noqa: E402


DEFAULTS = {
    "model_path": str(PERCEPTION_DIR / "model" / "yolov11_seg.pt"),
    "image_topic": "/camera/color/image_raw",
    "depth_topic": "/camera/depth/color/image_raw",
    "points_topic": "/camera/depth/color/points",
    "camera_frame": "camera_color_optical_frame",
    "service_name": "detect_object_top_rgbd_seg_crop",
    "display": "false",
    # Top-view training coverage appears weaker than wrist-view, so keep the
    # first-pass threshold low enough to inspect uncertain detections.
    "confidence": "0.15",
    # Digital zoom: run YOLO on the center 75% of RGB/depth/cloud. Set this to
    # 1.0 to disable, or smaller values such as 0.6 to zoom in more.
    "input_crop_ratio": "0.5",
    "save_dir": str(THIS_FILE.parent / "results" / "top_seg"),
    "annotated_image_topic": "/two_view/top_seg/detection_image",
    "roi_mask_topic": "/two_view/top_seg/mask",
    "roi_info_topic": "/two_view/top_seg/info",
    "roi_points_topic": "/two_view/top_seg/points",
    # Top view is farther away than the wrist camera, so allow a larger range.
    "max_depth_m": "3.0",
    # Keep this moderately tight for clutter. We will make this smarter in the
    # point-cloud filtering stage.
    "depth_margin_m": "0.035",
}


def has_param_override(argv, name):
    prefix = f"{name}:="
    return any(arg.startswith(prefix) or arg.endswith(prefix) for arg in argv)


def argv_with_default_params(argv):
    output = list(argv)
    additions = []
    for name, value in DEFAULTS.items():
        if not has_param_override(output, name):
            additions.extend(["-p", f"{name}:={value}"])
    if additions:
        output.extend(["--ros-args", *additions])
    return output


def main():
    import cv2
    import rclpy

    rclpy.init(args=argv_with_default_params(sys.argv))
    node = RgbdSegCropServiceNode()
    node.get_logger().info("Top-view segmentation crop server is using overhead camera defaults.")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
