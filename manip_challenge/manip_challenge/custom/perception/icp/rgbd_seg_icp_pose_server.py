#!/usr/bin/env python3
"""YOLO segmentation RGB-D crop service followed by ICP 6D pose estimation."""

from pathlib import Path

import cv2
import rclpy

try:
    from .icp_pose_pipeline import DEFAULT_MODEL_DIR, estimate_pose_for_crop
    from .rgbd_seg_crop_server import RgbdSegCropServiceNode
except ImportError:
    from icp_pose_pipeline import DEFAULT_MODEL_DIR, estimate_pose_for_crop
    from rgbd_seg_crop_server import RgbdSegCropServiceNode


class RgbdSegIcpPoseServiceNode(RgbdSegCropServiceNode):
    NODE_NAME = "rgbd_seg_icp_pose_service_node"
    DEFAULT_SERVICE_NAME = "detect_object_rgbd_seg_icp_pose"
    READY_LOG_NAME = "YOLO segmentation + ICP 6D pose service"

    def __init__(self):
        super().__init__()
        self.declare_parameter("icp_model_dir", str(DEFAULT_MODEL_DIR))
        self.declare_parameter("icp_scene_voxel_size", 0.005)
        self.declare_parameter("icp_model_voxel_size", 0.003)
        self.declare_parameter("icp_normal_radius", 0.015)
        self.declare_parameter("icp_normal_max_nn", 30)
        self.declare_parameter("icp_candidate_mode", "pca")
        self.declare_parameter("pose_axis_length", 0.08)

    def detect_and_save(self, target_label, request_text):
        info = super().detect_and_save(target_label, request_text)
        crop_dir = Path(info["save_dir"])
        object_name = info["target"]

        pose = estimate_pose_for_crop(
            crop_dir=crop_dir,
            object_name=object_name,
            model_dir=Path(self.get_parameter("icp_model_dir").value),
            scene_voxel_size=float(self.get_parameter("icp_scene_voxel_size").value),
            model_voxel_size=float(self.get_parameter("icp_model_voxel_size").value),
            normal_radius=float(self.get_parameter("icp_normal_radius").value),
            normal_max_nn=int(self.get_parameter("icp_normal_max_nn").value),
            candidate_mode=str(self.get_parameter("icp_candidate_mode").value),
            axis_length=float(self.get_parameter("pose_axis_length").value),
        )

        info["stage"] = "rgbd_seg_icp_pose_saved"
        info["message"] = "YOLO segmentation RGB-D crop and ICP 6D pose were saved."
        info["icp_pose"] = pose
        info["files"].update(pose.get("files", {}))

        pose_image = pose.get("files", {}).get("pose_axes_full")
        if pose_image:
            annotated = cv2.imread(str(pose_image), cv2.IMREAD_COLOR)
            if annotated is not None:
                self.annotated_img = annotated

        self.get_logger().info(
            f"ICP pose for {object_name}: fitness={pose['fitness']:.3f}, "
            f"rmse={pose['inlier_rmse']:.4f}m, t={pose['translation_xyz_m']}"
        )
        return info


def main():
    rclpy.init()
    node = RgbdSegIcpPoseServiceNode()
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
