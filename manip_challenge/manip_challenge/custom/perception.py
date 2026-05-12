#!/usr/bin/env python3
import os
import re

import cv2
import numpy as np
from cv_bridge import CvBridge
from google import genai
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from riro_srvs.srv import StringPose
from sensor_msgs.msg import Image, PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
import std_msgs.msg
from geometry_msgs.msg import Pose
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        self.declare_parameter('api_key', '')
        api_key = os.getenv('GEMINI_API_KEY') or \
                  self.get_parameter('api_key').get_parameter_value().string_value
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. Export the env var or pass as ROS parameter.")

        self.client = genai.Client(api_key=api_key)
        self.model = 'gemini-3.1-flash-lite-preview'

        best_effort_qos = QoSProfile(depth=10)
        best_effort_qos.reliability = ReliabilityPolicy.BEST_EFFORT

        self.create_subscription(Image, '/wrist_camera/image_raw',
                                 self.image_callback, best_effort_qos)
        self.create_subscription(Image, '/wrist_camera/depth/image_raw',
                                 self.depth_callback, best_effort_qos)
        self.create_subscription(PointCloud2, '/wrist_camera/points',
                                 self.points_callback, best_effort_qos)

        self.srv = self.create_service(
            StringPose, 'detect_objects_with_prompt', self.detect_callback)

        self.create_timer(0.1, self.timer_callback)

        latched_qos = QoSProfile(depth=10)
        latched_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        latched_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.cloud_pub = self.create_publisher(PointCloud2, '/roi_filtered_points', latched_qos)

        self.bridge = CvBridge()
        self.latest_cv_img = None
        self.latest_depth_img = None
        self.latest_cloud = None
        self.cv_img = None

        self.get_logger().info('Perception Node is Ready!')

    def image_callback(self, msg):
        self.latest_cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def depth_callback(self, msg):
        self.latest_depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def points_callback(self, msg):
        raw = pc2.read_points_numpy(msg, field_names=('x', 'y', 'z'))
        if msg.height > 1:
            self.latest_cloud = raw.reshape(msg.height, msg.width, 3)
        else:
            self.latest_cloud = raw

    def timer_callback(self):
        if self.cv_img is not None:
            cv2.imshow('Detection Result', self.cv_img)
            cv2.waitKey(1)

    def detect_callback(self, request, response):
        user_prompt = request.data or 'Detect objects and return [ymin, xmin, ymax, xmax, label]'

        if self.latest_cv_img is None or self.latest_cloud is None or self.latest_depth_img is None:
            self.get_logger().warn('Camera image or point cloud not received.')
            return response

        try:
            self.get_logger().info(f'Received Prompt: {user_prompt}')

            rgb_img = cv2.cvtColor(self.latest_cv_img, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(rgb_img)

            format_instruction = (
                '\n\nOutput format: [ymin, xmin, ymax, xmax, \'label\'] '
                'using normalized coordinates (0-1000). '
                'Only return the list, no other text.')

            vlm_response = self.client.models.generate_content(
                model=self.model,
                contents=[pil_img, user_prompt + format_instruction])
            result_text = vlm_response.text
            self.get_logger().info(f'Received Result: {result_text}')

            pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*["\']?([\w\s]+)["\']?\]'
            matches = re.findall(pattern, result_text)
            self.get_logger().info(f'Found {len(matches)} objects.')

            match = matches[0]
            ymin, xmin, ymax, xmax, label = (
                int(match[0]), int(match[1]), int(match[2]), int(match[3]), match[4].strip())
            self.get_logger().info(f'match: {match}')

            h, w, _ = self.latest_cv_img.shape
            left   = int(xmin * w / 1000)
            top    = int(ymin * h / 1000)
            right  = int(xmax * w / 1000)
            bottom = int(ymax * h / 1000)
            self.get_logger().info(f'ltrb: {left}, {top}, {right}, {bottom}')

            center_3d = self._get_bbox_center_3d(top, left, bottom, right)
            if center_3d is not None:
                response.pose.position.x = float(center_3d[0])
                response.pose.position.y = float(center_3d[1])
                response.pose.position.z = float(center_3d[2])
                self.get_logger().info(f'3D Center: {center_3d}')
            else:
                self.get_logger().warn('No valid depth points in BBox.')
                return response

            vis_img = self.latest_cv_img.copy()
            center_x = (left + right) // 2
            center_y = (top + bottom) // 2
            cv2.rectangle(vis_img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.circle(vis_img, (center_x, center_y), 5, (0, 0, 255), -1)
            cv2.putText(vis_img, f'{label} ({center_x}, {center_y})',
                        (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            self.cv_img = vis_img.copy()
            self.get_logger().info(f'Target Center: x={center_x}, y={center_y}')

        except Exception as e:
            self.get_logger().error(f'Error: {e}')

        return response

    def _get_bbox_center_3d(self, py1, px1, py2, px2):
        """Return mean 3D position of the top-2cm points inside the bounding box."""
        if self.latest_cloud is None:
            return None

        h, w, _ = self.latest_cloud.shape
        py1, py2 = np.clip([py1, py2], 0, h - 1)
        px1, px2 = np.clip([px1, px2], 0, w - 1)
        roi = self.latest_cloud[py1:py2, px1:px2]

        valid = roi[~np.isnan(roi).any(axis=2)]
        if len(valid) == 0:
            self.get_logger().info('No valid points in ROI.')
            return None

        self._publish_cloud(valid)

        z_min = np.min(valid[:, 2])
        top_mask = (valid[:, 2] >= z_min) & (valid[:, 2] <= z_min + 0.02)
        top_points = valid[top_mask]
        if top_points.size == 0:
            return None

        return np.mean(top_points, axis=0)

    def _publish_cloud(self, points, frame_id='wrist_camera_color_optical_frame'):
        header = std_msgs.msg.Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id

        fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        data = points[:, :3].astype(np.float32).tobytes()
        self.cloud_pub.publish(PointCloud2(
            header=header,
            height=1,
            width=len(points),
            is_dense=True,
            is_bigendian=False,
            fields=fields,
            point_step=12,
            row_step=12 * len(points),
            data=data,
        ))


def main():
    rclpy.init()
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
