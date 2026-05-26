#!/usr/bin/env python3
import argparse
import json
import os
import time
import zlib
import xml.etree.ElementTree as ET

import numpy as np
if not hasattr(np, 'float'):
    np.float = float
import tf_transformations

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.utilities import remove_ros_args
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


PICKABLE_MODELS = [
    'mustard_bottle',
    'coke_can',
    'strawberry',
    'meat_can',
    'hammer',
    'banana',
    'eraser',
    'biscuits',
    'snacks',
    'soap2',
    'soap',
    'glue',
    'book',
]


def parse_pose_text(text):
    if not text:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    values = [float(v) for v in text.split()]
    return (values + [0.0] * 6)[:6]


def parse_vector_text(text, size):
    if not text:
        return [0.0] * size
    values = [float(v) for v in text.split()]
    return (values + [0.0] * size)[:size]


def stable_marker_id(*parts):
    text = "::".join(str(part) for part in parts)
    return zlib.crc32(text.encode("utf-8")) & 0x7fffffff


def model_uri_to_package_uri(uri):
    if uri.startswith("model://"):
        rel = uri[len("model://"):]
        return "package://manip_challenge/data/models/" + rel
    return uri


class CollisionSpec:
    def __init__(self, name, kind, pose, size=None, radius=None, length=None,
                 mesh_uri=None, mesh_scale=None):
        self.name = name
        self.kind = kind
        self.pose = pose
        self.size = size
        self.radius = radius
        self.length = length
        self.mesh_uri = mesh_uri
        self.mesh_scale = mesh_scale or [1.0, 1.0, 1.0]


class RVizColliderVisualizer(Node):
    def __init__(self, target_names, world_topic, marker_topic, frame_id,
                 marker_namespace, z_offset, alpha, mesh_inflate):
        super().__init__('rviz_collider_visualizer_node')
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])

        self.target_names = set(target_names)
        self.world_topic = world_topic
        self.marker_topic = marker_topic
        self.frame_id = frame_id
        self.marker_namespace = marker_namespace
        self.z_offset = z_offset
        self.alpha = alpha
        self.mesh_inflate = mesh_inflate

        self.package_share = get_package_share_directory("manip_challenge")
        self.model_names = sorted(PICKABLE_MODELS, key=len, reverse=True)
        self.collision_specs = self.load_collision_specs()

        self.active_marker_ids = set()
        self.cleared_old_markers = False
        self.last_poses = {}
        self.last_status_log_time = 0.0

        self.marker_pub = self.create_publisher(MarkerArray, marker_topic, 10)
        self.create_subscription(String, world_topic, self.world_model_callback, 10)

        if self.target_names:
            self.get_logger().info(
                "RViz collider visualizer running for: "
                + ", ".join(sorted(self.target_names))
            )
        else:
            self.get_logger().info(
                "RViz collider visualizer running for all pickable table items."
            )
        self.get_logger().info(f"Publishing collider markers on {marker_topic}")

    def load_collision_specs(self):
        specs_by_model = {}
        for model_name in self.model_names:
            sdf_path = os.path.join(
                self.package_share, "data", "models", model_name, "model.sdf"
            )
            specs_by_model[model_name] = self.parse_model_sdf(model_name, sdf_path)
        return specs_by_model

    def parse_model_sdf(self, model_name, sdf_path):
        specs = []
        if not os.path.exists(sdf_path):
            self.get_logger().warning(f"No model.sdf found for {model_name}: {sdf_path}")
            return specs

        try:
            with open(sdf_path, "r", encoding="utf-8") as sdf_file:
                sdf_text = sdf_file.read().lstrip()
            root = ET.fromstring(sdf_text)
        except ET.ParseError as exc:
            self.get_logger().warning(f"Could not parse {sdf_path}: {exc}")
            return specs

        for idx, collision in enumerate(root.findall(".//collision")):
            name = collision.get("name", f"collision_{idx}")
            pose = parse_pose_text(collision.findtext("pose"))
            geometry = collision.find("geometry")
            if geometry is None:
                continue

            box = geometry.find("box")
            if box is not None:
                size = parse_vector_text(box.findtext("size"), 3)
                specs.append(CollisionSpec(name=name, kind="box", pose=pose, size=size))
                continue

            cylinder = geometry.find("cylinder")
            if cylinder is not None:
                radius = float(cylinder.findtext("radius", "0.0"))
                length = float(cylinder.findtext("length", "0.0"))
                specs.append(
                    CollisionSpec(
                        name=name, kind="cylinder", pose=pose,
                        radius=radius, length=length
                    )
                )
                continue

            sphere = geometry.find("sphere")
            if sphere is not None:
                radius = float(sphere.findtext("radius", "0.0"))
                specs.append(CollisionSpec(name=name, kind="sphere", pose=pose, radius=radius))
                continue

            mesh = geometry.find("mesh")
            if mesh is not None:
                uri = model_uri_to_package_uri(mesh.findtext("uri", ""))
                scale = parse_vector_text(mesh.findtext("scale", "1 1 1"), 3)
                specs.append(
                    CollisionSpec(
                        name=name, kind="mesh", pose=pose,
                        mesh_uri=uri, mesh_scale=scale
                    )
                )

        if not specs:
            self.get_logger().warning(f"No collision geometry found for {model_name}")
        return specs

    def base_model_name(self, object_name):
        for model_name in self.model_names:
            if object_name == model_name or object_name.startswith(model_name + "_"):
                return model_name
        return None

    def should_show_object(self, object_name, model_name):
        if not model_name:
            return False
        if not self.target_names:
            return True
        return object_name in self.target_names or model_name in self.target_names

    def filtered_pose(self, marker_key, pose):
        last_pose = self.last_poses.get(marker_key)
        if last_pose is None:
            self.last_poses[marker_key] = pose
            return pose

        deltas = [abs(a - b) for a, b in zip(pose, last_pose)]
        if (
            deltas[0] < 0.005 and deltas[1] < 0.005 and deltas[2] < 0.005
            and deltas[3] < 0.05 and deltas[4] < 0.05 and deltas[5] < 0.05
        ):
            return last_pose

        self.last_poses[marker_key] = pose
        return pose

    def make_marker(self, object_name, model_name, spec, object_pose):
        marker_id = stable_marker_id(object_name, spec.name)
        x, y, z, roll, pitch, yaw = self.filtered_pose(marker_id, object_pose)

        local_x, local_y, local_z, local_roll, local_pitch, local_yaw = spec.pose
        world_quat = tf_transformations.quaternion_from_euler(roll, pitch, yaw)
        local_quat = tf_transformations.quaternion_from_euler(
            local_roll, local_pitch, local_yaw
        )
        marker_quat = tf_transformations.quaternion_multiply(world_quat, local_quat)

        world_matrix = tf_transformations.quaternion_matrix(world_quat)
        local_offset = np.array([local_x, local_y, local_z, 1.0])
        rotated_offset = world_matrix.dot(local_offset)[:3]

        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = self.marker_namespace
        marker.id = marker_id
        marker.action = Marker.ADD
        marker.frame_locked = True

        marker.pose.position.x = float(x + rotated_offset[0])
        marker.pose.position.y = float(y + rotated_offset[1])
        marker.pose.position.z = float(z - self.z_offset + rotated_offset[2])
        marker.pose.orientation.x = float(marker_quat[0])
        marker.pose.orientation.y = float(marker_quat[1])
        marker.pose.orientation.z = float(marker_quat[2])
        marker.pose.orientation.w = float(marker_quat[3])

        if spec.kind == "box":
            marker.type = Marker.CUBE
            marker.scale.x = float(spec.size[0])
            marker.scale.y = float(spec.size[1])
            marker.scale.z = float(spec.size[2])
        elif spec.kind == "cylinder":
            marker.type = Marker.CYLINDER
            marker.scale.x = float(spec.radius * 2.0)
            marker.scale.y = float(spec.radius * 2.0)
            marker.scale.z = float(spec.length)
        elif spec.kind == "sphere":
            marker.type = Marker.SPHERE
            marker.scale.x = float(spec.radius * 2.0)
            marker.scale.y = float(spec.radius * 2.0)
            marker.scale.z = float(spec.radius * 2.0)
        elif spec.kind == "mesh":
            marker.type = Marker.MESH_RESOURCE
            marker.mesh_resource = spec.mesh_uri
            marker.mesh_use_embedded_materials = False
            marker.scale.x = float(spec.mesh_scale[0] * self.mesh_inflate)
            marker.scale.y = float(spec.mesh_scale[1] * self.mesh_inflate)
            marker.scale.z = float(spec.mesh_scale[2] * self.mesh_inflate)
        else:
            return None

        marker.color.a = float(self.alpha)
        marker.color.r = 0.05
        marker.color.g = 0.85
        marker.color.b = 1.0
        return marker

    def world_model_callback(self, msg):
        try:
            data = json.loads(msg.data)
            world_objects = data.get("world", [])
        except Exception as exc:
            self.get_logger().error(f"Failed to parse {self.world_topic}: {exc}")
            return

        marker_array = MarkerArray()
        if not self.cleared_old_markers:
            clear_marker = Marker()
            clear_marker.action = Marker.DELETEALL
            marker_array.markers.append(clear_marker)
            self.cleared_old_markers = True

        current_marker_ids = set()

        for obj in world_objects:
            object_name = obj.get("name", "")
            object_pose = obj.get("pose", [])
            if not object_name or len(object_pose) < 6:
                continue

            model_name = self.base_model_name(object_name)
            if not self.should_show_object(object_name, model_name):
                continue

            for spec in self.collision_specs.get(model_name, []):
                marker = self.make_marker(object_name, model_name, spec, object_pose[:6])
                if marker is None:
                    continue
                current_marker_ids.add(marker.id)
                marker_array.markers.append(marker)

        stale_marker_ids = self.active_marker_ids - current_marker_ids
        for marker_id in stale_marker_ids:
            delete_marker = Marker()
            delete_marker.header.frame_id = self.frame_id
            delete_marker.header.stamp = self.get_clock().now().to_msg()
            delete_marker.ns = self.marker_namespace
            delete_marker.id = marker_id
            delete_marker.action = Marker.DELETE
            marker_array.markers.append(delete_marker)
            self.last_poses.pop(marker_id, None)

        self.active_marker_ids = current_marker_ids
        self.marker_pub.publish(marker_array)

        now = time.monotonic()
        if now - self.last_status_log_time > 3.0:
            self.last_status_log_time = now
            self.get_logger().info(
                f"Published {len(current_marker_ids)} collider marker(s) "
                f"to {self.marker_topic}"
            )


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Publish RViz markers for Gazebo collision geometry."
    )
    parser.add_argument(
        "object_names", nargs="*",
        help="Optional full object name(s) or base model name(s), e.g. banana or banana_123_0.",
    )
    parser.add_argument("--world-topic", default="/world_model")
    parser.add_argument("--marker-topic", default="/gazebo_objects_markers")
    parser.add_argument("--marker-namespace", default="gazebo_visuals")
    parser.add_argument("--frame-id", default="base_link")
    parser.add_argument(
        "--z-offset", type=float, default=0.6,
        help="Gazebo world z offset to subtract when visualizing in base_link.",
    )
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument(
        "--mesh-inflate", type=float, default=1.08,
        help="Scale mesh colliders slightly so they are visible over the visual mesh.",
    )
    return parser.parse_args(remove_ros_args(argv)[1:])


def main(args=None):
    import sys

    parsed_args = parse_args(sys.argv if args is None else args)
    rclpy.init(args=args)
    node = RVizColliderVisualizer(
        target_names=parsed_args.object_names,
        world_topic=parsed_args.world_topic,
        marker_topic=parsed_args.marker_topic,
        frame_id=parsed_args.frame_id,
        marker_namespace=parsed_args.marker_namespace,
        z_offset=parsed_args.z_offset,
        alpha=parsed_args.alpha,
        mesh_inflate=parsed_args.mesh_inflate,
    )
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
