#!/usr/bin/env python3

import copy
import math

import numpy as np


def normalize_quaternion(q):
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if math.isclose(norm, 0.0):
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return q / norm


def pose_quaternion(pose):
    q = np.array([
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ], dtype=float)
    return normalize_quaternion(q)


def quaternion_from_euler(roll, pitch, yaw):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    return normalize_quaternion([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])


def quaternion_multiply(q1, q2):
    x1, y1, z1, w1 = normalize_quaternion(q1)
    x2, y2, z2, w2 = normalize_quaternion(q2)
    return normalize_quaternion([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def rotate_vector(q, vector):
    q = normalize_quaternion(q)
    q_xyz = q[:3]
    w = q[3]
    v = np.asarray(vector, dtype=float)

    return v + 2.0 * (
        w * np.cross(q_xyz, v) + np.cross(q_xyz, np.cross(q_xyz, v))
    )


def apply_grasp_transform(pose, transform):
    corrected_pose = copy.deepcopy(pose)

    base_q = pose_quaternion(pose)
    offset = np.array([transform["x"], transform["y"], transform["z"]], dtype=float)
    dx, dy, dz = rotate_vector(base_q, offset)

    corrected_pose.position.x += float(dx)
    corrected_pose.position.y += float(dy)
    corrected_pose.position.z += float(dz)

    local_q = quaternion_from_euler(
        transform["roll"],
        transform["pitch"],
        transform["yaw"],
    )
    corrected_q = quaternion_multiply(base_q, local_q)

    corrected_pose.orientation.x = float(corrected_q[0])
    corrected_pose.orientation.y = float(corrected_q[1])
    corrected_pose.orientation.z = float(corrected_q[2])
    corrected_pose.orientation.w = float(corrected_q[3])

    return corrected_pose
