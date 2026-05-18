#!/usr/bin/env python3
"""Convert a colored PLY point overlay into a self-contained Plotly HTML file."""

import argparse
import json
import re
from pathlib import Path

import numpy as np


def read_colored_ply(path):
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    vertex_count = None
    header_end = None
    for i, line in enumerate(lines):
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[2])
        if line.strip() == "end_header":
            header_end = i + 1
            break
    if vertex_count is None or header_end is None:
        raise RuntimeError(f"Invalid PLY: {path}")

    xyz = []
    rgb = []
    for line in lines[header_end:header_end + vertex_count]:
        parts = line.split()
        if len(parts) < 6:
            continue
        xyz.append([float(parts[0]), float(parts[1]), float(parts[2])])
        rgb.append([int(parts[3]), int(parts[4]), int(parts[5])])
    return np.asarray(xyz, dtype=np.float64), np.asarray(rgb, dtype=np.uint8)


def downsample(xyz, rgb, max_points):
    if max_points <= 0 or len(xyz) <= max_points:
        return xyz, rgb
    idx = np.linspace(0, len(xyz) - 1, max_points).astype(np.int64)
    return xyz[idx], rgb[idx]


def trace_for_color(name, xyz, rgb, color, size):
    mask = np.all(rgb == np.asarray(color, dtype=np.uint8), axis=1)
    pts = xyz[mask]
    rgb_string = f"rgb({color[0]},{color[1]},{color[2]})"
    return {
        "type": "scatter3d",
        "mode": "markers",
        "name": name,
        "x": pts[:, 0].tolist(),
        "y": pts[:, 1].tolist(),
        "z": pts[:, 2].tolist(),
        "marker": {"size": size, "color": rgb_string, "opacity": 0.9},
    }


def read_pose_matrix(path, pose_index):
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"pose_matrix:\s*!!opencv-matrix\s*\n\s*rows:\s*4\s*\n\s*cols:\s*4\s*\n\s*dt:\s*[df]\s*\n\s*data:\s*\[([^\]]+)\]", text, flags=re.S)
    if not matches:
        raise RuntimeError(f"No OpenCV pose_matrix found in {path}")
    if pose_index >= len(matches):
        raise RuntimeError(f"pose_index={pose_index} out of range. available={len(matches)}")
    values = np.fromstring(matches[pose_index].replace(",", " "), sep=" ", dtype=np.float64)
    if values.size != 16:
        raise RuntimeError(f"Expected 16 pose values, got {values.size}")
    return values.reshape(4, 4)


def orthonormalize_rotation(rotation):
    u, _, vt = np.linalg.svd(rotation)
    result = u @ vt
    if np.linalg.det(result) < 0:
        u[:, -1] *= -1
        result = u @ vt
    return result


def axis_trace(name, origin, direction, length, color):
    end = origin + direction * length
    return {
        "type": "scatter3d",
        "mode": "lines+markers",
        "name": name,
        "x": [float(origin[0]), float(end[0])],
        "y": [float(origin[1]), float(end[1])],
        "z": [float(origin[2]), float(end[2])],
        "line": {"color": color, "width": 8},
        "marker": {"size": [3, 5], "color": color},
    }


def main():
    parser = argparse.ArgumentParser(description="Create Plotly HTML from colored PLY overlay.")
    parser.add_argument("input_ply")
    parser.add_argument("output_html")
    parser.add_argument("--result", help="Optional PPF result .yml for RGB orientation axes.")
    parser.add_argument("--pose-index", type=int, default=0)
    parser.add_argument("--axis-length", type=float, default=0.08, help="Axis marker length in meters.")
    parser.add_argument("--max-points", type=int, default=20000)
    parser.add_argument("--point-size", type=float, default=3.0)
    args = parser.parse_args()

    xyz, rgb = read_colored_ply(args.input_ply)
    xyz, rgb = downsample(xyz, rgb, args.max_points)
    center = xyz.mean(axis=0)
    span = float(np.max(np.ptp(xyz, axis=0)))

    traces = [
        trace_for_color("scene foreground", xyz, rgb, (40, 180, 255), args.point_size),
        trace_for_color("transformed model", xyz, rgb, (255, 80, 40), args.point_size),
    ]

    axis_note = ""
    if args.result:
        pose = read_pose_matrix(args.result, args.pose_index)
        origin = pose[:3, 3]
        rotation = orthonormalize_rotation(pose[:3, :3])
        traces.extend([
            axis_trace("object X", origin, rotation[:, 0], args.axis_length, "rgb(255,0,0)"),
            axis_trace("object Y", origin, rotation[:, 1], args.axis_length, "rgb(0,200,0)"),
            axis_trace("object Z", origin, rotation[:, 2], args.axis_length, "rgb(0,80,255)"),
        ])
        axis_note = " axes: red=X, green=Y, blue=Z"

    layout = {
        "title": "PPF Pose Overlay",
        "scene": {
            "aspectmode": "data",
            "xaxis": {"title": "x"},
            "yaxis": {"title": "y"},
            "zaxis": {"title": "z"},
            "camera": {
                "eye": {"x": 1.4, "y": -1.8, "z": 1.0},
                "center": {"x": 0, "y": 0, "z": 0},
            },
        },
        "margin": {"l": 0, "r": 0, "b": 0, "t": 40},
        "annotations": [
            {
                "text": f"center=({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}), span={span:.3f}{axis_note}",
                "xref": "paper",
                "yref": "paper",
                "x": 0.01,
                "y": 0.01,
                "showarrow": False,
            }
        ],
    }

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>PPF Pose Overlay</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>html, body, #plot {{ width: 100%; height: 100%; margin: 0; }}</style>
</head>
<body>
  <div id="plot"></div>
  <script>
    const data = {json.dumps(traces)};
    const layout = {json.dumps(layout)};
    Plotly.newPlot('plot', data, layout, {{responsive: true}});
  </script>
</body>
</html>
"""
    Path(args.output_html).write_text(html, encoding="utf-8")
    print(f"wrote {args.output_html}")


if __name__ == "__main__":
    main()
