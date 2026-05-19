#!/usr/bin/env python3
"""Create visual checks for prepared ICP model point clouds."""

import argparse
import html
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = SCRIPT_DIR / "icp_models"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "visualizations"

np = None
o3d = None


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize prepared ICP .ply models and their normals.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Directory containing <object>.ply/.json files.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for generated PNG/HTML files.")
    parser.add_argument("--objects", nargs="+", default=None, help="Object names to visualize. Defaults to every .ply file.")
    parser.add_argument("--max-points", type=int, default=1000, help="Maximum displayed points per object.")
    parser.add_argument("--normal-count", type=int, default=80, help="Number of normal arrows to display.")
    parser.add_argument("--normal-scale", type=float, default=0.015, help="Normal arrow length in meters.")
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def require_deps():
    global np, o3d
    if np is not None and o3d is not None:
        return
    try:
        import numpy as numpy_module
        import open3d as open3d_module
    except ImportError as exc:
        raise ImportError("numpy and open3d are required for ICP model visualization.") from exc
    np = numpy_module
    o3d = open3d_module


def discover_objects(model_dir, requested):
    if requested:
        return requested
    return sorted(path.stem for path in model_dir.glob("*.ply"))


def load_cloud(model_dir, object_name):
    ply_path = model_dir / f"{object_name}.ply"
    metadata_path = model_dir / f"{object_name}.json"
    cloud = o3d.io.read_point_cloud(str(ply_path))
    if cloud.is_empty():
        raise RuntimeError(f"Empty or missing point cloud: {ply_path}")
    metadata = {}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return cloud, metadata, ply_path, metadata_path


def outward_scores(points, normals):
    center = points.mean(axis=0)
    rays = points - center
    ray_norms = np.linalg.norm(rays, axis=1)
    normal_norms = np.linalg.norm(normals, axis=1)
    valid = (ray_norms > 1e-12) & (normal_norms > 1e-12)
    scores = np.zeros(len(points), dtype=np.float64)
    scores[valid] = np.einsum("ij,ij->i", normals[valid], rays[valid]) / (ray_norms[valid] * normal_norms[valid])
    return scores


def sample_indices(count, limit, seed):
    if count <= limit:
        return np.arange(count)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(count, size=limit, replace=False))


def bbox_segments(points):
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    corners = np.array(
        [
            [mn[0], mn[1], mn[2]],
            [mx[0], mn[1], mn[2]],
            [mx[0], mx[1], mn[2]],
            [mn[0], mx[1], mn[2]],
            [mn[0], mn[1], mx[2]],
            [mx[0], mn[1], mx[2]],
            [mx[0], mx[1], mx[2]],
            [mn[0], mx[1], mx[2]],
        ],
        dtype=np.float64,
    )
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    return corners, edges


def set_equal_axes(ax, points):
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    center = (mn + mx) * 0.5
    radius = max(float(np.max(mx - mn)) * 0.55, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except AttributeError:
        pass


def save_png(object_name, points, normals, scores, metadata, output_dir, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    point_idx = sample_indices(len(points), args.max_points, args.seed)
    normal_idx = sample_indices(len(points), min(args.normal_count, len(points)), args.seed + 1)

    fig = plt.figure(figsize=(8, 7), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        points[point_idx, 0],
        points[point_idx, 1],
        points[point_idx, 2],
        c=scores[point_idx],
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        s=7,
        alpha=0.9,
    )
    ax.quiver(
        points[normal_idx, 0],
        points[normal_idx, 1],
        points[normal_idx, 2],
        normals[normal_idx, 0],
        normals[normal_idx, 1],
        normals[normal_idx, 2],
        length=args.normal_scale,
        normalize=True,
        color="#1f77b4",
        linewidth=0.6,
    )

    corners, edges = bbox_segments(points)
    for start, end in edges:
        segment = corners[[start, end]]
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color="black", linewidth=0.9, alpha=0.65)

    center = points.mean(axis=0)
    axis_len = max(float(np.max(points.max(axis=0) - points.min(axis=0))) * 0.2, 0.01)
    ax.quiver(center[0], center[1], center[2], axis_len, 0, 0, color="red", linewidth=1.5)
    ax.quiver(center[0], center[1], center[2], 0, axis_len, 0, color="green", linewidth=1.5)
    ax.quiver(center[0], center[1], center[2], 0, 0, axis_len, color="blue", linewidth=1.5)

    outward_ratio = float(np.mean(scores > 0.0))
    extent = points.max(axis=0) - points.min(axis=0)
    ax.set_title(f"{object_name} | points={len(points)} | outward={outward_ratio:.3f}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    set_equal_axes(ax, points)
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.72, pad=0.08)
    colorbar.set_label("normal outward score")
    fig.text(
        0.02,
        0.02,
        f"extent xyz (m): {extent[0]:.4f}, {extent[1]:.4f}, {extent[2]:.4f}\n"
        f"mesh loader: {metadata.get('mesh_loader', 'unknown')}",
        fontsize=8,
    )
    png_path = output_dir / f"{object_name}_model_normals.png"
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)
    return png_path


def trace_lines(points_a, points_b):
    xs, ys, zs = [], [], []
    for start, end in zip(points_a, points_b):
        xs.extend([start[0], end[0], None])
        ys.extend([start[1], end[1], None])
        zs.extend([start[2], end[2], None])
    return xs, ys, zs


def save_html(object_name, points, normals, scores, metadata, output_dir, args):
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    point_idx = sample_indices(len(points), args.max_points, args.seed)
    normal_idx = sample_indices(len(points), min(args.normal_count, len(points)), args.seed + 1)
    normal_starts = points[normal_idx]
    normal_ends = normal_starts + normals[normal_idx] * args.normal_scale
    nx, ny, nz = trace_lines(normal_starts, normal_ends)

    corners, edges = bbox_segments(points)
    bbox_starts = np.array([corners[start] for start, _ in edges])
    bbox_ends = np.array([corners[end] for _, end in edges])
    bx, by, bz = trace_lines(bbox_starts, bbox_ends)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=points[point_idx, 0],
            y=points[point_idx, 1],
            z=points[point_idx, 2],
            mode="markers",
            marker={
                "size": 3,
                "color": scores[point_idx],
                "colorscale": "RdBu",
                "cmin": -1.0,
                "cmax": 1.0,
                "colorbar": {"title": "outward"},
            },
            name="model points",
        )
    )
    fig.add_trace(go.Scatter3d(x=nx, y=ny, z=nz, mode="lines", line={"color": "royalblue", "width": 3}, name="normals"))
    fig.add_trace(go.Scatter3d(x=bx, y=by, z=bz, mode="lines", line={"color": "black", "width": 4}, name="bbox"))

    extent = points.max(axis=0) - points.min(axis=0)
    fig.update_layout(
        title=(
            f"{object_name}: {len(points)} points, "
            f"normal_outward={float(np.mean(scores > 0.0)):.3f}, "
            f"extent=({extent[0]:.4f}, {extent[1]:.4f}, {extent[2]:.4f}) m"
        ),
        scene={"aspectmode": "data", "xaxis_title": "X (m)", "yaxis_title": "Y (m)", "zaxis_title": "Z (m)"},
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
        showlegend=True,
    )
    html_path = output_dir / f"{object_name}_model_normals.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    return html_path


def write_index(output_dir, entries):
    rows = []
    for entry in entries:
        rows.append(
            "<tr>"
            f"<td>{html.escape(entry['object'])}</td>"
            f"<td>{entry['points']}</td>"
            f"<td>{entry['outward_ratio']:.3f}</td>"
            f"<td>{html.escape(entry['extent'])}</td>"
            f"<td><a href='{html.escape(entry['png'].name)}'>PNG</a></td>"
            f"<td>{entry['html_link']}</td>"
            "</tr>"
        )
    page = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ICP Model Visualization</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 24px; color: #222; }
    table { border-collapse: collapse; min-width: 760px; }
    th, td { border-bottom: 1px solid #ddd; padding: 8px 10px; text-align: left; }
    th { background: #f5f5f5; }
    code { background: #f5f5f5; padding: 2px 4px; }
  </style>
</head>
<body>
  <h1>ICP Model Visualization</h1>
  <p>Check that each cloud has an object-like shape, a tight bounding box, and mostly outward normals.</p>
  <table>
    <thead><tr><th>object</th><th>points</th><th>outward</th><th>extent xyz (m)</th><th>static</th><th>interactive</th></tr></thead>
    <tbody>
      %s
    </tbody>
  </table>
</body>
</html>
""" % "\n      ".join(rows)
    index_path = output_dir / "index.html"
    index_path.write_text(page, encoding="utf-8")
    return index_path


def write_overview(output_dir, entries):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None

    images = []
    tile_width = 520
    tile_height = 500
    label_height = 34
    for entry in entries:
        image = Image.open(entry["png"]).convert("RGB")
        image.thumbnail((tile_width, tile_height - label_height))
        tile = Image.new("RGB", (tile_width, tile_height), "white")
        x = (tile_width - image.width) // 2
        tile.paste(image, (x, label_height))
        draw = ImageDraw.Draw(tile)
        label = f"{entry['object']} | points={entry['points']} | outward={entry['outward_ratio']:.3f}"
        draw.text((12, 10), label, fill=(25, 25, 25))
        images.append(tile)

    if not images:
        return None

    columns = 2
    rows = (len(images) + columns - 1) // columns
    overview = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    for index, image in enumerate(images):
        row = index // columns
        col = index % columns
        overview.paste(image, (col * tile_width, row * tile_height))

    overview_path = output_dir / "all_models_overview.png"
    overview.save(overview_path)
    return overview_path


def main():
    args = parse_args()
    require_deps()
    model_dir = Path(args.model_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for object_name in discover_objects(model_dir, args.objects):
        cloud, metadata, _, _ = load_cloud(model_dir, object_name)
        points = np.asarray(cloud.points)
        normals = np.asarray(cloud.normals)
        if len(normals) != len(points):
            raise RuntimeError(f"{object_name} has no normal vectors. Run prepare_icp_models.py first.")

        scores = outward_scores(points, normals)
        png_path = save_png(object_name, points, normals, scores, metadata, output_dir, args)
        html_path = save_html(object_name, points, normals, scores, metadata, output_dir, args)
        extent = points.max(axis=0) - points.min(axis=0)
        html_link = f"<a href='{html.escape(html_path.name)}'>HTML</a>" if html_path else "plotly not installed"
        entries.append(
            {
                "object": object_name,
                "points": len(points),
                "outward_ratio": float(np.mean(scores > 0.0)),
                "extent": f"{extent[0]:.4f}, {extent[1]:.4f}, {extent[2]:.4f}",
                "png": png_path,
                "html": html_path,
                "html_link": html_link,
            }
        )
        print(f"{object_name}: png={png_path} html={html_path}")

    overview_path = write_overview(output_dir, entries)
    if overview_path:
        print(f"overview: {overview_path}")
    index_path = write_index(output_dir, entries)
    print(f"index: {index_path}")


if __name__ == "__main__":
    main()
