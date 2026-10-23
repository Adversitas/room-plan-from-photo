"""
room_topdown.py - turn a single photo of a room into a top-down (bird's-eye) map.

How it works:
  1. Depth Anything V2 (metric, indoor) estimates the distance in meters of every pixel.
  2. Pixels are back-projected into a 3D point cloud using an assumed camera field of view.
  3. The floor plane is found with RANSAC, and the cloud is rotated so the floor is flat.
  4. Points are projected straight down onto the floor to make:
       - a colored top-down view (the photo "seen from above")
       - an occupancy map (floor = free, furniture/walls = occupied) with a meter grid

Usage:
  python room_topdown.py photo.jpg
  python room_topdown.py photo.jpg --fov 70 --ceiling 2.0 --ply room.ply

Notes:
  - Only what the camera sees can be mapped. Take the photo from a corner, camera level,
    with as much floor visible as possible. A wide-angle shot helps (set --fov to match).
  - Distances are estimates (typically within ~10-20%), not survey-grade measurements.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODELS = {
    "small": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
}


def estimate_depth(img, model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device).eval()
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        depth = model(**inputs).predicted_depth  # (1, h, w), meters
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1), size=(img.height, img.width), mode="bicubic", align_corners=False
    )
    return depth.squeeze().cpu().numpy().clip(0.05, None)


def back_project(depth, fov_deg):
    h, w = depth.shape
    f = (w / 2) / np.tan(np.radians(fov_deg) / 2)
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x = (u - w / 2) * depth / f
    y = (v - h / 2) * depth / f  # camera convention: y points down
    return np.stack([x, y, depth], axis=-1)


def valid_mask(depth, max_depth):
    # Drop "flying pixels" at depth discontinuities and anything absurdly far.
    gy, gx = np.gradient(depth)
    rel_grad = np.hypot(gx, gy) / depth
    return (rel_grad < 0.05) & (depth < max_depth)


def fit_floor(points, rows, img_h, iters=800, thresh=0.03):
    """RANSAC a horizontal-ish plane among points in the lower half of the image."""
    cand = points[rows > img_h * 0.5]
    rng = np.random.default_rng(0)
    if len(cand) > 50000:
        cand = cand[rng.choice(len(cand), 50000, replace=False)]

    best_count, best = 0, None
    for _ in range(iters):
        s = cand[rng.choice(len(cand), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        # Floor normal lies in the image's vertical plane: mostly "up" for a level camera,
        # tipping towards the lens when the camera looks down. Rejects side walls and facing walls.
        if abs(n[0]) > 0.5 or abs(n[1]) < 0.25:
            continue
        d = -n @ s[0]
        count = np.count_nonzero(np.abs(cand @ n + d) < thresh)
        if count > best_count:
            best_count, best = count, (n, d)

    if best is None or best_count < 0.05 * len(cand):
        # Fallback: assume a level camera and put the floor at the lowest visible points.
        print("Warning: no clear floor plane found, assuming a level camera.")
        n = np.array([0.0, -1.0, 0.0])
        return n, np.percentile(points[:, 1], 98)

    # Refine with least squares on the inliers.
    n, d = best
    inliers = cand[np.abs(cand @ n + d) < thresh]
    centroid = inliers.mean(axis=0)
    n = np.linalg.svd(inliers - centroid)[2][-1]
    d = -n @ centroid
    if d < 0:  # orient the normal so the camera is above the floor (positive height)
        n, d = -n, -d
    return n, d


def rasterize(xy, height, colors, cell, extent):
    """Top-down raster: each cell shows the color of its highest point."""
    xmin, xmax, ymin, ymax = extent
    nx = int(np.ceil((xmax - xmin) / cell))
    ny = int(np.ceil((ymax - ymin) / cell))
    ix = ((xy[:, 0] - xmin) / cell).astype(int).clip(0, nx - 1)
    iy = ((ymax - xy[:, 1]) / cell).astype(int).clip(0, ny - 1)  # forward is "up" in the image
    cell_id = iy * nx + ix

    order = np.lexsort((height, cell_id))
    cid_sorted = cell_id[order]
    last = np.r_[cid_sorted[1:] != cid_sorted[:-1], True]  # highest point per cell
    top = order[last]

    rgb = np.ones((ny * nx, 3))
    rgb[cell_id[top]] = colors[top]

    counts = np.bincount(cell_id, minlength=ny * nx)
    obstacle = np.bincount(cell_id, weights=(height > 0.08).astype(float), minlength=ny * nx)
    occ = np.full(ny * nx, 0.5)  # unknown = gray
    seen = counts > 0
    occ[seen] = 1.0  # free floor = white
    occ[obstacle >= 3] = 0.0  # occupied = black
    return rgb.reshape(ny, nx, 3), occ.reshape(ny, nx)


def save_ply(path, pts, colors):
    c = (colors * 255).astype(np.uint8)
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, col in zip(pts, c):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {col[0]} {col[1]} {col[2]}\n")


def main():
    ap = argparse.ArgumentParser(description="Top-down room map from a single photo.")
    ap.add_argument("image", type=Path)
    ap.add_argument("--fov", type=float, default=65.0,
                    help="horizontal field of view of the camera in degrees (phone main camera ~65-70, ultrawide ~110)")
    ap.add_argument("--ceiling", type=float, default=2.0,
                    help="ignore points higher than this above the floor (m), so the ceiling doesn't hide the room")
    ap.add_argument("--max-depth", type=float, default=10.0, help="ignore points farther than this (m)")
    ap.add_argument("--cell", type=float, default=0.03, help="top-down grid cell size (m)")
    ap.add_argument("--model", choices=MODELS, default="small", help="depth model size (larger = better, slower)")
    ap.add_argument("--out", type=Path, help="output PNG (default: <image>_topdown.png)")
    ap.add_argument("--ply", type=Path, help="also save the 3D point cloud (open in MeshLab/CloudCompare)")
    args = ap.parse_args()

    img = Image.open(args.image).convert("RGB")
    img.thumbnail((800, 800))  # keeps it fast; plenty of detail for a floor map
    colors = np.asarray(img, dtype=np.float64).reshape(-1, 3) / 255.0

    print(f"Estimating depth with {MODELS[args.model]} ...")
    depth = estimate_depth(img, MODELS[args.model])
    pts = back_project(depth, args.fov).reshape(-1, 3)
    rows = np.repeat(np.arange(img.height), img.width)
    keep = valid_mask(depth, args.max_depth).ravel()
    pts, colors, rows = pts[keep], colors[keep], rows[keep]

    n, d = fit_floor(pts, rows, img.height)
    print(f"Estimated camera height above floor: {d:.2f} m (if this is way off, try a different --fov)")

    # Floor-aligned frame: up = floor normal, forward = camera view direction flattened onto the floor.
    up = n
    fwd = np.array([0.0, 0.0, 1.0]) - n[2] * n
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    height = pts @ up + d
    xy = np.stack([pts @ right, pts @ fwd], axis=1)

    sel = (height > -0.2) & (height < args.ceiling)
    xy, height_s, colors_s = xy[sel], height[sel], colors[sel]

    lo = np.percentile(xy, 0.5, axis=0)
    hi = np.percentile(xy, 99.5, axis=0)
    extent = (min(lo[0], -0.5), max(hi[0], 0.5), min(lo[1], -0.3), hi[1])
    rgb, occ = rasterize(xy, height_s, colors_s, args.cell, extent)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img)
    axes[0].set_title("Photo")
    axes[0].axis("off")
    for ax, data, title, kw in [
        (axes[1], rgb, "Top-down view (colored)", {}),
        (axes[2], occ, "Occupancy (black = furniture/walls, white = floor, gray = unseen)",
         {"cmap": "gray", "vmin": 0, "vmax": 1}),
    ]:
        ax.imshow(data, extent=extent, interpolation="nearest", **kw)
        ax.plot(0, 0, "r^", markersize=12, label="camera")
        half = np.radians(args.fov) / 2
        r = extent[3]
        ax.plot([0, -r * np.tan(half)], [0, r], "r--", lw=0.8)
        ax.plot([0, r * np.tan(half)], [0, r], "r--", lw=0.8)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("meters (left/right)")
        ax.set_ylabel("meters (forward)")
        ax.set_title(title, fontsize=10)
        ax.legend(loc="lower right")
    fig.tight_layout()

    out = args.out or args.image.with_name(args.image.stem + "_topdown.png")
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")

    if args.ply:
        # Save in the floor-aligned frame (z up) so it opens upright in viewers.
        aligned = np.stack([pts @ right, pts @ fwd, height], axis=1)
        save_ply(args.ply, aligned, colors)
        print(f"Saved {args.ply}")


if __name__ == "__main__":
    main()
