"""
room_plan.py - turn a single photo of a room into an illustrated floor plan.

How it works:
  1. Grounding DINO finds the furniture in the photo (sofa, armchair, table, rug, ...).
  2. Depth Anything V2 estimates the distance of every pixel; the floor is found with RANSAC
     (same steps as room_topdown.py).
  3. Each piece of furniture is placed on the floor using the 3D points inside its box:
     position, size and rotation come from those points.
  4. Everything is drawn as a clean floor plan with furniture icons, walls and a 1 m scale bar.

Usage:
  python room_plan.py photo.jpg
  python room_plan.py photo.jpg --fov 70 --out plan.svg --debug

Notes:
  - A photo only shows part of the room: the walls are drawn where the camera saw them,
    and furniture backs (which the camera can't see) get typical sizes.
  - Sizes are estimates, typically within ~10-20%.
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patches, patheffects
from matplotlib.transforms import Affine2D
from PIL import Image, ImageDraw
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from room_topdown import MODELS, back_project, estimate_depth, fit_floor, valid_mask

DET_MODEL = "IDEA-Research/grounding-dino-base"
CLIP_MODEL = "openai/clip-vit-base-patch32"

# Second opinion on look-alike labels: CLIP compares each crop against these descriptions.
LOOKALIKES = [
    {"armchair": "a photo of an upholstered armchair with a backrest and armrests",
     "chair": "a photo of a simple wooden chair"},
    {"coffee table": "a photo of a low coffee table in front of a sofa",
     "side table": "a photo of a small side table", "table": "a photo of a cafe or dining table",
     "desk": "a photo of an office desk"},
]

# Furniture we look for. Longer names first so "coffee table" wins over "table".
KINDS = [
    "coffee table", "dining table", "side table", "floor lamp", "potted plant",
    "sofa", "armchair", "chair", "table", "desk", "sideboard", "cabinet", "bookshelf",
    "bed", "ottoman", "rug", "tv",
]
# Other names the detector understands, mapped to the kinds above.
ALIASES = {
    "couch": "sofa", "dresser": "sideboard", "console table": "sideboard", "tv stand": "sideboard",
    "end table": "side table", "pouf": "ottoman", "footstool": "ottoman", "houseplant": "potted plant",
    "cafe table": "table", "wooden chair": "chair", "dining chair": "chair", "lounge chair": "armchair",
}
# Detected so their pixels can be ignored (people hide furniture), never drawn.
PERSON = "person"
TABLES = {"coffee table", "dining table", "side table", "table", "desk"}
ROUND = {"ottoman", "side table", "potted plant", "floor lamp"}
# Typical (width, depth) in meters, used when the camera can't see an object's back.
TYPICAL = {
    "sofa": (2.0, 0.9), "armchair": (0.8, 0.8), "chair": (0.5, 0.5), "bed": (1.6, 2.0),
    "coffee table": (1.1, 0.6), "dining table": (1.6, 0.9), "table": (1.2, 0.8), "desk": (1.2, 0.6),
    "side table": (0.5, 0.5), "sideboard": (1.6, 0.45), "cabinet": (1.0, 0.45),
    "bookshelf": (0.9, 0.35), "ottoman": (0.5, 0.5), "potted plant": (0.45, 0.45),
    "floor lamp": (0.35, 0.35), "rug": (2.0, 1.5), "tv": (1.2, 0.1),
}
# Typical height of the surface seen from above (m), used when an object is hidden (e.g. by people).
TOP_HEIGHT = {
    "sofa": 0.6, "armchair": 0.6, "chair": 0.6, "bed": 0.55, "coffee table": 0.45,
    "dining table": 0.75, "table": 0.75, "desk": 0.75, "side table": 0.55, "sideboard": 0.8,
    "cabinet": 0.9, "bookshelf": 1.8, "ottoman": 0.42, "potted plant": 0.7, "floor lamp": 1.5,
    "rug": 0.0, "tv": 1.0,
}

# Palette
FLOOR = "#F6F1E9"
WALL = "#3B3632"
LINE = "#5E554D"
FABRIC = "#E9E3D8"
FABRIC_DARK = "#D6CDBF"
WOOD = "#C9A27C"
WOOD_DARK = "#A9825D"
DARK_WOOD = "#5B5550"
LEATHER = "#9C7A68"
RUG = "#DCD5E0"
GREEN = ["#7FA37A", "#93B68C", "#6C9168"]
LAMP = "#F3E2A9"


def detect(img, threshold):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(DET_MODEL)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(DET_MODEL).to(device).eval()
    names = sorted(KINDS + list(ALIASES) + [PERSON], key=len, reverse=True)
    inputs = processor(images=img, text=". ".join(names) + ".", return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # Tables are often half-covered by laptops, plates and people, so accept them a bit less sure.
    table_threshold = max(0.15, threshold - 0.08)
    res = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=table_threshold, text_threshold=0.25,
        target_sizes=[(img.height, img.width)],
    )[0]
    labels = res.get("text_labels", res["labels"])

    found = []
    for box, score, label in zip(res["boxes"].tolist(), res["scores"].tolist(), labels):
        name = next((k for k in names if k in str(label)), None)
        kind = ALIASES.get(name, name)
        if kind is None or score < (table_threshold if kind in TABLES else threshold):
            continue
        area = (box[2] - box[0]) * (box[3] - box[1]) / (img.width * img.height)
        if area > 0.6 and kind != "rug":  # a box covering most of the photo is "the room", not furniture
            continue
        found.append({"kind": kind, "score": score, "box": box})
    return dedupe(found)


def relabel(img, objects):
    """Ask CLIP to choose between look-alike labels (armchair vs chair, coffee vs cafe table, ...)."""
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(device).eval()
    for group in LOOKALIKES:
        members = [o for o in objects if o["kind"] in group]
        if not members:
            continue
        crops = [img.crop(tuple(int(v) for v in o["box"])) for o in members]
        inputs = processor(text=list(group.values()), images=crops, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            probs = model(**inputs).logits_per_image.softmax(dim=-1).cpu().numpy()
        kinds = list(group)
        for o, p in zip(members, probs):
            p[kinds.index(o["kind"])] *= 2  # the detector's label wins unless CLIP clearly disagrees
            p /= p.sum()
            if p.max() > 0.6:
                o["kind"] = kinds[int(p.argmax())]


def dedupe(found):
    """Drop duplicate detections of the same object (e.g. 'chair' and 'armchair' on one chair)."""
    def overlap(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
        return inter / max(smaller, 1e-9)

    layered = {"rug", PERSON}  # rugs lie under furniture and people sit on it: only dedupe like with like
    kept = []
    for f in sorted(found, key=lambda f: -f["score"]):
        if f["kind"] in layered:
            clash = any(k["kind"] == f["kind"] and overlap(k["box"], f["box"]) > 0.6 for k in kept)
        else:
            clash = any(k["kind"] not in layered and overlap(k["box"], f["box"]) > 0.7 for k in kept)
        if not clash:
            kept.append(f)
    return kept


def place(obj, xy, height, depth, valid, occluders=()):
    """Footprint on the floor (center, axes, width, depth) from the 3D points inside the box.

    Pixels inside `occluders` (boxes of objects standing in front of this one) are ignored,
    so a chair in front of a table doesn't drag the table towards the camera.
    """
    h, w = depth.shape
    valid = valid.copy()
    for b in occluders:
        valid[int(b[1]):int(np.ceil(b[3])), int(b[0]):int(np.ceil(b[2]))] = False
    x0, y0, x1, y1 = obj["box"]
    sx, sy = (x1 - x0) * 0.08, (y1 - y0) * 0.08  # shrink the box a little to avoid edges
    c0, c1 = int(max(0, x0 + sx)), int(min(w, x1 - sx))
    r0, r1 = int(max(0, y0 + sy)), int(min(h, y1 - sy))
    if c1 - c0 < 3 or r1 - r0 < 3:
        return None

    m = valid[r0:r1, c0:c1].copy()
    hh = height[r0:r1, c0:c1]
    m &= (hh < 0.06) if obj["kind"] == "rug" else ((hh > 0.04) & (hh < 2.2))
    d = depth[r0:r1, c0:c1][m]
    pts = xy[r0:r1, c0:c1][m]
    if len(pts) < 30:
        return None
    if obj["kind"] != "rug":
        # Keep the object itself, not the wall/floor behind it.
        near = np.percentile(d, 10)
        pts = pts[d < near + max(0.5, TYPICAL[obj["kind"]][1])]

    center = pts.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov((pts - center).T))
    back = vecs[:, 0]  # minor axis = object depth direction
    if back @ center < 0:  # make "back" point away from the camera
        back = -back
    side = np.array([back[1], -back[0]])

    def span(axis):
        p = (pts - center) @ axis
        lo, hi = np.percentile(p, [3, 97])
        return lo, hi

    lo_s, hi_s = span(side)
    lo_b, hi_b = span(back)
    center = center + side * (lo_s + hi_s) / 2 + back * (lo_b + hi_b) / 2
    width, dep = hi_s - lo_s, hi_b - lo_b

    kind = resolve_kind(obj["kind"], width)
    tw, td = TYPICAL[kind]
    if kind in ROUND:
        width = dep = max(width, min(dep, width), td * 0.7)
    elif dep < td * 0.8:
        # The back is hidden: extend away from the camera to a typical depth.
        center = center + back * (td - dep) / 2
        dep = td
    width = float(np.clip(width, tw * 0.4, tw * 1.3))
    dep = float(min(dep, td * 1.5)) if kind != "rug" else dep
    return {**obj, "kind": kind, "center": center, "angle": np.arctan2(side[1], side[0]),
            "w": width, "d": dep}


def resolve_kind(kind, width):
    """A table that measures this small is a side table, whatever it was called."""
    if kind in ("coffee table", "table", "dining table") and width < 0.6:
        return "side table"
    return kind


def place_overhead(obj, height, valid, to_floor):
    """Footprint for a photo taken from above: the object's box in the photo is projected
    straight onto the floor, at the height of the object's top (so tall things don't drift).
    Works even when the object is mostly hidden by people, using a typical height."""
    x0, y0, x1, y1 = obj["box"]
    r0, r1, c0, c1 = int(y0), int(np.ceil(y1)), int(x0), int(np.ceil(x1))
    hh = height[r0:r1, c0:c1][valid[r0:r1, c0:c1]]
    hh = hh[(hh > 0.03) & (hh < 2.2)]
    kind = obj["kind"]
    if kind == "rug":
        top = 0.0
    elif len(hh) >= 30:
        top = float(np.percentile(hh, 75))
    else:
        top = TOP_HEIGHT[kind]
    # Project the box corners: top-left, top-right, bottom-right, bottom-left.
    quad = to_floor(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float), top)
    tl, tr, br, bl = quad
    width = (np.linalg.norm(br - bl) + np.linalg.norm(tr - tl)) / 2
    dep = (np.linalg.norm(tl - bl) + np.linalg.norm(tr - br)) / 2
    side = (br - bl) / max(np.linalg.norm(br - bl), 1e-9)
    kind = resolve_kind(kind, width)
    tw, td = TYPICAL[kind]
    if kind in ROUND:
        width = dep = float(np.clip(min(width, dep), td * 0.6, td * 1.6))
    elif kind != "rug":
        # A box around a turned object is bigger than the object: keep sizes plausible.
        width = float(np.clip(width, tw * 0.4, tw * 1.5))
        dep = float(np.clip(dep, td * 0.4, td * 1.5))
    return {**obj, "kind": kind, "center": quad.mean(axis=0), "angle": np.arctan2(side[1], side[0]),
            "w": width, "d": dep}


def rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def straighten(placed):
    """Find the room's main direction from the furniture (rooms are mostly right angles),
    then rotate everything so it lines up with the page and snap each piece to it."""
    square = [p for p in placed if p["kind"] not in ROUND]
    if not square:
        return 0.0
    weights = np.array([p["w"] * p["d"] for p in square])
    angles = np.array([p["angle"] for p in square])
    theta = np.angle(np.sum(weights * np.exp(4j * angles))) / 4  # average direction, modulo 90 degrees
    R = rot(-theta)
    for p in placed:
        p["center"] = R @ p["center"]
        if p["kind"] not in ROUND:
            p["angle"] = np.round((p["angle"] - theta) / (np.pi / 2)) * (np.pi / 2)
    return theta


def half_extent(p):
    if p["kind"] in ROUND:
        return p["w"] / 2, p["w"] / 2
    c, s = abs(np.cos(p["angle"])), abs(np.sin(p["angle"]))
    return (p["w"] * c + p["d"] * s) / 2, (p["w"] * s + p["d"] * c) / 2


def separate(placed, gap=0.05, rounds=60):
    """Nudge overlapping furniture apart (depth estimates are rough; real furniture doesn't overlap).
    Smaller pieces move more than big ones. Rugs are left alone since things stand on them."""
    items = [p for p in placed if p["kind"] != "rug"]
    for _ in range(rounds):
        moved = False
        for i, a in enumerate(items):
            for b in items[i + 1:]:
                (ax_, ay), (bx, by) = half_extent(a), half_extent(b)
                delta = b["center"] - a["center"]
                ox = ax_ + bx + gap - abs(delta[0])
                oy = ay + by + gap - abs(delta[1])
                if ox <= 0 or oy <= 0:
                    continue
                area_a, area_b = a["w"] * a["d"], b["w"] * b["d"]
                share_a = area_b / (area_a + area_b)  # the smaller one moves more
                axis = 0 if ox < oy else 1
                amount = (ox if axis == 0 else oy) * (1 if delta[axis] >= 0 else -1)
                a["center"][axis] -= amount * share_a
                b["center"][axis] += amount * (1 - share_a)
                moved = True
        if not moved:
            break


# ---------- drawing ----------

def rrect(ax, tr, x, y, w, h, r, fc, ec=LINE, lw=1.0, z=3, **kw):
    r = max(0.0, min(r, w / 2, h / 2))
    ax.add_patch(patches.FancyBboxPatch(
        (x + r, y + r), max(w - 2 * r, 1e-4), max(h - 2 * r, 1e-4),
        boxstyle=f"round,pad={r}", transform=tr, fc=fc, ec=ec, lw=lw, zorder=z, **kw))


def draw_sofa(ax, tr, w, d, cushion_w=0.75):
    arm = min(0.2, w * 0.12)
    back = min(0.22, d * 0.3)
    rrect(ax, tr, -w / 2, -d / 2, w, d, 0.08, FABRIC_DARK)
    rrect(ax, tr, -w / 2, d / 2 - back, w, back, 0.07, FABRIC_DARK)
    for sgn in (-1, 1):
        x = -w / 2 if sgn < 0 else w / 2 - arm
        rrect(ax, tr, x, -d / 2, arm, d - back * 0.5, 0.07, FABRIC_DARK, z=4)
    seat_w = w - 2 * arm
    n = max(1, int(round(seat_w / cushion_w)))
    for i in range(n):
        x = -w / 2 + arm + i * seat_w / n
        rrect(ax, tr, x + 0.02, -d / 2 + 0.04, seat_w / n - 0.04, d - back - 0.06, 0.06, FABRIC, z=4)


def draw_armchair(ax, tr, w, d):
    back = min(0.2, d * 0.28)
    arm = min(0.16, w * 0.2)
    rrect(ax, tr, -w / 2, -d / 2, w, d, 0.2, FABRIC_DARK)
    rrect(ax, tr, -w / 2 + arm, -d / 2 + 0.05, w - 2 * arm, d - back - 0.05, 0.12, FABRIC, z=4)


def draw_chair(ax, tr, w, d):
    rrect(ax, tr, -w / 2, -d / 2, w, d, 0.06, WOOD)
    rrect(ax, tr, -w / 2, d / 2 - 0.08, w, 0.08, 0.03, WOOD_DARK, z=4)


def draw_table(ax, tr, w, d, color=WOOD, edge=WOOD_DARK):
    rrect(ax, tr, -w / 2, -d / 2, w, d, 0.04, color, ec=edge, lw=1.2)
    inset = min(0.06, d * 0.12)
    rrect(ax, tr, -w / 2 + inset, -d / 2 + inset, w - 2 * inset, d - 2 * inset, 0.02, "none",
          ec=edge, lw=0.5, z=4)
    for f in (-0.15, 0.1):  # a couple of grain lines
        ax.plot([-w / 2 + inset * 2, w / 2 - inset * 2], [f * d, f * d], color=edge, lw=0.4,
                alpha=0.6, transform=tr, zorder=4)


def draw_storage(ax, tr, w, d):
    rrect(ax, tr, -w / 2, -d / 2, w, d, 0.02, DARK_WOOD, ec=WALL, lw=1.0)
    n = max(2, int(round(w / 0.45)))
    for i in range(1, n):
        x = -w / 2 + i * w / n
        ax.plot([x, x], [-d / 2 + 0.04, d / 2 - 0.04], color="#8C857E", lw=0.6, transform=tr, zorder=4)


def draw_bed(ax, tr, w, d):
    rrect(ax, tr, -w / 2, -d / 2, w, d, 0.05, "#FFFFFF")
    for sgn in (-1, 1):
        pw = w / 2 - 0.15
        rrect(ax, tr, sgn * (w / 4) - pw / 2, d / 2 - 0.45, pw, 0.32, 0.08, FABRIC, z=4)
    rrect(ax, tr, -w / 2, -d / 2, w, d * 0.62, 0.05, "#C9D6E3", z=5)
    ax.plot([-w / 2, w / 2], [-d / 2 + d * 0.62 - 0.12] * 2, color=LINE, lw=0.6, transform=tr, zorder=6)


def draw_tv(ax, tr, w, d):
    rrect(ax, tr, -w / 2, -0.04, w, 0.08, 0.01, "#2E2E2E", ec="#2E2E2E")


def draw_rug(ax, tr, w, d):
    rrect(ax, tr, -w / 2, -d / 2, w, d, 0.02, RUG, ec="#B9AFC0", lw=0.8, z=1.5)
    i = min(0.15, w * 0.08, d * 0.08)
    rrect(ax, tr, -w / 2 + i, -d / 2 + i, w - 2 * i, d - 2 * i, 0.0, "none", ec="#B9AFC0", lw=0.6,
          z=1.6, ls=(0, (3, 2)))


def draw_round(ax, kind, center, r):
    cx, cy = center
    if kind == "ottoman":
        ax.add_patch(patches.Circle((cx, cy), r, fc=LEATHER, ec="#6E5547", lw=1.0, zorder=3))
        ax.add_patch(patches.Circle((cx, cy), r * 0.72, fc="none", ec="#B8998A", lw=0.7, zorder=4))
    elif kind == "side table":
        ax.add_patch(patches.Circle((cx, cy), r, fc=WOOD, ec=WOOD_DARK, lw=1.2, zorder=3))
        ax.add_patch(patches.Circle((cx, cy), r * 0.7, fc="none", ec=WOOD_DARK, lw=0.5, zorder=4))
    elif kind == "floor lamp":
        ax.add_patch(patches.Circle((cx, cy), r, fc=LAMP, ec="#C9B26A", lw=1.0, zorder=6))
        ax.plot([cx - r * 0.6, cx + r * 0.6], [cy, cy], color="#C9B26A", lw=0.7, zorder=7)
        ax.plot([cx, cx], [cy - r * 0.6, cy + r * 0.6], color="#C9B26A", lw=0.7, zorder=7)
    elif kind == "potted plant":
        for i in range(7):
            a = i * 2 * np.pi / 7
            leaf = patches.Ellipse((cx + np.cos(a) * r * 0.5, cy + np.sin(a) * r * 0.5), r * 1.0, r * 0.45,
                                   angle=np.degrees(a), fc=GREEN[i % 3], ec="#4F6E4B", lw=0.6, zorder=6)
            ax.add_patch(leaf)
        ax.add_patch(patches.Circle((cx, cy), r * 0.18, fc="#4F6E4B", ec="none", zorder=7))


DRAW = {
    "sofa": draw_sofa, "armchair": draw_armchair, "chair": draw_chair, "bed": draw_bed,
    "coffee table": draw_table, "dining table": draw_table, "table": draw_table, "desk": draw_table,
    "sideboard": draw_storage, "cabinet": draw_storage, "bookshelf": draw_storage,
    "tv": draw_tv, "rug": draw_rug,
}


def render(objects, room, out, labels=True, fov=70, view=0.0, camera_note="photo taken here"):
    left, right, near, far = room
    pad = 0.6
    fig_w = 9
    fig_h = fig_w * (far - near + 2 * pad) / (right - left + 2 * pad)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_xlim(left - pad, right + pad)
    ax.set_ylim(near - pad, far + pad)
    ax.set_aspect("equal")
    ax.axis("off")

    # Floor with a subtle plank pattern
    ax.add_patch(patches.Rectangle((left, near), right - left, far - near, fc=FLOOR, ec="none", zorder=0))
    for x in np.arange(left + 0.2, right, 0.2):
        ax.plot([x, x], [near, far], color="#ECE4D8", lw=0.5, zorder=0.5)

    # Walls the camera saw (left, back, right); the near side is open (behind the photographer).
    t = 0.12
    for xy, w, h in [((left - t, near), t, far - near + t), ((left - t, far), right - left + 2 * t, t),
                     ((right, near), t, far - near + t)]:
        ax.add_patch(patches.Rectangle(xy, w, h, fc=WALL, ec="none", zorder=8))
    ax.plot([left, right], [near, near], color="#B5ADA3", lw=1, ls=(0, (5, 4)), zorder=8)

    for obj in sorted(objects, key=lambda o: (o["kind"] != "rug", o["kind"] == "floor lamp")):
        cx, cy = obj["center"]
        if obj["kind"] in ROUND:
            draw_round(ax, obj["kind"], obj["center"], obj["w"] / 2)
        else:
            tr = Affine2D().rotate(obj["angle"]).translate(cx, cy) + ax.transData
            DRAW[obj["kind"]](ax, tr, obj["w"], obj["d"])
        if labels:
            ax.text(cx, cy, obj["kind"], ha="center", va="center", fontsize=7, color="#3F3933",
                    zorder=20, path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white")])

    # Camera position and view cone
    half = np.radians(fov) / 2
    cone_len = min(1.2, (far - near) * 0.3)
    cone = np.array([(0, 0), (-cone_len * np.tan(half), cone_len), (cone_len * np.tan(half), cone_len)])
    ax.add_patch(patches.Polygon(cone @ rot(view).T, fc="#E37B5B", alpha=0.15, ec="none", zorder=9))
    ax.add_patch(patches.Circle((0, 0), 0.1, fc="#E37B5B", ec="white", lw=1.2, zorder=10))
    ax.text(0, -0.22, camera_note, ha="center", va="top", fontsize=7, color="#B4583C", zorder=10)

    # 1 m scale bar
    sx, sy = left, near - pad * 0.55
    ax.plot([sx, sx + 1], [sy, sy], color=WALL, lw=2, solid_capstyle="butt")
    for x in (sx, sx + 1):
        ax.plot([x, x], [sy - 0.06, sy + 0.06], color=WALL, lw=1)
    ax.text(sx + 0.5, sy + 0.1, "1 m", ha="center", va="bottom", fontsize=7, color=WALL)

    fig.tight_layout(pad=0.3)
    retry_save(lambda: fig.savefig(out, dpi=200, facecolor="white"))
    plt.close(fig)


def retry_save(save, tries=5):
    """Windows sometimes holds a just-viewed image for a moment (previews, antivirus): retry briefly."""
    for i in range(tries):
        try:
            return save()
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(0.5)


def save_debug(img, objects, path):
    im = img.copy()
    draw = ImageDraw.Draw(im)
    for o in objects:
        draw.rectangle(o["box"], outline=(227, 123, 91), width=3)
        draw.text((o["box"][0] + 4, o["box"][1] + 4), f'{o["kind"]} {o["score"]:.2f}', fill=(227, 123, 91))
    retry_save(lambda: im.save(path))


def main():
    ap = argparse.ArgumentParser(description="Illustrated floor plan from a single room photo.")
    ap.add_argument("image", type=Path)
    ap.add_argument("--fov", type=float, default=70.0, help="horizontal field of view in degrees")
    ap.add_argument("--model", choices=MODELS, default="large", help="depth model size")
    ap.add_argument("--threshold", type=float, default=0.3, help="furniture detection confidence (0-1)")
    ap.add_argument("--no-labels", action="store_true", help="don't write furniture names on the plan")
    ap.add_argument("--out", type=Path, help="output .png or .svg (default: <image>_plan.png)")
    ap.add_argument("--debug", action="store_true", help="also save the photo with detected furniture boxes")
    ap.add_argument("--view", choices=["auto", "eye", "overhead"], default="auto",
                    help="eye-level or from-above photo (auto decides from the camera tilt)")
    args = ap.parse_args()

    img = Image.open(args.image).convert("RGB")
    img.thumbnail((1024, 1024))

    print("Finding furniture ...")
    found = detect(img, args.threshold)
    objects = [o for o in found if o["kind"] != PERSON]
    people = [o for o in found if o["kind"] == PERSON]
    relabel(img, objects)
    print("  found:", ", ".join(o["kind"] for o in objects) or "nothing",
          f"(+ {len(people)} people, ignored)" if people else "")

    print(f"Estimating depth with {MODELS[args.model]} ...")
    depth = estimate_depth(img, MODELS[args.model])
    pts = back_project(depth, args.fov)
    valid = valid_mask(depth, 12.0)
    rows = np.repeat(np.arange(img.height), img.width)
    flat = pts.reshape(-1, 3)
    n, d = fit_floor(flat[valid.ravel()], rows[valid.ravel()], img.height)
    tilt = np.degrees(np.arcsin(np.clip(-n[2], -1, 1)))  # how far the camera looks down
    overhead = args.view == "overhead" or (args.view == "auto" and tilt > 40)
    print(f"Estimated camera height: {d:.2f} m, looking down {tilt:.0f} degrees"
          f" -> {'overhead' if overhead else 'eye-level'} photo")

    up = n
    fwd = np.array([0.0, 0.0, 1.0]) - n[2] * n
    if np.linalg.norm(fwd) < 0.1:  # looking straight down: "forward" is the top of the photo
        fwd = np.array([0.0, -1.0, 0.0]) - (-n[1]) * n
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    height = pts @ up + d
    xy = np.stack([pts @ right, pts @ fwd], axis=-1)

    # People sit on and in front of furniture: ignore their pixels when measuring it.
    free = valid.copy()
    for o in people:
        x0, y0, x1, y1 = o["box"]
        free[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))] = False

    f = (img.width / 2) / np.tan(np.radians(args.fov) / 2)

    def to_floor(px, h):
        """Where pixel rays hit the horizontal plane `h` meters above the floor, in plan coordinates."""
        rays = np.stack([(px[:, 0] - img.width / 2) / f, (px[:, 1] - img.height / 2) / f,
                         np.ones(len(px))], axis=1)
        p = rays * ((h - d) / (rays @ n))[:, None]
        return np.stack([p @ right, p @ fwd], axis=1)

    placed = []
    for o in objects:
        if overhead:
            p = place_overhead(o, height, free, to_floor)
        else:
            # Furniture stands on the floor, so a box whose bottom edge is lower in the photo is closer.
            occluders = [] if o["kind"] == "rug" else [
                q["box"] for q in objects
                if q is not o and q["kind"] != "rug" and q["box"][3] > o["box"][3] + 0.02 * img.height
            ]
            p = place(o, xy, height, depth, free, occluders)
        if p is not None:
            placed.append(p)

    # Room outline from the wall band (above most furniture, below the ceiling).
    # Inner percentiles keep things seen through windows/doors from pushing the walls out.
    theta = straighten(placed)
    separate(placed)
    band = valid & (height > 1.0) & (height < 2.2)
    sxy = (xy[band] if band.sum() > 500 else xy[valid]) @ rot(-theta).T
    left, right_x = np.percentile(sxy[:, 0], [3, 97])
    far = np.percentile(sxy[:, 1], 80)
    near = -0.4
    if placed:
        # Keep walls close to the furniture: depth estimates through windows/doors run long.
        def edges(p):
            hx, hy = half_extent(p)
            return p["center"][0] - hx, p["center"][0] + hx, p["center"][1] + hy, p["center"][1] - hy

        e = np.array([edges(p) for p in placed])
        left = np.clip(left, e[:, 0].min() - 0.6, e[:, 0].min() - 0.05)
        right_x = np.clip(right_x, e[:, 1].max() + 0.05, e[:, 1].max() + 0.8)
        far = np.clip(far, e[:, 2].max() + 0.05, e[:, 2].max() + 0.5)
        near = min(near, e[:, 3].min() - 0.3)  # from above, furniture can be right under the camera
    room = (min(left, -0.8), max(right_x, 0.8), near, far)

    out = args.out or args.image.with_name(args.image.stem + "_plan.png")
    render(placed, room, out, labels=not args.no_labels, fov=args.fov, view=-theta,
           camera_note="photo taken from above here" if overhead else "photo taken here")
    print(f"Saved {out}")
    if args.debug:
        dbg = args.image.with_name(args.image.stem + "_detections.png")
        save_debug(img, objects, dbg)
        print(f"Saved {dbg}")


if __name__ == "__main__":
    main()
