"""
occupancy.py - which seats are free? Check a new photo against a plan made by room_plan.py.

Take the new photo from the same spot as the original. Small shifts are fine: the two photos
are lined up automatically.

How it works:
  1. The new photo is aligned to the original with ORB feature matching + a homography,
     so every seat's box from the original lands in the right place in the new photo.
  2. Grounding DINO finds the people in the new photo.
  3. Each person is assigned to at most one seat: the one their hips (or, from above, their
     middle) fall into and that their box covers most. Sofas hold several people.
  4. The plan is redrawn with free seats green, taken seats red, and part-filled sofas amber.
     People who aren't on a known seat (standing, or on a seat hidden in the first photo) are blue dots.

Tip: take the first photo (the one room_plan.py uses) when the room is empty, so every seat is visible.

Usage:
  python room_plan.py cafe.jpg                       # once: saves cafe_plan.png + cafe_layout.json
  python occupancy.py cafe_layout.json cafe_now.jpg  # saves cafe_now_occupancy.png + cafe_now_seats.png
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from room_plan import PERSON, detect, render, retry_save

SEAT_KINDS = {"sofa", "armchair", "chair", "ottoman"}


def capacity(obj):
    if obj["kind"] == "sofa":
        return int(np.clip(round(obj["w"] / 0.7), 2, 4))
    return 1


def align(ref, new):
    """Homography mapping pixels of the original photo onto the new one, or None if they don't match."""
    g1 = cv2.cvtColor(np.asarray(ref), cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(np.asarray(new), cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(5000)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    if len(matches) < 30:
        return None
    src = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None or inliers.sum() < 25:
        return None
    return H


def warp_box(box, H):
    x0, y0, x1, y1 = box
    corners = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]).reshape(-1, 1, 2)
    p = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    return [p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()]


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection(a, b):
    return area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def assign(seats, people, overhead):
    """Give each person at most one seat. Returns the people who aren't sitting on a known seat."""
    candidates = []
    for pi, p in enumerate(people):
        x0, y0, x1, y1 = p["box"]
        # Where someone sits: their hips at eye level, their middle when seen from above.
        hx, hy = (x0 + x1) / 2, (y0 + y1) / 2 if overhead else y0 + 0.65 * (y1 - y0)
        for si, s in enumerate(seats):
            b = s["photo_box"]
            grow = 0.25 if overhead else 0.1  # from above, people lean over tables
            gx, gy = (b[2] - b[0]) * grow, (b[3] - b[1]) * grow
            inside = b[0] - gx <= hx <= b[2] + gx and b[1] - gy <= hy <= b[3] + gy
            cover = intersection(p["box"], b) / max(area(b), 1e-9)
            if inside or cover > 0.4:
                candidates.append((cover + inside, pi, si))

    seated = set()
    for _, pi, si in sorted(candidates, reverse=True):
        if pi in seated or seats[si]["occupied"] >= seats[si]["seats"]:
            continue
        seats[si]["occupied"] += 1
        seated.add(pi)
    return [p for i, p in enumerate(people) if i not in seated]


def save_photo_overlay(img, seats, people, path):
    im = img.copy()
    draw = ImageDraw.Draw(im)
    colors = {"free": (91, 174, 106), "taken": (217, 83, 79), "partly": (232, 163, 61)}
    for p in people:
        draw.rectangle(p["box"], outline=(70, 130, 220), width=2)
    for s in seats:
        state = "free" if s["occupied"] == 0 else "taken" if s["occupied"] >= s["seats"] else "partly"
        draw.rectangle(s["photo_box"], outline=colors[state], width=4)
        label = f'{s["kind"]} {s["occupied"]}/{s["seats"]}' if s["seats"] > 1 else state
        draw.text((s["photo_box"][0] + 5, s["photo_box"][1] + 4), label, fill=colors[state])
    retry_save(lambda: im.save(path))


def main():
    ap = argparse.ArgumentParser(description="Mark seats free/taken in a new photo, using a saved room plan.")
    ap.add_argument("layout", type=Path, help="the _layout.json saved by room_plan.py")
    ap.add_argument("image", type=Path, help="new photo, taken from the same spot")
    ap.add_argument("--threshold", type=float, default=0.25, help="person detection confidence (0-1)")
    ap.add_argument("--out", type=Path, help="output plan (default: <image>_occupancy.png)")
    args = ap.parse_args()

    layout = json.loads(args.layout.read_text())
    img = Image.open(args.image).convert("RGB")
    img.thumbnail((1024, 1024))

    # Line the new photo up with the original so the saved seat boxes land in the right place.
    ref_path = args.layout.parent / layout["image"]
    H = None
    if ref_path.exists():
        ref = Image.open(ref_path).convert("RGB")
        ref.thumbnail((1024, 1024))
        H = align(ref, img)
    if H is None:
        print("Could not line the photos up; assuming the camera didn't move.")
        sx, sy = img.width / layout["size"][0], img.height / layout["size"][1]
        H = np.diag([sx, sy, 1.0])
    else:
        shift = np.hypot(H[0, 2], H[1, 2])
        print(f"Photos lined up (camera moved about {shift:.0f} px).")

    print("Finding people ...")
    people = [o for o in detect(img, args.threshold) if o["kind"] == PERSON]
    print(f"  found {len(people)} people")

    objects = layout["objects"]
    for o in objects:
        o["center"] = np.array(o["center"])
    seats = [o for o in objects if o["kind"] in SEAT_KINDS]
    for s in seats:
        s["photo_box"] = warp_box(s["box"], H)
        s["seats"] = capacity(s)
        s["occupied"] = 0
    others = assign(seats, people, layout["overhead"])

    total = sum(s["seats"] for s in seats)
    taken = sum(s["occupied"] for s in seats)
    summary = f"{total - taken} of {total} seats free"
    if others:
        summary += f"  ·  {len(others)} other people (blue)"
    print(summary)

    # People not on a known seat become dots: back to the original photo's pixels, then onto the plan.
    # From above, use their middle at seat height; at eye level, their feet on the floor.
    key = "seat_to_plan" if layout["overhead"] else "floor_to_plan"
    dots = []
    if others and layout.get(key):
        pts = np.float32([[(p["box"][0] + p["box"][2]) / 2,
                           (p["box"][1] + p["box"][3]) / 2 if layout["overhead"] else p["box"][3]]
                          for p in others]).reshape(-1, 1, 2)
        in_ref = cv2.perspectiveTransform(pts, np.linalg.inv(H))
        dots = cv2.perspectiveTransform(in_ref, np.array(layout[key])).reshape(-1, 2).tolist()
    for s in seats:
        print(f'  {s["kind"]:<9} {s["occupied"]}/{s["seats"]}')

    out = args.out or args.image.with_name(args.image.stem + "_occupancy.png")
    render(objects, layout["room"], out, labels=False, fov=layout["fov"], view=layout["view"],
           camera_note=layout["camera_note"], title=summary, people=dots)
    print(f"Saved {out}")
    overlay = args.image.with_name(args.image.stem + "_seats.png")
    save_photo_overlay(img, seats, people, overlay)
    print(f"Saved {overlay}")


if __name__ == "__main__":
    main()
