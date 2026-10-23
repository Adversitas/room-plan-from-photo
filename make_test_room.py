"""Ray-cast a simple synthetic room (4m x 5m, 2.5m ceiling, a 0.8x1.0x0.75m table) for testing."""
import sys
import numpy as np
from PIL import Image

W, H, FOV = 800, 600, 70.0
cam = np.array([0.0, 1.5, 0.2])
tilt = np.radians(15)
f = (W / 2) / np.tan(np.radians(FOV) / 2)
u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
d = np.stack([(u - W / 2) / f, -(v - H / 2) / f, np.ones_like(u)], -1)
c, s = np.cos(tilt), np.sin(tilt)
d = d @ np.array([[1, 0, 0], [0, c, s], [0, -s, c]])  # pitch down
d /= np.linalg.norm(d, axis=-1, keepdims=True)

t_best = np.full((H, W), np.inf)
col = np.zeros((H, W, 3))

def plane(axis, val, color_fn, bounds):
    global t_best, col
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (val - cam[axis]) / d[..., axis]
    p = cam + t[..., None] * d
    ok = (t > 1e-4) & (t < t_best)
    for ax, (lo, hi) in bounds.items():
        ok &= (p[..., ax] >= lo - 1e-6) & (p[..., ax] <= hi + 1e-6)
    t_best = np.where(ok, t, t_best)
    col[ok] = color_fn(p[ok])

def checker(p):
    k = (np.floor(p[:, 0] / 0.5) + np.floor(p[:, 2] / 0.5)) % 2
    return np.where(k[:, None] == 0, [0.62, 0.45, 0.30], [0.55, 0.39, 0.25])

def wall(base):
    return lambda p: np.clip(np.array(base) * (0.85 + 0.1 * p[:, 1:2] / 2.5), 0, 1)

plane(1, 0.0, checker, {0: (-2, 2), 2: (0, 5)})
plane(1, 2.5, lambda p: np.tile([0.95, 0.95, 0.93], (len(p), 1)), {0: (-2, 2), 2: (0, 5)})
plane(0, -2.0, wall([0.85, 0.80, 0.70]), {1: (0, 2.5), 2: (0, 5)})
plane(0, 2.0, wall([0.80, 0.78, 0.68]), {1: (0, 2.5), 2: (0, 5)})
plane(2, 5.0, wall([0.75, 0.80, 0.85]), {0: (-2, 2), 1: (0, 2.5)})
# table: x 0.5..1.3, z 2..3, height 0.75
brown = lambda k: (lambda p: np.tile(np.array([0.40, 0.25, 0.15]) * k, (len(p), 1)))
plane(1, 0.75, brown(1.2), {0: (0.5, 1.3), 2: (2, 3)})
plane(2, 2.0, brown(0.9), {0: (0.5, 1.3), 1: (0, 0.75)})
plane(0, 0.5, brown(0.75), {1: (0, 0.75), 2: (2, 3)})

img = (col.clip(0, 1) * 255).astype(np.uint8)
Image.fromarray(img).save(sys.argv[1])
print("saved", sys.argv[1])
