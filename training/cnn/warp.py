"""Perspective warp (inverse homography + bilinear), pure Python / NumPy.

Given the source RGB image and four board corners, produces a square board
image where every grid cell is `cell` pixels wide, with a `cell`-pixel margin
on each side. Only the value channel ``V`` (max of R,G,B, 0..255) is produced —
the CNN consumes it directly to cut patches. No hand-crafted features are used.

The warp maps the playable-grid quadrilateral to a square of side
(N-1)*cell with a `cell`-pixel margin on each side.
"""
from __future__ import annotations

import numpy as np


def warp_perspective(rgb, quad, n_lines, cell):
    """Warp the board region into a square image.

    Returns dict with ``V`` (uint8 [Wd,Wd], max of R/G/B) and ``Wd`` (side
    length in pixels), plus ``cell``.
    """
    sH, sW = rgb.shape[:2]
    margin = cell
    S = cell * (n_lines - 1)
    Wd = S + 2 * margin

    # Inverse homography: unit board (u,v) -> source raster (x,y)
    dst = np.asarray([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    A_mat, b_vec = [], []
    for (x, y), (u, v) in zip(quad.astype(float), dst):
        A_mat.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); b_vec.append(u)
        A_mat.append([0, 0, 0, x, y, 1, -v * x, -v * y]); b_vec.append(v)
    sol = np.linalg.solve(np.asarray(A_mat, float), np.asarray(b_vec, float))
    H = np.array([[sol[0], sol[1], sol[2]],
                  [sol[3], sol[4], sol[5]],
                  [sol[6], sol[7], 1.0]])
    Hi = np.linalg.inv(H)  # unit -> raster

    # Destination pixel grid -> unit board coordinates
    xs = np.arange(Wd, dtype=np.float64)
    ys = np.arange(Wd, dtype=np.float64)
    gx, gy = np.meshgrid(xs, ys)  # gx=col(x), gy=row(y)
    u = (gx - margin) / S
    v = (gy - margin) / S

    # Unit -> source raster
    sx = Hi[0, 0] * u + Hi[0, 1] * v + Hi[0, 2]
    sy = Hi[1, 0] * u + Hi[1, 1] * v + Hi[1, 2]
    sw = Hi[2, 0] * u + Hi[2, 1] * v + Hi[2, 2]
    sx = sx / sw
    sy = sy / sw

    # Bilinear interpolation — vectorised over all destination pixels.
    x0 = np.floor(sx).astype(int)
    y0 = np.floor(sy).astype(int)
    fx = sx - x0
    fy = sy - y0
    x0c = np.clip(x0, 0, sW - 1)
    x1c = np.clip(x0 + 1, 0, sW - 1)
    y0c = np.clip(y0, 0, sH - 1)
    y1c = np.clip(y0 + 1, 0, sH - 1)

    src = (rgb * 255).clip(0, 255).astype(np.uint8)
    src_r = src[:, :, 0]
    src_g = src[:, :, 1]
    src_b = src[:, :, 2]
    w00 = (1 - fx) * (1 - fy)
    w10 = fx * (1 - fy)
    w01 = (1 - fx) * fy
    w11 = fx * fy
    R  = src_r[y0c, x0c] * w00 + src_r[y0c, x1c] * w10 + src_r[y1c, x0c] * w01 + src_r[y1c, x1c] * w11
    Gv = src_g[y0c, x0c] * w00 + src_g[y0c, x1c] * w10 + src_g[y1c, x0c] * w01 + src_g[y1c, x1c] * w11
    Bv = src_b[y0c, x0c] * w00 + src_b[y0c, x1c] * w10 + src_b[y1c, x0c] * w01 + src_b[y1c, x1c] * w11

    V = np.maximum(np.maximum(R, Gv), Bv).astype(np.uint8)
    return {"V": V, "cell": cell, "Wd": Wd}
