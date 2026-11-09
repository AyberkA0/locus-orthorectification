"""Render synthetic urban scenes under an orthographic camera.

For each scene the script writes an oblique view, the matching nadir view
(same heading), ground-truth depth, normals and segment labels for the oblique
view, and the true camera parameters, so that rectification can be checked
against ground truth. Scenes contain blocks of buildings and parks with trees.

    python scripts/synthetic.py --out examples/synthetic --scenes 8
"""
import argparse
import json
import os

import numpy as np
from PIL import Image

SIZE = 512
STEP = 0.5


def city(rng, extent):
    """Return the block and road size, buildings (x0, y0, x1, y1, h, roof_rgb, wall_rgb)
    and trees (x, y, crown radius, crown centre height, rgb)."""
    block, road = 136, 22
    buildings, trees = [], []
    for bx in range(-extent, extent, block):
        for by in range(-extent, extent, block):
            x, y = bx + road / 2 + 6, by + road / 2 + 6
            w = block - road - 12
            if rng.uniform() < 0.3:
                # park: trees on grass, kept clear of the sidewalks
                for _ in range(rng.integers(4, 10)):
                    r = rng.uniform(5, 11)
                    trees.append((bx + rng.uniform(road + 8 + r, block - r), by + rng.uniform(road + 8 + r, block - r),
                                  r, r + rng.uniform(3, 9), rng.uniform(0.6, 1.0) * np.array([0.20, 0.42, 0.13])))
                continue
            n = rng.integers(1, 4)
            cuts = np.sort(rng.uniform(0.25, 0.75, n - 1)) if n > 1 else []
            edges = [0, *cuts, 1]
            for a, b in zip(edges[:-1], edges[1:]):
                x0, x1 = x + a * w + 3, x + b * w - 3
                y0 = y + rng.uniform(0, 0.2) * w
                y1 = y + w - rng.uniform(0, 0.2) * w
                h = float(rng.choice([rng.uniform(8, 30), rng.uniform(30, 110)], p=[0.6, 0.4]))
                roof = rng.uniform(0.45, 0.85, 3) * np.array([1.0, 0.97, 0.93])
                wall = roof * rng.uniform(0.55, 0.75)
                buildings.append((x0, y0, x1, y1, h, roof, wall))
    return block, road, buildings, trees


def ground_color(X, Y, block, road, rng_noise):
    gx, gy = np.mod(X, block), np.mod(Y, block)
    on_road = (gx < road) | (gy < road)
    side = ((gx < road + 6) | (gy < road + 6)) & ~on_road
    col = np.empty(X.shape + (3,))
    col[...] = [0.33, 0.47, 0.26]
    col[side] = [0.70, 0.69, 0.66]
    col[on_road] = [0.25, 0.26, 0.28]
    dash = on_road & ((np.abs(gx - road / 2) < 0.8) & (np.mod(Y, 24) < 12) | (np.abs(gy - road / 2) < 0.8) & (np.mod(X, 24) < 12))
    col[dash] = [0.92, 0.88, 0.55]
    col *= 1 + 0.08 * rng_noise(X, Y)[..., None]
    return np.clip(col, 0, 1)


def noise_fn(seed, cell=6.0):
    """Isotropic value noise in -1..1: random values on a coarse grid, bilinearly interpolated."""
    rng = np.random.default_rng(seed)
    n = 512
    v = rng.uniform(-1, 1, (n, n))

    def f(X, Y):
        gx, gy = X / cell, Y / cell
        x0, y0 = np.floor(gx).astype(int), np.floor(gy).astype(int)
        fx, fy = gx - x0, gy - y0
        g = lambda i, j: v[np.mod(j, n), np.mod(i, n)]
        top = g(x0, y0) * (1 - fx) + g(x0 + 1, y0) * fx
        bot = g(x0, y0 + 1) * (1 - fx) + g(x0 + 1, y0 + 1) * fx
        return top * (1 - fy) + bot * fy
    return f


def ground_labels(X, Y, block, road):
    """One segment per block: road network, sidewalk ring, interior."""
    gx, gy = np.mod(X, block), np.mod(Y, block)
    on_road = (gx < road) | (gy < road)
    side = ((gx < road + 6) | (gy < road + 6)) & ~on_road
    cell = (np.floor(X / block) * 37 + np.floor(Y / block)).astype(int) % 4000
    return np.where(on_road, 1, np.where(side, 2 + 2 * cell, 3 + 2 * cell))


def render(buildings, trees, block, road, theta, phi, psi, noise, extent):
    """Splat ground, roof, facade and crown samples with a z-buffer.
    Returns rgb, depth (larger = closer), valid mask, normals and segment labels."""
    t, f = np.radians(theta), np.radians(phi)
    hx, hy = np.sin(f), np.cos(f)
    rx, ry = np.cos(f), -np.sin(f)

    pts, cols, nrm, ids = [], [], [], []
    g = np.arange(-extent, extent, STEP)
    X, Y = np.meshgrid(g, g)
    X, Y = X.ravel(), Y.ravel()
    pts.append(np.stack([X, Y, np.zeros_like(X)], 1))
    cols.append(ground_color(X, Y, block, road, noise))
    nrm.append(np.tile([0.0, 0.0, 1.0], (X.size, 1)))
    ids.append(ground_labels(X, Y, block, road))
    nid = 10000
    for x0, y0, x1, y1, h, roof, wall in buildings:
        xs, ys = np.arange(x0, x1, STEP), np.arange(y0, y1, STEP)
        RX, RY = np.meshgrid(xs, ys)
        pts.append(np.stack([RX.ravel(), RY.ravel(), np.full(RX.size, h)], 1))
        cols.append(np.tile(roof, (RX.size, 1)))
        nrm.append(np.tile([0.0, 0.0, 1.0], (RX.size, 1)))
        ids.append(np.full(RX.size, nid))
        nid += 1
        zs = np.arange(0, h, STEP)
        for (ax, ay, bx, by, nx, ny) in [(x0, y0, x1, y0, 0, -1), (x1, y0, x1, y1, 1, 0), (x1, y1, x0, y1, 0, 1), (x0, y1, x0, y0, -1, 0)]:
            if nx * hx + ny * hy >= 0:
                continue
            L = np.hypot(bx - ax, by - ay)
            s = np.arange(0, L, STEP)
            S, Z = np.meshgrid(s, zs)
            FX, FY = ax + (bx - ax) * S / L, ay + (by - ay) * S / L
            win = (np.mod(S, 9) > 3) & (np.mod(Z, 7) > 2.5) & (Z > 3)
            shade = 0.85 + 0.15 * (nx * 0.6 + ny * 0.8)
            c = np.tile(wall * shade, (S.size, 1))
            c[win.ravel()] *= 0.55
            pts.append(np.stack([FX.ravel(), FY.ravel(), Z.ravel()], 1))
            cols.append(c)
            nrm.append(np.tile([float(nx), float(ny), 0.0], (S.size, 1)))
            ids.append(np.full(S.size, nid))
            nid += 1
    for k, (cx, cy, r, hc, rgb) in enumerate(trees):
        # crown: a sphere with leafy texture, sampled uniformly in area
        n = int(4 * np.pi * r * r / STEP ** 2)
        rng = np.random.default_rng(k)
        z = rng.uniform(-1, 1, n)
        a = rng.uniform(0, 2 * np.pi, n)
        d = np.stack([np.sqrt(1 - z * z) * np.cos(a), np.sqrt(1 - z * z) * np.sin(a), z], 1)
        leaf = 0.75 + 0.25 * np.sin(1.7 * d[:, 0] * r + 3 * a) * np.cos(2.3 * d[:, 1] * r) + 0.15 * d[:, 2]
        pts.append(np.array([cx, cy, hc]) + r * d)
        cols.append(np.clip(rgb[None] * leaf[:, None], 0, 1))
        nrm.append(d)
        ids.append(np.full(n, nid))
        nid += 1
    P, C, N, L = np.concatenate(pts), np.concatenate(cols), np.concatenate(nrm), np.concatenate(ids)

    u = P[:, 0] * rx + P[:, 1] * ry
    a = P[:, 0] * hx + P[:, 1] * hy
    v = a * np.cos(t) + P[:, 2] * np.sin(t)
    w = -a * np.sin(t) + P[:, 2] * np.cos(t)
    c, s = np.cos(np.radians(psi)), np.sin(np.radians(psi))
    u, v = c * u - s * v, s * u + c * v
    na = N[:, 0] * hx + N[:, 1] * hy
    nx_, ny_ = N[:, 0] * rx + N[:, 1] * ry, na * np.cos(t) + N[:, 2] * np.sin(t)
    N = np.stack([c * nx_ - s * ny_, s * nx_ + c * ny_, -na * np.sin(t) + N[:, 2] * np.cos(t)], 1)
    col = np.round(u + SIZE / 2).astype(int)
    row = np.round(SIZE / 2 - v).astype(int)
    ok = (col >= 0) & (col < SIZE) & (row >= 0) & (row < SIZE)
    col, row, w, C, N, L = col[ok], row[ok], w[ok], C[ok], N[ok], L[ok]
    order = np.argsort(w)
    img = np.zeros((SIZE, SIZE, 3))
    dep = np.full((SIZE, SIZE), -np.inf)
    img[row[order], col[order]] = C[order]
    nmap = np.zeros((SIZE, SIZE, 3))
    nmap[row[order], col[order]] = N[order]
    lab = np.zeros((SIZE, SIZE), np.int64)
    lab[row[order], col[order]] = L[order]
    dep[row[order], col[order]] = w[order]
    valid = np.isfinite(dep)
    dep[~valid] = dep[valid].min()
    # renumber labels compactly for a 16-bit PNG
    _, lab = np.unique(lab, return_inverse=True)
    return img, dep, valid, nmap, (lab.reshape(SIZE, SIZE) + 1).astype(np.uint16)


def save_normals(n, path):
    """16-bit RGB PNG, v = (n + 1) / 2 · 65535, camera coordinates x right, y up, z towards the camera."""
    import cv2
    v = np.round((np.clip(n, -1, 1) + 1) / 2 * 65535).astype(np.uint16)
    cv2.imwrite(path, v[..., ::-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='examples/synthetic')
    ap.add_argument('--scenes', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    meta = {}
    for i in range(args.scenes):
        theta = float(rng.uniform(20, 45))
        phi = float(rng.uniform(-30, 30))
        psi = float(rng.uniform(-60, 60))
        extent = 560
        block, road, buildings, trees = city(rng, extent)
        noise = noise_fn(args.seed * 100 + i)
        name = f'scene{i}'
        obl, dep, valid, nmap, seg = render(buildings, trees, block, road, theta, phi, psi, noise, extent)
        nad = render(buildings, trees, block, road, 0.0, phi, psi, noise, extent)[0]
        Image.fromarray((obl * 255).astype(np.uint8)).save(f'{args.out}/{name}_oblique.png')
        Image.fromarray((nad * 255).astype(np.uint8)).save(f'{args.out}/{name}_nadir.png')
        d = (dep - dep.min()) / (dep.max() - dep.min())
        Image.fromarray((d * 65535).astype(np.uint16)).save(f'{args.out}/{name}_depth_gt.png')
        save_normals(nmap, f'{args.out}/{name}_normals_gt.png')
        Image.fromarray(seg).save(f'{args.out}/{name}_segments_gt.png')
        meta[name] = {'tilt_deg': theta, 'image_azimuth_deg': -psi, 'valid_frac': float(valid.mean())}
        print(name, meta[name], flush=True)
    json.dump(meta, open(f'{args.out}/truth.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
