"""Figures for the README from the synthetic scenes.

    python scripts/figures.py --scenes examples/synthetic
"""
import argparse
import os
import subprocess
import tempfile

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def run(binary, img, level, extra, out):
    subprocess.run([binary, img, '-o', out, '--level', str(level)] + extra, check=True, capture_output=True)
    return np.asarray(Image.open(os.path.join(out, os.path.splitext(os.path.basename(img))[0] + '.png')).convert('RGB'))


def labels_rgb(lab):
    rng = np.random.default_rng(0)
    pal = rng.uniform(0.15, 0.95, (int(lab.max()) + 1, 3))
    pal[0] = 0
    return pal[lab]


def show(ax, im, title):
    im = np.asarray(im)
    ax.imshow(im, cmap='magma' if im.ndim == 2 else None)
    ax.set_title(title, fontsize=9)
    ax.axis('off')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', default='examples/synthetic')
    ap.add_argument('--bin', default='target/release/locus-ortho')
    ap.add_argument('--out', default='docs/figures')
    ap.add_argument('--pipeline', nargs='+', default=['scene0', 'scene6'])
    ap.add_argument('--depth-scene', default='scene0')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cols = ['oblique input', 'normals', 'depth', 'segments', 'level 1', 'level 2', 'nadir reference']
    fig, axs = plt.subplots(len(args.pipeline), len(cols), figsize=(2.2 * len(cols), 2.35 * len(args.pipeline)))
    for r, name in enumerate(args.pipeline):
        b = os.path.join(args.scenes, name)
        gt = ['--normals', b + '_normals_gt.png', '--depth', b + '_depth_gt.png', '--segments', b + '_segments_gt.png']
        with tempfile.TemporaryDirectory() as tmp:
            l1 = run(args.bin, b + '_oblique.png', 1, gt, tmp)
            l2 = run(args.bin, b + '_oblique.png', 2, gt, tmp)
        ims = [Image.open(b + '_oblique.png'), Image.open(b + '_normals_gt.png'),
               np.asarray(Image.open(b + '_depth_gt.png'), np.float32), labels_rgb(np.asarray(Image.open(b + '_segments_gt.png'))),
               l1, l2, Image.open(b + '_nadir.png')]
        ims[1] = np.asarray(ims[1].convert('RGB'))
        for c, im in enumerate(ims):
            show(axs[r, c], im, cols[c] if r == 0 else '')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'synthetic_pipeline.png'), dpi=110)
    plt.close()

    b = os.path.join(args.scenes, args.depth_scene)
    seg = ['--normals', b + '_normals_gt.png', '--segments', b + '_segments_gt.png']
    with tempfile.TemporaryDirectory() as tmp:
        true_d = run(args.bin, b + '_oblique.png', 2, seg + ['--depth', b + '_depth_gt.png'], tmp)
        est_d = run(args.bin, b + '_oblique.png', 2, seg, tmp)
    fig, axs = plt.subplots(1, 4, figsize=(10, 2.8))
    for ax, im, t in zip(axs, [Image.open(b + '_oblique.png'), true_d, est_d, Image.open(b + '_nadir.png')],
                         ['oblique input', 'level 2, true depth', 'level 2, Depth Anything V2 Small', 'nadir reference']):
        show(ax, im, t)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'depth_source.png'), dpi=110)
    plt.close()


if __name__ == '__main__':
    main()
