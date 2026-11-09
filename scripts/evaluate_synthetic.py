"""Evaluate locus-ortho on synthetic scenes with known tilt, azimuth, depth and normals.

    python scripts/synthetic.py --out examples/synthetic --scenes 8
    python scripts/evaluate_synthetic.py --scenes examples/synthetic --bin target/release/locus-ortho \
        --model models/depth_anything_v2_small.onnx
"""
import argparse
import json
import os
import subprocess
import tempfile

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

CONFIGS = {
    'normals (GT) + depth (GT)': ['--normals', '{n}', '--depth', '{d}'],
    'normals (GT) + depth (model)': ['--normals', '{n}'],
    'depth only (GT)': ['--depth', '{d}'],
    'depth only (model)': [],
}


def gray(p):
    return np.asarray(Image.open(p).convert('L'))


def run(binary, model, img, extra, level, out):
    cmd = [binary, img, '-o', out, '--level', str(level), '-m', model, '--report', os.path.join(out, 'r.json')] + extra
    subprocess.run(cmd, check=True, capture_output=True)
    return json.load(open(os.path.join(out, 'r.json')))[0], os.path.join(out, os.path.splitext(os.path.basename(img))[0] + '.png')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', default='examples/synthetic')
    ap.add_argument('--bin', default='target/release/locus-ortho')
    ap.add_argument('--model', default='models/depth_anything_v2_small.onnx')
    ap.add_argument('--out', default='docs/results_synthetic.json')
    args = ap.parse_args()
    truth = json.load(open(os.path.join(args.scenes, 'truth.json')))
    rows = {c: [] for c in CONFIGS}
    for name, tr in truth.items():
        base = os.path.join(args.scenes, name)
        img, nad = base + '_oblique.png', gray(base + '_nadir.png')
        ssim0 = structural_similarity(gray(img), nad, data_range=255)
        for cfg, extra in CONFIGS.items():
            extra = [e.format(n=base + '_normals_gt.png', d=base + '_depth_gt.png') for e in extra]
            r = {'scene': name, 'ssim_input': ssim0}
            for level in (1, 2):
                with tempfile.TemporaryDirectory() as tmp:
                    est, out = run(args.bin, args.model, img, extra, level, tmp)
                    r[f'ssim_l{level}'] = structural_similarity(gray(out), nad, data_range=255)
            r['tilt_err'] = abs(est['tilt_deg'] - tr['tilt_deg'])
            r['azimuth_err'] = abs((est['azimuth_deg'] - tr['image_azimuth_deg'] + 180) % 360 - 180)
            r['hole_frac'] = est['hole_frac']
            rows[cfg].append(r)
    summary = {}
    print('| Tilt / depth source | Tilt error (°) | Azimuth error (°) | SSIM input | SSIM level 1 | SSIM level 2 |')
    print('| --- | --- | --- | --- | --- | --- |')
    for cfg, rs in rows.items():
        m = {k: float(np.mean([r[k] for r in rs])) for k in ['tilt_err', 'azimuth_err', 'ssim_input', 'ssim_l1', 'ssim_l2', 'hole_frac']}
        summary[cfg] = m
        print(f"| {cfg} | {m['tilt_err']:.1f} | {m['azimuth_err']:.1f} | {m['ssim_input']:.3f} | {m['ssim_l1']:.3f} | {m['ssim_l2']:.3f} |")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({'summary': summary, 'per_scene': rows, 'n_scenes': len(truth)}, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
