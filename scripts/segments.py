"""Segment label maps for locus-ortho with SAM (ViT-B, automatic mask generation).

Writes one 16-bit grayscale PNG per input image: 0 = no segment, 1..n = segments.
Overlapping masks are painted largest first, so smaller segments stay on top.

    pip install torch transformers pillow numpy
    python scripts/segments.py images/*.jpg --out segments/
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image
from transformers import pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', nargs='+')
    ap.add_argument('--out', default='segments')
    ap.add_argument('--model', default='facebook/sam-vit-base')
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    gen = pipeline('mask-generation', model=args.model, device=dev, points_per_batch=64)
    os.makedirs(args.out, exist_ok=True)
    for p in args.inputs:
        im = Image.open(p).convert('RGB')
        masks = gen(im, points_per_batch=64, pred_iou_thresh=0.86, stability_score_thresh=0.9)['masks']
        lab = np.zeros((im.height, im.width), np.uint16)
        for k, m in enumerate(sorted(masks, key=lambda m: -np.asarray(m).sum()), 1):
            lab[np.asarray(m, bool)] = k
        out = os.path.join(args.out, os.path.splitext(os.path.basename(p))[0] + '.png')
        Image.fromarray(lab).save(out)
        print(out, len(masks), flush=True)


if __name__ == '__main__':
    main()
