"""Surface normals for locus-ortho with Marigold Normals v1-1.

Writes one 16-bit RGB PNG per input image, v = (n + 1) / 2 · 65535, in camera
coordinates (x right, y up, z towards the camera), at the input resolution.

    pip install torch diffusers transformers accelerate opencv-python-headless
    python scripts/normals.py images/*.jpg --out normals/
"""
import argparse
import os

import cv2
import diffusers
import numpy as np
import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', nargs='+')
    ap.add_argument('--out', default='normals')
    ap.add_argument('--resolution', type=int, default=512)
    ap.add_argument('--ensemble', type=int, default=3)
    ap.add_argument('--steps', type=int, default=4)
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    dtype = torch.float32 if dev == 'cpu' else torch.float16
    pipe = diffusers.MarigoldNormalsPipeline.from_pretrained('prs-eth/marigold-normals-v1-1', variant='fp16', torch_dtype=dtype).to(dev)
    os.makedirs(args.out, exist_ok=True)
    for p in args.inputs:
        im = Image.open(p).convert('RGB')
        n = pipe(im, num_inference_steps=args.steps, ensemble_size=args.ensemble, processing_resolution=args.resolution,
                 output_type='np', generator=torch.Generator('cpu').manual_seed(0)).prediction[0]
        n = cv2.resize(n, im.size, interpolation=cv2.INTER_LINEAR)
        n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-6
        v = np.round((np.clip(n, -1, 1) + 1) / 2 * 65535).astype(np.uint16)
        out = os.path.join(args.out, os.path.splitext(os.path.basename(p))[0] + '.png')
        cv2.imwrite(out, v[..., ::-1])
        print(out, flush=True)


if __name__ == '__main__':
    main()
