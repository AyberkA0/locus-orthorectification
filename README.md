# locus-orthorectification

`locus-ortho` is a command-line tool that rectifies an oblique aerial image to a nadir (top-down) view from the single image itself. It estimates the viewing tilt per image, corrects the ground plane, and optionally moves above-ground pixels back to their footprint using monocular depth.

It was written for cross-view geo-localization, where models trained on nadir satellite tiles are tested on cities whose reference imagery is oblique.

![Synthetic scenes: input, normals, depth, level 1, level 2 and the nadir reference](docs/figures/synthetic_pipeline.png)

*Two synthetic scenes with known geometry. From left to right: oblique input, surface normals, depth, ground-plane rectification (level 1), relief correction (level 2), and the nadir rendering of the same scene.*

## Model

The camera is treated as orthographic, which holds for aerial and satellite imagery where the scene depth is small compared to the distance to the camera. Let θ be the angle between the viewing direction and the ground normal (θ = 0 is nadir), and let **t** be the image direction in which the ground recedes.

- **Ground plane.** Distances along **t** are foreshortened by cos θ. Stretching the image by 1 / cos θ along **t** restores the nadir ground plane.
- **Relief displacement.** A point at height *h* appears *h* sin θ pixels further along **t** than its footprint.
- **Depth gradients.** Along **t**, depth towards the camera changes by −tan θ per pixel on the ground and by +cot θ on a vertical facade. Their ratio is cot² θ, independent of the unknown scale of a relative depth map.

![Relief displacement under an orthographic camera](docs/figures/geometry.png)

## Method

1. **Tilt and azimuth**, from one of three sources:
   - **normals** (recommended): the horizontal-surface normal is the mode of a surface-normal map, found by iterative re-centring within 15°. Its angle to the optical axis is θ, and its image-plane component gives **t**.
   - **depth**: a plane is fitted to the lower envelope of the depth map (the ground), which gives **t** and the ground gradient. Pixels whose gradient along **t** has the opposite sign are taken as facades, and θ = arctan √(g<sub>ground</sub> / g<sub>facade</sub>).
   - **manual**: `--tilt` and `--azimuth`.
2. **Level 1.** Affine stretch by 1 / cos θ along **t** about the image centre, with bilinear resampling.
3. **Level 2.** The height of every pixel is computed from its depth above the fitted ground plane. With normals, the depth scale is calibrated on facade pixels; otherwise it is calibrated on the ground gradient. Every pixel is shifted back by *h* sin θ along **t**, level 1 is applied, and the pixels are splatted with a z-buffer (higher points win). Ground hidden behind buildings is not observed. It is filled from the nearest observed pixel and reported as `hole_frac`.

Depth comes from [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) through ONNX Runtime, or from a depth map given with `--depth`. Normal maps can be produced with [Marigold](https://github.com/prs-eth/Marigold) using `scripts/normals.py`.

## Installation

Building requires Rust 1.92 or newer. ONNX Runtime is downloaded by the `ort` crate at build time.

```bash
cargo build --release
mkdir -p models
curl -L -o models/depth_anything_v2_small.onnx \
  https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx
```

Normal maps need Python with `torch`, `diffusers`, `transformers`, `accelerate` and `opencv-python-headless`:

```bash
python scripts/normals.py images/*.jpg --out normals/
```

## Usage

```bash
# level 2 with normal maps (normals/<stem>.png for every input)
target/release/locus-ortho images/*.jpg --normals normals/ -o rectified/ --report report.json

# ground-plane rectification only
target/release/locus-ortho images/*.jpg --normals normals/ --level 1 -o rectified/

# known camera parameters
target/release/locus-ortho tile.jpg --tilt 30 --azimuth 0 --level 1
```

Each run prints one line per image:

```
tile.jpg  [normals]  tilt  24.3°  azimuth   -2.0°  facade 0.137  holes 0.214
```

| Option | Default | Meaning |
| --- | --- | --- |
| `-l, --level` | 2 | 1 = ground plane only, 2 = with relief correction |
| `--normals` | — | normal map file, or a directory of `<stem>.png` |
| `--depth` | — | depth map (16-bit PNG, larger = closer) used instead of the model |
| `-m, --model` | `models/depth_anything_v2_small.onnx` | Depth Anything V2 ONNX model (also `LOCUS_ORTHO_MODEL`) |
| `--tilt`, `--azimuth` | — | override the estimate; azimuth is clockwise from image up |
| `--max-shift` | 0.25 | upper bound on the relief shift, as a fraction of image height |
| `--min-facade` | 0.005 | minimum facade fraction for a non-zero depth-only tilt |
| `--report` | — | JSON file with `source`, `tilt_deg`, `azimuth_deg`, `facade_frac`, `scale`, `hole_frac` per image |

Normal maps are 16-bit RGB PNGs with **n** = v / 65535 · 2 − 1 in camera coordinates (x right, y up, z towards the camera).

## Evaluation

### Synthetic scenes

`scripts/synthetic.py` renders block-and-street scenes under an orthographic camera with a random tilt (20–45°) and image rotation. For each scene it writes the oblique view, the nadir view with the same image orientation, and ground-truth depth and normals. The table reports means over 8 scenes. SSIM is measured against the nadir view; the input scores 0.434.

| Tilt / depth source | Tilt error (°) | Azimuth error (°) | SSIM, level 1 | SSIM, level 2 |
| --- | --- | --- | --- | --- |
| true normals, true depth | 0.0 | 0.0 | 0.610 | 0.713 |
| true normals, Depth Anything V2 Small | 0.0 | 0.0 | 0.610 | 0.601 |
| depth only, true depth | 0.9 | 0.0 | 0.590 | 0.686 |
| depth only, Depth Anything V2 Small | 9.6 | 18.1 | 0.463 | 0.438 |

The geometry is exact when its inputs are, and level 2 improves on level 1 with true depth. With Depth Anything V2 Small, level 2 is slightly worse than level 1, and the depth-only tilt estimate is unreliable (see Limitations).

![Level 2 with true and with estimated depth](docs/figures/depth_source.png)

```bash
python scripts/synthetic.py --out examples/synthetic --scenes 8
python scripts/evaluate_synthetic.py --scenes examples/synthetic
```

### Real oblique imagery

We tested the method on CVG-Text [3] satellite tiles from Brisbane (near-nadir), New York and Tokyo (oblique). For each city, 200 test pairs were drawn at random (seed 0) and used as both the query set and the gallery. Each tile was rectified, and ground-to-aerial retrieval was measured with models trained on nadir CVUSA tiles:

- the published Sample4Geo [4] CVUSA weights, without further training;
- ConvNeXt-B models initialized from Sample4Geo CVACT weights and trained for 4 epochs on CVUSA, two seeds each: RGB, and RGB plus a panorama-depth branch ("RGB + depth").

These numbers come from the Python prototype of the same geometry, not from the Rust binary. The prototype used Marigold Normals v1-1 (4 steps, ensemble of 3, 512 px), Depth Anything V2 Base, and Telea inpainting for holes. Values are R@1 (%) on the 200-image gallery, and each cell lists input / level 1 / level 2. The tiles are not reproduced here because of the imagery license.

| Model | Brisbane | New York | Tokyo |
| --- | --- | --- | --- |
| Sample4Geo CVUSA, no training | 98.0 / 98.0 / 95.0 | 47.0 / 48.5 / 38.5 | 65.0 / **72.0** / 61.5 |
| RGB, seed 0 | 97.0 / 96.5 / 94.5 | 39.0 / **43.5** / 41.0 | 63.5 / 65.0 / 58.0 |
| RGB, seed 1 | 96.5 / 96.5 / 94.0 | 42.5 / 45.0 / 43.0 | 62.5 / 63.5 / 56.0 |
| RGB + depth, seed 0 | 96.0 / 96.5 / 94.5 | 44.5 / 41.5 / 41.5 | 62.0 / 61.5 / 59.0 |
| RGB + depth, seed 1 | 95.5 / 95.5 / 93.0 | 44.5 / 44.0 / 41.5 | 60.5 / 64.5 / 61.5 |

Bold values differ from the input at p < 0.05 (exact McNemar test on per-query R@1: p = 0.001 for Tokyo and p = 0.049 for New York). No other level-1 change is significant at this sample size.

- **Level 1, RGB models.** On the oblique cities it raised R@1 by 1.0–7.0 points in all six cases.
- **Level 1, RGB + depth models.** No consistent effect: −3.0 to +4.0 points.
- **Brisbane.** Near-nadir tiles are unchanged, as expected.
- **Level 2.** It never scored above level 1: equal in one of 15 cases, lower in the others.

## Limitations

- **Depth-only tilt is not reliable on real imagery.** On 600 tiles from Brisbane, New York and Tokyo, the depth-only estimate from Depth Anything V2 Small differed from the Marigold-normal estimate by 9.7–19.0° on average. For Brisbane tiles, which Marigold places at 9.6° on average, it reported about 28°. Use normal maps or known camera parameters.
- **Level 2 depends on depth quality.** Errors in monocular depth turn into wrong shifts, striped facades and smeared roofs. In the retrieval tests on real imagery it never scored above level 1.
- **Single orthographic view.** Ground hidden behind buildings cannot be recovered from one image; it is filled from neighbouring pixels. Perspective effects of wide-angle imagery are not modelled.
- **Horizontal ground is assumed.** Strongly sloped terrain biases the tilt estimate.
- **Depth is normalized per image.** On nearly flat depth maps, small noise is amplified and can produce a spurious depth-only tilt.

## Third-party models

| Model | Used for | License |
| --- | --- | --- |
| Depth Anything V2 Small ([ONNX](https://huggingface.co/onnx-community/depth-anything-v2-small)) | depth | Apache-2.0 |
| Marigold Normals v1-1 ([model card](https://huggingface.co/prs-eth/marigold-normals-v1-1)) | normal maps (`scripts/normals.py`) | OpenRAIL++ |

## References

- [1] L. Yang et al. *Depth Anything V2.* NeurIPS 2024.
- [2] B. Ke et al. *Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation* (Marigold). CVPR 2024; Marigold v1-1 normals.
- [3] J. Ye et al. *Where am I? Cross-View Geo-localization with Natural Language Descriptions* (CVG-Text). ICCV 2025.
- [4] F. Deuser, K. Habel, N. Oswald. *Sample4Geo: Hard Negative Sampling For Cross-View Geo-Localisation.* ICCV 2023.
