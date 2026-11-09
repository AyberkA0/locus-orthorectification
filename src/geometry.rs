//! Orthographic oblique-view model.
//!
//! A camera looking at the ground with tilt `theta` (angle between the viewing
//! direction and the ground normal) maps a ground point to the image with a
//! foreshortening of cos(theta) along the tilt direction `t`, and displaces a
//! point at height h by h·sin(theta) along `t` (relief displacement).
//! Depth along the viewing ray changes by +tan(theta) per image pixel on the
//! ground and by −cot(theta) on a vertical facade, measured along `t`. The
//! ratio of the two gradients is −cot²(theta), which makes the tilt observable
//! from a relative (affine) depth map without knowing its scale.

use image::RgbImage;
use serde::Serialize;

use crate::grid::{quantile, Grid};

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    /// Ground normal from a surface-normal map.
    Normals,
    /// Ground/facade depth-gradient ratio.
    Depth,
    /// Given on the command line.
    Manual,
}

#[derive(Clone, Serialize)]
pub struct Estimate {
    pub source: Source,
    /// Tilt from nadir, degrees.
    pub tilt_deg: f32,
    /// Direction in which the ground recedes, clockwise from image up, degrees.
    pub azimuth_deg: f32,
    /// Fraction of pixels classified as vertical facade.
    pub facade_frac: f32,
    /// Depth units to pixel units along the viewing ray (level 2 only).
    pub scale: Option<f32>,
    #[serde(skip)]
    pub t: [f32; 2],
    #[serde(skip)]
    pub plane: [f32; 3],
}

pub struct Params {
    pub min_facade: f32,
    pub max_shift: f32,
}

/// Unit surface normals in camera coordinates: x right, y up, z towards the camera.
pub struct Normals {
    pub n: Vec<[f32; 3]>,
}

impl Normals {
    /// Mode of the horizontal-surface normals (ground and roofs) by iterative
    /// re-centring, plus ground and facade masks.
    fn ground(&self) -> Option<([f32; 3], Vec<bool>, Vec<bool>)> {
        let dot = |a: &[f32; 3], b: &[f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let unit = |v: [f32; 3]| {
            let l = dot(&v, &v).sqrt().max(1e-12);
            [v[0] / l, v[1] / l, v[2] / l]
        };
        let mean = |sel: &dyn Fn(&[f32; 3]) -> bool| {
            let mut acc = [0f32; 3];
            let mut c = 0;
            for v in self.n.iter().filter(|v| sel(v)) {
                acc = [acc[0] + v[0], acc[1] + v[1], acc[2] + v[2]];
                c += 1;
            }
            (unit(acc), c)
        };
        let cos15 = 15f32.to_radians().cos();
        let (mut m, _) = mean(&|_| true);
        for _ in 0..15 {
            let (next, c) = mean(&|v| dot(v, &m) > cos15);
            if c < 100 {
                return None;
            }
            m = next;
        }
        let cos70 = 70f32.to_radians().cos();
        let ground = self.n.iter().map(|v| dot(v, &m) > cos15).collect();
        let facade = self.n.iter().map(|v| dot(v, &m).abs() < cos70).collect();
        Some((m, ground, facade))
    }
}

/// Fit the lower envelope of the depth map (the ground) with a plane d = a·x + b·y + c,
/// optionally restricted to a mask.
fn ground_plane(d: &Grid, mask: Option<&[bool]>) -> [f32; 3] {
    let mut keep: Vec<bool> = mask
        .map(|m| m.to_vec())
        .unwrap_or_else(|| vec![true; d.data.len()]);
    let base = keep.clone();
    let mut coef = [0f32; 3];
    for _ in 0..5 {
        let mut m = [[0f64; 3]; 3];
        let mut r = [0f64; 3];
        for y in 0..d.h {
            for x in 0..d.w {
                let i = y * d.w + x;
                if !keep[i] {
                    continue;
                }
                let a = [x as f64, y as f64, 1.0];
                for j in 0..3 {
                    for k in 0..3 {
                        m[j][k] += a[j] * a[k];
                    }
                    r[j] += a[j] * d.data[i] as f64;
                }
            }
        }
        coef = solve3(m, r).map(|v| v as f32);
        let res: Vec<f32> = (0..d.data.len())
            .map(|i| {
                d.data[i] - (coef[0] * (i % d.w) as f32 + coef[1] * (i / d.w) as f32 + coef[2])
            })
            .collect();
        let inside: Vec<f32> = res
            .iter()
            .zip(&base)
            .filter(|(_, &b)| b)
            .map(|(&v, _)| v)
            .collect();
        let cut = quantile(&inside, 0.4);
        keep = res
            .iter()
            .zip(&base)
            .map(|(&v, &b)| b && v <= cut)
            .collect();
    }
    coef
}

fn solve3(m: [[f64; 3]; 3], r: [f64; 3]) -> [f64; 3] {
    let det = |a: [[f64; 3]; 3]| {
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    };
    let d = det(m);
    let mut out = [0f64; 3];
    for c in 0..3 {
        let mut a = m;
        for row in 0..3 {
            a[row][c] = r[row];
        }
        out[c] = det(a) / d;
    }
    out
}

pub fn azimuth_vector(deg: f32) -> [f32; 2] {
    let a = deg.to_radians();
    [a.sin(), -a.cos()]
}

fn along(depth: &Grid, t: [f32; 2]) -> Vec<f32> {
    let (gx, gy) = depth.gradient();
    gx.data
        .iter()
        .zip(&gy.data)
        .map(|(a, b)| a * t[0] + b * t[1])
        .collect()
}

fn inside(i: usize, d: &Grid) -> bool {
    let (x, y, b) = (i % d.w, i / d.w, 4);
    x >= b && y >= b && x + b < d.w && y + b < d.h
}

/// Estimate tilt, azimuth and the depth scale. `depth` is smoothed and
/// normalized to [0, 1] in place. A manual tilt/azimuth overrides the estimate.
pub fn estimate(
    depth: &mut Grid,
    normals: Option<&Normals>,
    p: &Params,
    tilt: Option<f32>,
    azimuth: Option<f32>,
) -> Estimate {
    *depth = depth.gaussian(1.5).normalized();
    let manual = tilt.is_some() && azimuth.is_some();
    if let Some((m, ground, facade_mask)) = normals.and_then(|n| n.ground()) {
        let theta = tilt
            .map(f32::to_radians)
            .unwrap_or_else(|| m[2].clamp(-1.0, 1.0).acos());
        let tn = [m[0], -m[1]];
        let l = (tn[0] * tn[0] + tn[1] * tn[1]).sqrt().max(1e-12);
        let t = azimuth
            .map(azimuth_vector)
            .unwrap_or([tn[0] / l, tn[1] / l]);
        let plane = ground_plane(depth, Some(&ground));
        let a = along(depth, t);
        let fac: Vec<f32> = (0..a.len())
            .filter(|&i| facade_mask[i] && inside(i, depth))
            .map(|i| a[i])
            .collect();
        let facade_frac = fac.len() as f32 / a.len() as f32;
        let gf = if fac.len() >= 200 {
            quantile(&fac, 0.5)
        } else {
            0.0
        };
        let scale = (theta.to_degrees() >= 3.0 && gf > 1e-9).then(|| 1.0 / theta.tan() / gf);
        let source = if manual {
            Source::Manual
        } else {
            Source::Normals
        };
        return Estimate {
            source,
            tilt_deg: theta.to_degrees(),
            azimuth_deg: t[0].atan2(-t[1]).to_degrees(),
            facade_frac,
            scale,
            t,
            plane,
        };
    }
    let plane = ground_plane(depth, None);
    let g = [plane[0], plane[1]];
    let gnorm = (g[0] * g[0] + g[1] * g[1]).sqrt();
    let t = match azimuth {
        Some(a) => azimuth_vector(a),
        None => [-g[0] / gnorm.max(1e-12), -g[1] / gnorm.max(1e-12)],
    };
    let gg = -(t[0] * g[0] + t[1] * g[1]);
    let a = along(depth, t);
    let fac: Vec<f32> = (0..a.len())
        .filter(|&i| inside(i, depth) && a[i] > 0.5 * gg)
        .map(|i| a[i])
        .collect();
    let facade_frac = fac.len() as f32 / a.len() as f32;
    let theta = match tilt {
        Some(v) => v.to_radians(),
        None if gg > 1e-9 && facade_frac >= p.min_facade => {
            (gg / quantile(&fac, 0.5)).sqrt().atan()
        }
        None => 0.0,
    };
    let scale = (theta.to_degrees() >= 3.0 && gg > 1e-9).then(|| theta.tan() / gg);
    let source = if manual {
        Source::Manual
    } else {
        Source::Depth
    };
    Estimate {
        source,
        tilt_deg: theta.to_degrees(),
        azimuth_deg: t[0].atan2(-t[1]).to_degrees(),
        facade_frac,
        scale,
        t,
        plane,
    }
}

/// Affine stretch by 1/cos(theta) along t, about the image centre.
fn stretch(e: &Estimate, w: usize, h: usize) -> impl Fn(f32, f32) -> (f32, f32) {
    let s = 1.0 / e.tilt_deg.to_radians().cos().max(0.4);
    let t = e.t;
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    move |x, y| {
        let (dx, dy) = (x - cx, y - cy);
        let k = (s - 1.0) * (dx * t[0] + dy * t[1]);
        (cx + dx + k * t[0], cy + dy + k * t[1])
    }
}

fn sample(img: &RgbImage, x: f32, y: f32) -> [u8; 3] {
    let reflect = |v: f32, n: u32| {
        let m = 2.0 * (n as f32 - 1.0);
        let v = v.rem_euclid(m);
        if v > n as f32 - 1.0 {
            m - v
        } else {
            v
        }
    };
    let (x, y) = (reflect(x, img.width()), reflect(y, img.height()));
    let (x0, y0) = (x.floor() as u32, y.floor() as u32);
    let (x1, y1) = (
        (x0 + 1).min(img.width() - 1),
        (y0 + 1).min(img.height() - 1),
    );
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    std::array::from_fn(|c| {
        let p = |xx, yy| img.get_pixel(xx, yy)[c] as f32;
        let v = (p(x0, y0) * (1.0 - fx) + p(x1, y0) * fx) * (1.0 - fy)
            + (p(x0, y1) * (1.0 - fx) + p(x1, y1) * fx) * fy;
        v.round().clamp(0.0, 255.0) as u8
    })
}

/// Level 1: rectify the ground plane only.
pub fn level1(img: &RgbImage, e: &Estimate) -> RgbImage {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let s = 1.0 / e.tilt_deg.to_radians().cos().max(0.4);
    let t = e.t;
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    RgbImage::from_fn(w as u32, h as u32, |x, y| {
        let (dx, dy) = (x as f32 - cx, y as f32 - cy);
        let k = (1.0 / s - 1.0) * (dx * t[0] + dy * t[1]);
        image::Rgb(sample(img, cx + dx + k * t[0], cy + dy + k * t[1]))
    })
}

/// Level 2: move every pixel to its footprint using the height implied by the
/// depth map, then rectify the ground plane. Returns the image and the fraction
/// of output pixels that no source pixel reached (filled from the nearest one).
pub fn level2(img: &RgbImage, depth: &Grid, e: &Estimate, p: &Params) -> (RgbImage, f32) {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let Some(scale) = e.scale else {
        return (img.clone(), 0.0);
    };
    let theta = e.tilt_deg.to_radians();
    let fwd = stretch(e, w, h);
    let max_shift = p.max_shift * h as f32;
    let mut src: Vec<(f32, usize, usize)> = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let ground = e.plane[0] * x as f32 + e.plane[1] * y as f32 + e.plane[2];
            let hc = (scale * (depth.at(x, y) - ground)).max(0.0);
            let shift = (hc * theta.tan()).min(max_shift);
            let (u, v) = fwd(x as f32 - shift * e.t[0], y as f32 - shift * e.t[1]);
            let (u, v) = (u.round(), v.round());
            if u >= 0.0 && v >= 0.0 && (u as usize) < w && (v as usize) < h {
                src.push((hc, y * w + x, v as usize * w + u as usize));
            }
        }
    }
    src.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut out = vec![None::<[u8; 3]>; w * h];
    for (_, from, to) in src {
        let px = img.get_pixel((from % w) as u32, (from / w) as u32);
        out[to] = Some([px[0], px[1], px[2]]);
    }
    let holes = out.iter().filter(|v| v.is_none()).count() as f32 / (w * h) as f32;
    let filled = fill_nearest(out, w, h);
    (
        RgbImage::from_fn(w as u32, h as u32, |x, y| {
            image::Rgb(filled[y as usize * w + x as usize])
        }),
        holes,
    )
}

/// Fill empty pixels from the nearest filled pixel (multi-source BFS, 4-neighbourhood).
fn fill_nearest(mut px: Vec<Option<[u8; 3]>>, w: usize, h: usize) -> Vec<[u8; 3]> {
    let mut queue: std::collections::VecDeque<usize> =
        (0..px.len()).filter(|&i| px[i].is_some()).collect();
    while let Some(i) = queue.pop_front() {
        let (x, y) = (i % w, i / w);
        let v = px[i];
        for (nx, ny) in [
            (x.wrapping_sub(1), y),
            (x + 1, y),
            (x, y.wrapping_sub(1)),
            (x, y + 1),
        ] {
            if nx < w && ny < h && px[ny * w + nx].is_none() {
                px[ny * w + nx] = v;
                queue.push_back(ny * w + nx);
            }
        }
    }
    px.into_iter().map(|v| v.unwrap_or([0, 0, 0])).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Depth (larger = closer) of a scene seen with tilt `theta`, ground receding
    /// towards the top of the image, and one building whose facade spans rows
    /// `v0 - l ..= v0` and roof spans `v0 - l - r .. v0 - l` in columns 40..90.
    fn scene(theta: f32, w: usize, h: usize, v0: usize, l: usize, r: usize) -> Grid {
        let (a, c) = (theta.tan(), 1.0 / theta.tan());
        let mut g = Grid::new(w, h, 0.0);
        for y in 0..h {
            for x in 0..w {
                let ground = a * y as f32;
                let d = if (40..90).contains(&x) && y <= v0 && y + l >= v0 {
                    a * v0 as f32 + c * (v0 - y) as f32
                } else if (40..90).contains(&x) && y + l < v0 && y + l + r >= v0 {
                    ground + l as f32 * (a + c)
                } else {
                    ground
                };
                g.data[y * w + x] = d;
            }
        }
        g
    }

    fn params() -> Params {
        Params {
            min_facade: 0.005,
            max_shift: 0.25,
        }
    }

    #[test]
    fn depth_only_recovers_tilt_and_azimuth() {
        for deg in [20.0f32, 30.0, 40.0] {
            let mut d = scene(deg.to_radians(), 128, 160, 120, 30, 40);
            let e = estimate(&mut d, None, &params(), None, None);
            assert!((e.tilt_deg - deg).abs() < 2.0, "tilt {deg}: {}", e.tilt_deg);
            assert!(e.azimuth_deg.abs() < 2.0, "azimuth: {}", e.azimuth_deg);
        }
    }

    #[test]
    fn flat_ground_gives_zero_tilt() {
        let mut d = Grid::new(64, 64, 0.5);
        let e = estimate(&mut d, None, &params(), None, None);
        assert_eq!(e.tilt_deg, 0.0);
    }

    #[test]
    fn level1_at_nadir_is_identity() {
        let img = RgbImage::from_fn(32, 24, |x, y| image::Rgb([x as u8 * 7, y as u8 * 9, 100]));
        let mut d = Grid::new(32, 24, 0.0);
        let e = estimate(&mut d, None, &params(), Some(0.0), Some(0.0));
        assert_eq!(level1(&img, &e), img);
    }

    #[test]
    fn normals_give_exact_tilt() {
        let theta = 30f32.to_radians();
        let n = Normals {
            n: vec![[0.0, theta.sin(), theta.cos()]; 64 * 64],
        };
        let mut d = scene(theta, 64, 64, 50, 10, 10);
        let e = estimate(&mut d, Some(&n), &params(), None, None);
        assert!((e.tilt_deg - 30.0).abs() < 1e-2);
        assert!(e.azimuth_deg.abs() < 1e-2);
    }
}
