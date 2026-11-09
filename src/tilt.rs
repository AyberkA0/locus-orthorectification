//! Viewing tilt and azimuth.
//!
//! The azimuth (the image direction in which the ground recedes) and a first tilt
//! come from the dominant horizontal-surface normal. In scenes with a street grid
//! the tilt is refined by metric rectification: lines on horizontal surfaces
//! (streets, roof edges) are mostly at right angles in a nadir view, so the
//! stretch along the tilt direction that makes them most orthogonal is 1/cos(tilt).

use serde::Serialize;

use crate::grid::Mask;
use crate::warp::Rgb;

pub type Normals = Vec<[f32; 3]>;

#[derive(Clone, Copy, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    Lines,
    Normals,
    Manual,
}

#[derive(Clone, Serialize)]
pub struct Estimate {
    pub source: Source,
    /// Tilt from nadir, degrees.
    pub tilt_deg: f32,
    /// Direction in which the ground recedes, clockwise from image up, degrees.
    pub azimuth_deg: f32,
    /// Tilt of the dominant horizontal-surface normal, degrees.
    pub normals_tilt_deg: Option<f32>,
    /// Tilt from line orthogonality, degrees, with its score and family balance.
    pub lines_tilt_deg: Option<f32>,
    pub lines_score: Option<f32>,
    pub lines_balance: Option<f32>,
    pub lines_contrast: Option<f32>,
    pub lines: usize,
    /// Ground normal in camera coordinates (x right, y up, z towards the camera).
    #[serde(skip)]
    pub ground: [f32; 3],
}

fn dot(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn unit(v: [f32; 3]) -> [f32; 3] {
    let l = dot(&v, &v).sqrt().max(1e-12);
    [v[0] / l, v[1] / l, v[2] / l]
}

/// Mode of the horizontal-surface normals: re-centre on the normals within 15°
/// of the current guess, starting from nadir, or from the mean normal when no
/// surface lies within 15° of nadir. Vegetation is left out: crowns have normals
/// in all directions.
pub fn ground_normal(all: &Normals, exclude: Option<&Mask>) -> [f32; 3] {
    let kept: Normals = match exclude {
        Some(e) if e.data.iter().filter(|&&v| !v).count() >= 1000 => all
            .iter()
            .zip(&e.data)
            .filter(|(_, &x)| !x)
            .map(|(v, _)| *v)
            .collect(),
        _ => all.clone(),
    };
    let n = &kept;
    let cos15 = 15f32.to_radians().cos();
    let climb = |mut m: [f32; 3]| {
        for k in 0..15 {
            let mut acc = [0f32; 3];
            let mut c = 0;
            for v in n.iter().filter(|v| dot(v, &m) > cos15) {
                acc = [acc[0] + v[0], acc[1] + v[1], acc[2] + v[2]];
                c += 1;
            }
            if c < 100 {
                return (k > 0).then_some(m);
            }
            m = unit(acc);
        }
        Some(m)
    };
    let mean = unit(
        n.iter()
            .fold([0f32; 3], |a, v| [a[0] + v[0], a[1] + v[1], a[2] + v[2]]),
    );
    climb([0.0, 0.0, 1.0])
        .or_else(|| climb(mean))
        .unwrap_or(mean)
}

/// Angle between each normal and the ground normal, degrees.
pub fn angle_to(n: &Normals, m: &[f32; 3]) -> Vec<f32> {
    n.iter()
        .map(|v| dot(v, m).clamp(-1.0, 1.0).acos().to_degrees())
        .collect()
}

/// Excess-green vegetation index 2G − R − B (0..1 scale), averaged over 5 x 5.
pub fn vegetation(img: &Rgb, thr: f32) -> Mask {
    let mut exg = img[0].clone();
    for i in 0..exg.data.len() {
        exg.data[i] = (2.0 * img[1].data[i] - img[0].data[i] - img[2].data[i]) / 255.0;
    }
    let e = exg.box_mean(5);
    Mask {
        w: e.w,
        h: e.h,
        data: e.data.iter().map(|&v| v > thr).collect(),
    }
}

pub fn gray(img: &Rgb) -> Vec<f32> {
    (0..img[0].data.len())
        .map(|i| 0.299 * img[0].data[i] + 0.587 * img[1].data[i] + 0.114 * img[2].data[i])
        .collect()
}

pub fn azimuth_vector(deg: f32) -> [f32; 2] {
    let a = deg.to_radians();
    [a.sin(), -a.cos()]
}

struct LineFit {
    tilt_deg: f32,
    score: f32,
    balance: f32,
    /// max − min of the score over the stretch range; near 0 when the line
    /// families run along and across the tilt direction, which no stretch changes
    contrast: f32,
}

/// Segments of at least 12 px lying (70 % of 8 samples) inside `mask`.
fn lines_on(img: &Rgb, mask: &Mask) -> Vec<[f32; 4]> {
    let (w, h) = (mask.w, mask.h);
    let g = gray(img);
    let Ok(segs) = lsdetect::detect(&g, w, h, lsdetect::SCALE) else {
        return Vec::new();
    };
    segs.iter()
        .map(|s| [s.x1 as f32, s.y1 as f32, s.x2 as f32, s.y2 as f32])
        .filter(|s| {
            if (s[2] - s[0]).hypot(s[3] - s[1]) < 12.0 {
                return false;
            }
            let inside = (0..8)
                .filter(|&k| {
                    let f = k as f32 / 7.0;
                    let x = (s[0] + f * (s[2] - s[0]))
                        .round()
                        .clamp(0.0, (w - 1) as f32);
                    let y = (s[1] + f * (s[3] - s[1]))
                        .round()
                        .clamp(0.0, (h - 1) as f32);
                    mask.data[y as usize * w + x as usize]
                })
                .count();
            inside as f32 / 8.0 > 0.7
        })
        .collect()
}

/// Stretch s = 1/cos(tilt) along `t` that maximizes the 4-fold alignment
/// R(s) = |Σ w·exp(4iφ)| / Σ w of the line directions φ, w = length.
fn fit_lines(segs: &[[f32; 4]], t: [f32; 2]) -> LineFit {
    let p = [-t[1], t[0]];
    let d: Vec<(f32, f32, f32)> = segs
        .iter()
        .map(|s| {
            let (dx, dy) = (s[2] - s[0], s[3] - s[1]);
            (dx * t[0] + dy * t[1], dx * p[0] + dy * p[1], dx.hypot(dy))
        })
        .collect();
    let total: f32 = d.iter().map(|v| v.2).sum();
    let score = |s: f32| {
        let (mut re, mut im) = (0f32, 0f32);
        for &(dt, dp, w) in &d {
            let phi = 4.0 * (dt * s).atan2(dp);
            re += w * phi.cos();
            im += w * phi.sin();
        }
        (re.hypot(im) / total, im.atan2(re) / 4.0)
    };
    let (mut best, mut best_s, mut best_dom, mut worst) = (-1f32, 1f32, 0f32, f32::MAX);
    for k in 0..=300 {
        let s = 1.0 + 1.5 * k as f32 / 300.0;
        let (r, dom) = score(s);
        worst = worst.min(r);
        if r > best {
            (best, best_s, best_dom) = (r, s, dom);
        }
    }
    // weight of the two orthogonal families (within 15° of the dominant directions)
    let (mut f1, mut f2) = (0f32, 0f32);
    let pi = std::f32::consts::PI;
    for &(dt, dp, w) in &d {
        let phi = (dt * best_s).atan2(dp);
        let dd = ((phi - best_dom + pi / 2.0).rem_euclid(pi) - pi / 2.0).abs();
        if dd < 15f32.to_radians() {
            f1 += w;
        } else if (dd - pi / 2.0).abs() < 15f32.to_radians() {
            f2 += w;
        }
    }
    LineFit {
        tilt_deg: (1.0 / best_s).acos().to_degrees(),
        score: best,
        balance: f1.min(f2) / f1.max(f2).max(1e-6),
        contrast: best - worst,
    }
}

/// Estimate tilt and azimuth. The line tilt is used when the scene shows two
/// orthogonal line families (score ≥ 0.6, balance ≥ 0.5, at least 40 lines)
/// that are not aligned with the tilt direction (contrast ≥ 0.1); otherwise the
/// normal tilt. A manual tilt/azimuth overrides either.
pub fn estimate(
    img: &Rgb,
    normals: Option<&Normals>,
    tilt: Option<f32>,
    azimuth: Option<f32>,
) -> Estimate {
    let mut e = Estimate {
        source: Source::Manual,
        tilt_deg: tilt.unwrap_or(0.0),
        azimuth_deg: azimuth.unwrap_or(0.0),
        normals_tilt_deg: None,
        lines_tilt_deg: None,
        lines_score: None,
        lines_balance: None,
        lines_contrast: None,
        lines: 0,
        ground: [0.0, 0.0, 1.0],
    };
    let Some(n) = normals else {
        return e;
    };
    let veg = vegetation(img, 0.06);
    let m = ground_normal(n, Some(&veg));
    e.ground = m;
    let theta = m[2].clamp(-1.0, 1.0).acos().to_degrees();
    let tl = m[0].hypot(m[1]).max(1e-9);
    let t = match azimuth {
        Some(a) => azimuth_vector(a),
        None => [m[0] / tl, -m[1] / tl],
    };
    e.azimuth_deg = t[0].atan2(-t[1]).to_degrees();
    e.normals_tilt_deg = Some(theta);
    if tilt.is_some() {
        return e;
    }
    e.source = Source::Normals;
    e.tilt_deg = theta;
    let ang = angle_to(n, &m);
    let flat = Mask {
        w: veg.w,
        h: veg.h,
        data: (0..ang.len())
            .map(|i| ang[i] < 25.0 && !veg.data[i])
            .collect(),
    };
    let segs = lines_on(img, &flat);
    e.lines = segs.len();
    if segs.len() < 40 {
        return e;
    }
    let f = fit_lines(&segs, t);
    e.lines_tilt_deg = Some(f.tilt_deg);
    e.lines_score = Some(f.score);
    e.lines_balance = Some(f.balance);
    e.lines_contrast = Some(f.contrast);
    if f.score >= 0.6 && f.balance >= 0.5 && f.contrast >= 0.1 {
        e.source = Source::Lines;
        e.tilt_deg = f.tilt_deg;
    }
    e
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A street grid rotated by `rot_deg` on the ground, foreshortened by cos(tilt) along image y.
    fn grid_lines(tilt_deg: f32, rot_deg: f32) -> Vec<[f32; 4]> {
        let c = tilt_deg.to_radians().cos();
        let a = rot_deg.to_radians();
        let mut out = Vec::new();
        for k in 0..30 {
            let len = 15.0 + k as f32;
            for (dx, dy) in [(a.cos(), a.sin()), (-a.sin(), a.cos())] {
                out.push([100.0, 100.0, 100.0 + len * dx, 100.0 + len * c * dy]);
            }
        }
        out
    }

    #[test]
    fn line_fit_recovers_tilt() {
        for tilt in [0.0f32, 20.0, 35.0, 45.0] {
            let f = fit_lines(&grid_lines(tilt, 25.0), [0.0, -1.0]);
            assert!((f.tilt_deg - tilt).abs() < 1.0, "{tilt}: {}", f.tilt_deg);
            assert!(
                f.score > 0.99 && f.balance > 0.8,
                "{tilt}: score {} balance {}",
                f.score,
                f.balance
            );
        }
    }

    #[test]
    fn ground_normal_climbs_to_the_mode() {
        // most surfaces at 30°, the rest spread from 10° to 50°
        let at = |deg: f32| {
            let a = deg.to_radians();
            [0.0, a.sin(), a.cos()]
        };
        let mut n = vec![at(30.0); 3000];
        n.extend((0..1000).map(|k| at(10.0 + 40.0 * k as f32 / 999.0)));
        let m = ground_normal(&n, None);
        assert!((m[2].acos().to_degrees() - 30.0).abs() < 0.5);
    }
}
