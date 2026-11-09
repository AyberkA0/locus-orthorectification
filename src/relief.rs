//! Ground-plane rectification (level 1) and relief correction (level 2).
//!
//! Orthographic oblique view with tilt θ: ground distances along the tilt
//! direction are foreshortened by cos θ, and a point at height h appears
//! h·sin θ pixels further along that direction than its footprint.
//!
//! Level 2 works in a frame rotated so that the tilt direction points up. The
//! image is split into surfaces at segment boundaries, image edges, depth edges
//! and vertical facades. Surfaces are visited from the bottom of the frame
//! upwards; a surface whose lower border meets a facade is a roof, and its relief
//! shift is the facade length below it plus the shift of whatever the facade
//! stands on. A surface continuing a neighbour at the same depth inherits the
//! neighbour's shift; one separated by a depth step lies behind it on the ground.
//! Raised surfaces without a visible facade, and tree crowns, get their height
//! from depth above the ground plane, scaled by the roofs whose facade is
//! visible. Facades are removed, surfaces are moved back to their footprints
//! (higher ones drawn last), uncovered ground is inpainted, and the result is
//! stretched by 1/cos θ.

use anyhow::Result;
use serde::Serialize;

use crate::grid::{canny, median, nearest_label, quantile, Grid, Mask};
use crate::models::Models;
use crate::tilt::{angle_to, azimuth_vector, gray, vegetation, Estimate, Normals};
use crate::warp::{warp, warp_mask, warp_rgb, Affine, Border, Rgb};

/// Below this tilt only level 1 is applied.
pub const MIN_TILT: f32 = 8.0;
/// Normals further than this from the ground normal are facades.
const FACADE_DEG: f32 = 55.0;
/// Shortest facade run (pixels) that counts as facade evidence.
const MIN_RUN: usize = 3;
/// Depth difference (normalized) under which two surfaces are continuous.
const SAME_DEPTH: f32 = 0.04;
const MIN_REGION: usize = 15;
const VEGETATION: f32 = 0.06;

#[derive(Serialize, Default)]
pub struct Info {
    pub surfaces: usize,
    pub roofs: usize,
    pub inherited: usize,
    pub raised: usize,
    pub trees: usize,
    pub hole_frac: f32,
}

fn stretch_affine(e: &Estimate, w: usize, h: usize) -> Affine {
    let s = 1.0 / e.tilt_deg.to_radians().cos().max(0.5);
    let t = azimuth_vector(e.azimuth_deg);
    let a = [
        1.0 + (s - 1.0) * t[0] * t[0],
        (s - 1.0) * t[0] * t[1],
        (s - 1.0) * t[1] * t[0],
        1.0 + (s - 1.0) * t[1] * t[1],
    ];
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    Affine([
        a[0],
        a[1],
        cx - a[0] * cx - a[1] * cy,
        a[2],
        a[3],
        cy - a[2] * cx - a[3] * cy,
    ])
}

/// Level 1: stretch by 1/cos θ along the tilt direction about the image centre.
pub fn level1(img: &Rgb, e: &Estimate) -> Rgb {
    let (w, h) = (img[0].w, img[0].h);
    warp_rgb(img, &stretch_affine(e, w, h), w, h, Border::Reflect)
}

/// Fill unknown pixels by mask-weighted downsampling and upsampling.
fn push_pull(img: &Rgb, known: &Mask) -> Rgb {
    let mut levels: Vec<(Vec<Grid>, Grid)> = vec![(
        img.iter()
            .map(|g| Grid {
                data: g
                    .data
                    .iter()
                    .zip(&known.data)
                    .map(|(v, &k)| if k { *v } else { 0.0 })
                    .collect(),
                ..g.clone()
            })
            .collect(),
        Grid {
            w: known.w,
            h: known.h,
            data: known.data.iter().map(|&k| k as u8 as f32).collect(),
        },
    )];
    let half = |g: &Grid| {
        let (w, h) = ((g.w / 2).max(1), (g.h / 2).max(1));
        let mut o = Grid::new(w, h, 0.0);
        for y in 0..h {
            for x in 0..w {
                let (x0, y0) = (2 * x, 2 * y);
                let (x1, y1) = ((x0 + 1).min(g.w - 1), (y0 + 1).min(g.h - 1));
                o.data[y * w + x] =
                    0.25 * (g.at(x0, y0) + g.at(x1, y0) + g.at(x0, y1) + g.at(x1, y1));
            }
        }
        o
    };
    while levels.last().unwrap().1.w.min(levels.last().unwrap().1.h) > 2 {
        let (c, wt) = levels.last().unwrap();
        let next = (c.iter().map(&half).collect(), half(wt));
        levels.push(next);
    }
    let ratio = |c: &[Grid], wt: &Grid| -> Vec<Grid> {
        c.iter()
            .map(|g| Grid {
                data: g
                    .data
                    .iter()
                    .zip(&wt.data)
                    .map(|(v, w)| v / w.max(1e-6))
                    .collect(),
                ..g.clone()
            })
            .collect()
    };
    let (c, wt) = levels.last().unwrap();
    let mut est = ratio(c, wt);
    for (c, wt) in levels.iter().rev().skip(1) {
        let cur = ratio(c, wt);
        est = (0..3)
            .map(|k| {
                let up = est[k].resize(wt.w, wt.h);
                Grid {
                    data: (0..up.data.len())
                        .map(|i| {
                            let a = wt.data[i].clamp(0.0, 1.0);
                            a * cur[k].data[i] + (1.0 - a) * up.data[i]
                        })
                        .collect(),
                    ..up
                }
            })
            .collect();
    }
    std::array::from_fn(|k| Grid {
        data: (0..known.data.len())
            .map(|i| {
                if known.data[i] {
                    img[k].data[i]
                } else {
                    est[k].data[i].clamp(0.0, 255.0)
                }
            })
            .collect(),
        ..img[k].clone()
    })
}

/// Square dilation with radius `r`.
fn dilate_square(m: &Mask, r: isize) -> Mask {
    let mut o = Mask::new(m.w, m.h, false);
    for y in 0..m.h as isize {
        for x in 0..m.w as isize {
            if !m.data[y as usize * m.w + x as usize] {
                continue;
            }
            for dy in -r..=r {
                for dx in -r..=r {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx >= 0 && ny >= 0 && (nx as usize) < m.w && (ny as usize) < m.h {
                        o.data[ny as usize * m.w + nx as usize] = true;
                    }
                }
            }
        }
    }
    o
}

/// Edges that separate surfaces, in the input frame.
fn surface_edges(img: &Rgb, depth: &Grid, segments: Option<&[u32]>) -> Mask {
    let (w, h) = (depth.w, depth.h);
    let mut e = canny(
        &Grid {
            w,
            h,
            data: gray(img),
        },
        60.0,
        160.0,
    );
    if let Some(seg) = segments {
        for y in 0..h {
            for x in 0..w {
                let l = seg[y * w + x];
                if l == 0 {
                    continue;
                }
                let border = x == 0 || y == 0 || x + 1 == w || y + 1 == h;
                let differs = !border
                    && (seg[y * w + x - 1] != l
                        || seg[y * w + x + 1] != l
                        || seg[(y - 1) * w + x] != l
                        || seg[(y + 1) * w + x] != l);
                if border || differs {
                    e.data[y * w + x] = true;
                }
            }
        }
    }
    let (lo, hi) = depth
        .data
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let dn = Grid {
        data: depth.data.iter().map(|v| v / (hi - lo + 1e-6)).collect(),
        ..depth.clone()
    };
    let (gx, gy) = dn.gaussian(1.0).gradient();
    for i in 0..e.data.len() {
        if gx.data[i].hypot(gy.data[i]) > 0.01 {
            e.data[i] = true;
        }
    }
    e.dilate()
}

/// Plane d = a·x + b·y + c through the lower 40 % of the samples, refitted four times.
fn ground_plane(d: &Grid, sel: &[bool]) -> [f32; 3] {
    let fit = |keep: &dyn Fn(usize) -> bool| {
        let mut m = [[0f64; 3]; 3];
        let mut r = [0f64; 3];
        for i in (0..d.data.len()).filter(|&i| keep(i)) {
            let a = [(i % d.w) as f64, (i / d.w) as f64, 1.0];
            for j in 0..3 {
                for k in 0..3 {
                    m[j][k] += a[j] * a[k];
                }
                r[j] += a[j] * d.data[i] as f64;
            }
        }
        solve3(m, r)
    };
    let mut c = fit(&|i| sel[i]);
    for _ in 0..4 {
        let res = |i: usize| d.data[i] - (c[0] * (i % d.w) as f32 + c[1] * (i / d.w) as f32 + c[2]);
        let r: Vec<f32> = (0..d.data.len()).filter(|&i| sel[i]).map(res).collect();
        let cut = quantile(&r, 0.4);
        c = fit(&|i| sel[i] && res(i) <= cut);
    }
    c
}

fn solve3(m: [[f64; 3]; 3], r: [f64; 3]) -> [f32; 3] {
    let det = |a: [[f64; 3]; 3]| {
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    };
    let d = det(m);
    std::array::from_fn(|c| {
        let mut a = m;
        for row in 0..3 {
            a[row][c] = r[row];
        }
        (det(a) / d) as f32
    })
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Ground,
    Roof,
    Inherited,
    Raised,
}

/// Level 2. Returns the rectified image and a summary.
pub fn level2(
    img: &Rgb,
    normals: &Normals,
    depth: &Grid,
    segments: Option<&[u32]>,
    e: &Estimate,
    models: &mut Models,
) -> Result<(Rgb, Info)> {
    let (w, h) = (img[0].w, img[0].h);
    let theta = e.tilt_deg.to_radians();
    let s = ((w * w + h * h) as f32).sqrt().ceil() as usize + 2;
    let rot = Affine::rotation(w as f32 / 2.0, h as f32 / 2.0, e.azimuth_deg)
        .translate((s - w) as f32 / 2.0, (s - h) as f32 / 2.0);
    let n_px = s * s;
    let valid = warp_mask(&Mask::new(w, h, true), &rot, s, s);
    let im = warp_rgb(img, &rot, s, s, Border::Zero);
    let ang = warp(
        &Grid {
            w,
            h,
            data: angle_to(normals, &e.ground),
        },
        &rot,
        s,
        s,
        true,
        Border::Zero,
    );
    let mut dp = warp(depth, &rot, s, s, false, Border::Zero);
    {
        let vals: Vec<f32> = (0..n_px)
            .filter(|&i| valid.data[i])
            .map(|i| dp.data[i])
            .collect();
        let lo = vals.iter().cloned().fold(f32::MAX, f32::min);
        let hi = vals.iter().cloned().fold(f32::MIN, f32::max);
        dp.data
            .iter_mut()
            .for_each(|v| *v = (*v - lo) / (hi - lo + 1e-6));
    }
    let dps = dp.gaussian(1.0);
    let (_, gy) = dp.gaussian(1.5).gradient();
    let g: Vec<f32> = gy.data.iter().map(|v| -v).collect();
    let gnd = median(
        &(0..n_px)
            .filter(|&i| valid.data[i] && ang.data[i] < 25.0)
            .map(|i| g[i])
            .collect::<Vec<_>>(),
    );
    let veg = vegetation(&im, VEGETATION);
    let veg = Mask {
        data: (0..n_px).map(|i| veg.data[i] && valid.data[i]).collect(),
        ..veg
    };
    let use_depth = e.tilt_deg >= 10.0;
    let dthr = (3.0 * gnd.abs()).max(2e-3);
    let facade = Mask {
        w: s,
        h: s,
        data: (0..n_px)
            .map(|i| {
                valid.data[i]
                    && !veg.data[i]
                    && (ang.data[i] > FACADE_DEG || (use_depth && g[i] > dthr))
            })
            .collect(),
    }
    .open(1);
    let edges = warp_mask(&surface_edges(img, depth, segments), &rot, s, s);

    // surfaces, visited from the bottom of the rotated frame upwards
    let free = Mask {
        w: s,
        h: s,
        data: (0..n_px)
            .map(|i| valid.data[i] && !facade.data[i] && !edges.data[i])
            .collect(),
    };
    let (reg, nreg) = free.label();
    let mut size = vec![0usize; nreg + 1];
    let mut lowest = vec![0usize; nreg + 1];
    let mut bottoms: Vec<std::collections::BTreeMap<usize, usize>> =
        vec![Default::default(); nreg + 1];
    for (i, &r) in reg.iter().enumerate() {
        let r = r as usize;
        if r == 0 {
            continue;
        }
        let (x, y) = (i % s, i / s);
        size[r] += 1;
        lowest[r] = lowest[r].max(y);
        let b = bottoms[r].entry(x).or_insert(y);
        *b = (*b).max(y);
    }
    let mut order: Vec<usize> = (1..=nreg).collect();
    order.sort_by(|a, b| lowest[*b].cmp(&lowest[*a]));
    let mut shift = vec![0f32; nreg + 1];
    let mut kind = vec![Kind::Ground; nreg + 1];
    let at = |m: &Mask, x: usize, y: usize| m.data[y * s + x];
    for &r in &order {
        if size[r] < MIN_REGION {
            continue;
        }
        let (mut f, mut same, mut occl) = (Vec::new(), Vec::new(), 0usize);
        for (&x, &y) in &bottoms[r] {
            let mut yy = y + 1;
            while yy < s && at(&edges, x, yy) && !at(&facade, x, yy) {
                yy += 1;
            }
            let mut run = 0;
            while yy < s && at(&facade, x, yy) {
                yy += 1;
                run += 1;
            }
            while yy < s && at(&edges, x, yy) && !at(&facade, x, yy) {
                yy += 1;
            }
            if yy >= s || !at(&valid, x, yy) {
                continue;
            }
            let j = reg[yy * s + x] as usize;
            if run >= MIN_RUN {
                f.push(run as f32 + if j > 0 { shift[j] } else { 0.0 });
            } else if j > 0 && run == 0 {
                if (dps.at(x, y) - dps.at(x, yy)).abs() < SAME_DEPTH {
                    same.push(shift[j]);
                } else {
                    occl += 1;
                }
            }
        }
        let n_ev = f.len() + same.len() + occl;
        if n_ev == 0 {
            continue;
        }
        if f.len() >= 5usize.max((0.2 * n_ev as f32).ceil() as usize) {
            shift[r] = median(&f);
            kind[r] = Kind::Roof;
        } else if same.len() > occl {
            shift[r] = median(&same);
            kind[r] = if shift[r] > 0.0 {
                Kind::Inherited
            } else {
                Kind::Ground
            };
        }
    }
    let mut region_px: Vec<Vec<usize>> = vec![Vec::new(); nreg + 1];
    for i in 0..n_px {
        region_px[reg[i] as usize].push(i);
    }
    let frac = |r: usize, m: &Mask| {
        region_px[r].iter().filter(|&&i| m.data[i]).count() as f32
            / region_px[r].len().max(1) as f32
    };
    for r in 1..=nreg {
        if size[r] >= MIN_REGION && frac(r, &veg) > 0.5 {
            shift[r] = 0.0;
            kind[r] = Kind::Ground;
        }
    }

    // heights from depth above the ground plane, scaled by roofs with a visible facade
    let gsel: Vec<bool> = (0..n_px)
        .map(|i| {
            let r = reg[i] as usize;
            r > 0 && size[r] >= MIN_REGION && kind[r] == Kind::Ground
        })
        .collect();
    let pl = ground_plane(&dps, &gsel);
    let above = Grid {
        data: (0..n_px)
            .map(|i| dps.data[i] - (pl[0] * (i % s) as f32 + pl[1] * (i / s) as f32 + pl[2]))
            .collect(),
        ..dps.clone()
    };
    let region_median =
        |r: usize, g: &Grid| median(&region_px[r].iter().map(|&i| g.data[i]).collect::<Vec<_>>());
    let samples: Vec<(f32, f32)> = (1..=nreg)
        .filter(|&r| kind[r] == Kind::Roof && size[r] >= MIN_REGION)
        .map(|r| (shift[r], region_median(r, &above)))
        .filter(|&(_, a)| a > 1e-3)
        .map(|(sh, a)| (sh / a, sh))
        .collect();
    let ratios: Vec<f32> = samples.iter().map(|v| v.0).collect();
    let scale_raised = (samples.len() >= 3).then(|| median(&ratios));
    // trees: broken facade runs underestimate a shift, so weight the ratios by run length
    let scale_trees = (!samples.is_empty()).then(|| {
        let mut v = samples.clone();
        v.sort_by(|a, b| a.0.total_cmp(&b.0));
        let half = v.iter().map(|p| p.1).sum::<f32>() / 2.0;
        let mut acc = 0.0;
        v.iter()
            .find(|p| {
                acc += p.1;
                acc >= half
            })
            .map_or(v[v.len() - 1].0, |p| p.0)
    });
    let mut info = Info::default();
    if let Some(k) = scale_raised {
        for r in 1..=nreg {
            if kind[r] != Kind::Ground || size[r] < 4 * MIN_REGION || frac(r, &veg) > 0.5 {
                continue;
            }
            let est = k * region_median(r, &above);
            if est > 6.0 {
                shift[r] = est;
                kind[r] = Kind::Raised;
            }
        }
    }

    // every pixel takes the shift of a nearby surface, preferring similar depth
    let lab: Vec<u32> = reg
        .iter()
        .map(|&r| {
            if r > 0 && size[r as usize] >= MIN_REGION {
                r
            } else {
                0
            }
        })
        .collect();
    let mut near = nearest_label(&lab, s, s);
    let rd: Vec<f32> = (0..=nreg)
        .map(|r| {
            if r > 0 && size[r] >= MIN_REGION {
                region_median(r, &dps)
            } else {
                0.0
            }
        })
        .collect();
    for y in 0..s {
        for x in 0..s {
            let i = y * s + x;
            if lab[i] != 0 {
                continue;
            }
            let mut bestd = (dps.data[i] - rd[near[i] as usize]).abs();
            for dy in -3isize..=3 {
                for dx in -3isize..=3 {
                    let (yy, xx) = (
                        (y as isize - dy).rem_euclid(s as isize) as usize,
                        (x as isize - dx).rem_euclid(s as isize) as usize,
                    );
                    let c = lab[yy * s + xx];
                    if c == 0 {
                        continue;
                    }
                    let d = (dps.data[i] - rd[c as usize]).abs();
                    if d < bestd {
                        bestd = d;
                        near[i] = c;
                    }
                }
            }
        }
    }
    let mut px: Vec<f32> = (0..n_px)
        .map(|i| {
            if facade.data[i] {
                -1.0
            } else {
                shift[near[i] as usize]
            }
        })
        .collect();

    // a segment that mostly moves, moves as one piece (at its own depth)
    if let Some(seg) = segments {
        let sl = warp(
            &Grid {
                w,
                h,
                data: seg.iter().map(|&v| v as f32).collect(),
            },
            &rot,
            s,
            s,
            true,
            Border::Zero,
        );
        let mut members: std::collections::BTreeMap<u32, Vec<usize>> = Default::default();
        for i in 0..n_px {
            let l = sl.data[i].round() as u32;
            if l > 0 && valid.data[i] && !facade.data[i] {
                members.entry(l).or_default().push(i);
            }
        }
        for body in members.values() {
            if body.len() < 50
                || median(&body.iter().map(|&i| ang.data[i]).collect::<Vec<_>>()) > 25.0
            {
                continue;
            }
            let moved: Vec<usize> = body.iter().cloned().filter(|&i| px[i] > 0.0).collect();
            if (moved.len() as f32) <= 0.5 * body.len() as f32 {
                continue;
            }
            let ref_d = median(&moved.iter().map(|&i| dps.data[i]).collect::<Vec<_>>());
            let v = median(&moved.iter().map(|&i| px[i]).collect::<Vec<_>>());
            for &i in body {
                if (dps.data[i] - ref_d).abs() < SAME_DEPTH {
                    px[i] = v;
                }
            }
        }
    }

    // facades reaching the top border belong to buildings whose roof is outside the image
    for x in 0..s {
        let Some(start) = (0..s).find(|&y| valid.data[y * s + x]) else {
            continue;
        };
        let mut y = start;
        while y < s && valid.data[y * s + x] && (facade.data[y * s + x] || edges.data[y * s + x]) {
            y += 1;
        }
        let n_f = (start..y).filter(|&yy| facade.data[yy * s + x]).count();
        if y - start >= 3 && n_f as f32 > 0.3 * (y - start) as f32 {
            for yy in start..y {
                px[yy * s + x] = -1.0;
            }
        }
    }

    // median-clean the shifts; removed facades stay removed
    let sm = Grid {
        w: s,
        h: s,
        data: px.iter().map(|&v| v.max(0.0)).collect(),
    }
    .median(7);
    for (v, m) in px.iter_mut().zip(&sm.data) {
        if *v >= 0.0 {
            *v = *m;
        }
    }

    // tree crowns: back over the trunk, compressed by cos θ so the stretch keeps them round
    let mut prio = px.clone();
    let mut tree = vec![false; n_px];
    if let Some(k) = scale_trees {
        let hpx = Grid {
            data: above.data.iter().map(|v| k * v).collect(),
            ..above.clone()
        }
        .gaussian(2.0);
        let cand = Mask {
            w: s,
            h: s,
            data: (0..n_px)
                .map(|i| veg.data[i] && hpx.data[i] > 4.0)
                .collect(),
        }
        .open(2);
        let (tl, nt) = cand.label();
        let mut crowns: Vec<Vec<usize>> = vec![Vec::new(); nt + 1];
        for i in 0..n_px {
            crowns[tl[i] as usize].push(i);
        }
        for c in crowns.iter().skip(1).filter(|c| c.len() >= 40) {
            let sh = median(&c.iter().map(|&i| hpx.data[i]).collect::<Vec<_>>());
            let cy = c.iter().map(|&i| (i / s) as f32).sum::<f32>() / c.len() as f32;
            for &i in c {
                let y = (i / s) as f32;
                px[i] = cy + sh + (y - cy) * theta.cos() - y;
                prio[i] = sh;
                tree[i] = true;
            }
            info.trees += 1;
        }
    }

    // move every surface to its footprint, lower ones first
    let mut src: Vec<usize> = (0..n_px)
        .filter(|&i| valid.data[i] && (px[i] >= 0.0 || tree[i]))
        .collect();
    src.sort_by(|a, b| prio[*a].total_cmp(&prio[*b]));
    let mut out: Rgb = std::array::from_fn(|_| Grid::new(s, s, 0.0));
    let mut filled = Mask::new(s, s, false);
    for i in src {
        let (x, y) = (i % s, (i / s) as isize);
        let yd = y + px[i].round() as isize;
        if yd < 0 || yd as usize >= s {
            continue;
        }
        let j = yd as usize * s + x;
        for c in 0..3 {
            out[c].data[j] = im[c].data[i];
        }
        filled.data[j] = true;
    }
    let hole = valid.and_not(&filled);
    info.hole_frac = hole.count() as f32 / valid.count().max(1) as f32;
    let out = push_pull(&out, &filled);

    // stretch, rotate back, inpaint what was hidden
    let st = Affine::stretch_y(1.0 / theta.cos().max(0.5), s as f32 / 2.0);
    let back = rot.inverse();
    let out = warp_rgb(
        &warp_rgb(&out, &st, s, s, Border::Reflect),
        &back,
        w,
        h,
        Border::Reflect,
    );
    let mut out = out;
    if models.has_lama() {
        let hm = warp_mask(&warp_mask(&hole, &st, s, s), &back, w, h);
        out = models.inpaint(&out, &dilate_square(&hm, 2))?;
    }
    info.surfaces = (1..=nreg).filter(|&r| size[r] >= MIN_REGION).count();
    info.roofs = kind.iter().filter(|&&k| k == Kind::Roof).count();
    info.inherited = kind.iter().filter(|&&k| k == Kind::Inherited).count();
    info.raised = kind.iter().filter(|&&k| k == Kind::Raised).count();
    Ok((out, info))
}
