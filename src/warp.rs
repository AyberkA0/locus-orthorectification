//! 2-D affine maps and resampling.

use crate::grid::{Grid, Mask};

/// Forward map from source to destination pixel coordinates:
/// x' = a·x + b·y + c, y' = d·x + e·y + f.
#[derive(Clone, Copy)]
pub struct Affine(pub [f32; 6]);

#[derive(Clone, Copy, PartialEq)]
pub enum Border {
    Zero,
    Reflect,
}

impl Affine {
    /// Rotation by `deg` counter-clockwise (as displayed, y down) about `(cx, cy)`.
    pub fn rotation(cx: f32, cy: f32, deg: f32) -> Affine {
        let (s, c) = deg.to_radians().sin_cos();
        Affine([
            c,
            s,
            (1.0 - c) * cx - s * cy,
            -s,
            c,
            s * cx + (1.0 - c) * cy,
        ])
    }

    /// Stretch by `s` along the image y axis about row `cy`.
    pub fn stretch_y(s: f32, cy: f32) -> Affine {
        Affine([1.0, 0.0, 0.0, 0.0, s, cy * (1.0 - s)])
    }

    pub fn translate(mut self, dx: f32, dy: f32) -> Affine {
        self.0[2] += dx;
        self.0[5] += dy;
        self
    }

    pub fn apply(&self, x: f32, y: f32) -> (f32, f32) {
        let m = self.0;
        (m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5])
    }

    pub fn inverse(&self) -> Affine {
        let [a, b, c, d, e, f] = self.0;
        let det = a * e - b * d;
        let (ia, ib, id, ie) = (e / det, -b / det, -d / det, a / det);
        Affine([ia, ib, -(ia * c + ib * f), id, ie, -(id * c + ie * f)])
    }
}

fn reflect(v: isize, n: usize) -> usize {
    let n = n as isize;
    let m = 2 * n;
    let mut v = v.rem_euclid(m);
    if v >= n {
        v = m - 1 - v;
    }
    v as usize
}

/// Resample `src` into a `w` x `h` grid under the forward map `m`.
pub fn warp(src: &Grid, m: &Affine, w: usize, h: usize, nearest: bool, border: Border) -> Grid {
    let inv = m.inverse();
    let get = |x: isize, y: isize| -> f32 {
        if x >= 0 && y >= 0 && (x as usize) < src.w && (y as usize) < src.h {
            src.at(x as usize, y as usize)
        } else if border == Border::Reflect {
            src.at(reflect(x, src.w), reflect(y, src.h))
        } else {
            0.0
        }
    };
    let mut out = Grid::new(w, h, 0.0);
    for y in 0..h {
        for x in 0..w {
            let (u, v) = inv.apply(x as f32, y as f32);
            out.data[y * w + x] = if nearest {
                get(u.round() as isize, v.round() as isize)
            } else {
                let (x0, y0) = (u.floor(), v.floor());
                let (fx, fy) = (u - x0, v - y0);
                let (x0, y0) = (x0 as isize, y0 as isize);
                (get(x0, y0) * (1.0 - fx) + get(x0 + 1, y0) * fx) * (1.0 - fy)
                    + (get(x0, y0 + 1) * (1.0 - fx) + get(x0 + 1, y0 + 1) * fx) * fy
            };
        }
    }
    out
}

pub fn warp_mask(src: &Mask, m: &Affine, w: usize, h: usize) -> Mask {
    let g = Grid {
        w: src.w,
        h: src.h,
        data: src.data.iter().map(|&b| b as u8 as f32).collect(),
    };
    let o = warp(&g, m, w, h, true, Border::Zero);
    Mask {
        w,
        h,
        data: o.data.iter().map(|&v| v > 0.5).collect(),
    }
}

/// An RGB image as three float planes in 0..255.
pub type Rgb = [Grid; 3];

pub fn warp_rgb(src: &Rgb, m: &Affine, w: usize, h: usize, border: Border) -> Rgb {
    std::array::from_fn(|c| warp(&src[c], m, w, h, false, border))
}
