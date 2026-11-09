#[derive(Clone)]
pub struct Grid {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

impl Grid {
    pub fn new(w: usize, h: usize, v: f32) -> Self {
        Grid {
            w,
            h,
            data: vec![v; w * h],
        }
    }

    pub fn at(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.w + x]
    }

    pub fn bilinear(&self, x: f32, y: f32) -> f32 {
        let x = x.clamp(0.0, (self.w - 1) as f32);
        let y = y.clamp(0.0, (self.h - 1) as f32);
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.w - 1), (y0 + 1).min(self.h - 1));
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let top = self.at(x0, y0) * (1.0 - fx) + self.at(x1, y0) * fx;
        let bot = self.at(x0, y1) * (1.0 - fx) + self.at(x1, y1) * fx;
        top * (1.0 - fy) + bot * fy
    }

    pub fn resize(&self, w: usize, h: usize) -> Grid {
        let sx = self.w as f32 / w as f32;
        let sy = self.h as f32 / h as f32;
        let mut out = Grid::new(w, h, 0.0);
        for y in 0..h {
            for x in 0..w {
                out.data[y * w + x] =
                    self.bilinear((x as f32 + 0.5) * sx - 0.5, (y as f32 + 0.5) * sy - 0.5);
            }
        }
        out
    }

    pub fn gaussian(&self, sigma: f32) -> Grid {
        let r = (3.0 * sigma).ceil() as isize;
        let k: Vec<f32> = (-r..=r)
            .map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp())
            .collect();
        let s: f32 = k.iter().sum();
        let k: Vec<f32> = k.iter().map(|v| v / s).collect();
        let pass = |src: &Grid, horizontal: bool| {
            let mut out = Grid::new(src.w, src.h, 0.0);
            for y in 0..src.h {
                for x in 0..src.w {
                    let mut acc = 0.0;
                    for (j, kv) in k.iter().enumerate() {
                        let o = j as isize - r;
                        let (xx, yy) = if horizontal {
                            ((x as isize + o).clamp(0, src.w as isize - 1) as usize, y)
                        } else {
                            (x, (y as isize + o).clamp(0, src.h as isize - 1) as usize)
                        };
                        acc += kv * src.at(xx, yy);
                    }
                    out.data[y * src.w + x] = acc;
                }
            }
            out
        };
        pass(&pass(self, true), false)
    }

    pub fn normalized(&self) -> Grid {
        let (lo, hi) = self
            .data
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        let span = (hi - lo).max(1e-12);
        Grid {
            w: self.w,
            h: self.h,
            data: self.data.iter().map(|v| (v - lo) / span).collect(),
        }
    }

    /// Central-difference gradient (d/dx, d/dy), one-sided at the borders.
    pub fn gradient(&self) -> (Grid, Grid) {
        let mut gx = Grid::new(self.w, self.h, 0.0);
        let mut gy = Grid::new(self.w, self.h, 0.0);
        for y in 0..self.h {
            for x in 0..self.w {
                let (xl, xr) = (x.saturating_sub(1), (x + 1).min(self.w - 1));
                let (yu, yd) = (y.saturating_sub(1), (y + 1).min(self.h - 1));
                gx.data[y * self.w + x] =
                    (self.at(xr, y) - self.at(xl, y)) / (xr - xl).max(1) as f32;
                gy.data[y * self.w + x] =
                    (self.at(x, yd) - self.at(x, yu)) / (yd - yu).max(1) as f32;
            }
        }
        (gx, gy)
    }
}

pub fn quantile(values: &[f32], q: f32) -> f32 {
    let mut v = values.to_vec();
    let i = ((v.len() - 1) as f32 * q).round() as usize;
    let (_, m, _) = v.select_nth_unstable_by(i, |a, b| a.total_cmp(b));
    *m
}
