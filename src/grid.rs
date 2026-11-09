//! Dense 2-D float grids and binary masks with the few filters the method needs.

#[derive(Clone)]
pub struct Grid {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

/// Mirror an index into 0..n ("dcba|abcd|dcba", the scipy.ndimage default).
fn reflect(i: isize, n: usize) -> usize {
    let n = n as isize;
    let m = 2 * n;
    let mut i = i.rem_euclid(m);
    if i >= n {
        i = m - 1 - i;
    }
    i as usize
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
        if w == self.w && h == self.h {
            return self.clone();
        }
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

    /// Separable filter with a symmetric kernel, reflected borders.
    fn separable(&self, k: &[f32]) -> Grid {
        let r = (k.len() / 2) as isize;
        let pass = |src: &Grid, horizontal: bool| {
            let mut out = Grid::new(src.w, src.h, 0.0);
            for y in 0..src.h {
                for x in 0..src.w {
                    let mut acc = 0.0;
                    for (j, kv) in k.iter().enumerate() {
                        let o = j as isize - r;
                        acc += kv
                            * if horizontal {
                                src.at(reflect(x as isize + o, src.w), y)
                            } else {
                                src.at(x, reflect(y as isize + o, src.h))
                            };
                    }
                    out.data[y * src.w + x] = acc;
                }
            }
            out
        };
        pass(&pass(self, true), false)
    }

    pub fn gaussian(&self, sigma: f32) -> Grid {
        let r = (4.0 * sigma).round() as isize;
        let k: Vec<f32> = (-r..=r)
            .map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp())
            .collect();
        let s: f32 = k.iter().sum();
        self.separable(&k.iter().map(|v| v / s).collect::<Vec<_>>())
    }

    /// Mean over a `size` x `size` window (odd size).
    pub fn box_mean(&self, size: usize) -> Grid {
        self.separable(&vec![1.0 / size as f32; size])
    }

    /// Median over a `size` x `size` window (odd size).
    pub fn median(&self, size: usize) -> Grid {
        let r = (size / 2) as isize;
        let mut out = Grid::new(self.w, self.h, 0.0);
        let mut buf = Vec::with_capacity(size * size);
        for y in 0..self.h {
            for x in 0..self.w {
                buf.clear();
                for dy in -r..=r {
                    for dx in -r..=r {
                        buf.push(self.at(
                            reflect(x as isize + dx, self.w),
                            reflect(y as isize + dy, self.h),
                        ));
                    }
                }
                let mid = buf.len() / 2;
                let (_, m, _) = buf.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
                out.data[y * self.w + x] = *m;
            }
        }
        out
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
    if values.is_empty() {
        return 0.0;
    }
    let mut v = values.to_vec();
    let i = ((v.len() - 1) as f32 * q).round() as usize;
    let (_, m, _) = v.select_nth_unstable_by(i, |a, b| a.total_cmp(b));
    *m
}

pub fn median(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Binary image, row-major.
#[derive(Clone)]
pub struct Mask {
    pub w: usize,
    pub h: usize,
    pub data: Vec<bool>,
}

const CROSS: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

impl Mask {
    pub fn new(w: usize, h: usize, v: bool) -> Self {
        Mask {
            w,
            h,
            data: vec![v; w * h],
        }
    }

    pub fn count(&self) -> usize {
        self.data.iter().filter(|&&v| v).count()
    }

    fn neighbours(&self, i: usize) -> impl Iterator<Item = Option<usize>> + '_ {
        let (x, y) = ((i % self.w) as isize, (i / self.w) as isize);
        CROSS.iter().map(move |(dx, dy)| {
            let (nx, ny) = (x + dx, y + dy);
            (nx >= 0 && ny >= 0 && (nx as usize) < self.w && (ny as usize) < self.h)
                .then(|| ny as usize * self.w + nx as usize)
        })
    }

    /// Erosion with a 3x3 cross; pixels outside the image count as unset.
    pub fn erode(&self) -> Mask {
        let data = (0..self.data.len())
            .map(|i| self.data[i] && self.neighbours(i).all(|n| n.is_some_and(|j| self.data[j])))
            .collect();
        Mask { data, ..*self }
    }

    /// Dilation with a 3x3 cross.
    pub fn dilate(&self) -> Mask {
        let data = (0..self.data.len())
            .map(|i| self.data[i] || self.neighbours(i).any(|n| n.is_some_and(|j| self.data[j])))
            .collect();
        Mask { data, ..*self }
    }

    pub fn open(&self, iterations: usize) -> Mask {
        let mut m = self.clone();
        for _ in 0..iterations {
            m = m.erode();
        }
        for _ in 0..iterations {
            m = m.dilate();
        }
        m
    }

    pub fn and_not(&self, other: &Mask) -> Mask {
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| *a && !b)
            .collect();
        Mask { data, ..*self }
    }

    /// 4-connected components; 0 = background, labels 1..=n.
    pub fn label(&self) -> (Vec<u32>, usize) {
        let mut lab = vec![0u32; self.data.len()];
        let mut n = 0;
        let mut stack = Vec::new();
        for s in 0..self.data.len() {
            if !self.data[s] || lab[s] != 0 {
                continue;
            }
            n += 1;
            lab[s] = n as u32;
            stack.push(s);
            while let Some(i) = stack.pop() {
                for j in self.neighbours(i).flatten() {
                    if self.data[j] && lab[j] == 0 {
                        lab[j] = n as u32;
                        stack.push(j);
                    }
                }
            }
        }
        (lab, n)
    }
}

/// For every pixel, the label of the nearest labelled pixel (multi-source BFS, 8-neighbourhood).
pub fn nearest_label(lab: &[u32], w: usize, h: usize) -> Vec<u32> {
    let mut out = lab.to_vec();
    let mut queue: std::collections::VecDeque<usize> =
        (0..lab.len()).filter(|&i| lab[i] != 0).collect();
    while let Some(i) = queue.pop_front() {
        let (x, y) = ((i % w) as isize, (i / w) as isize);
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (nx, ny) = (x + dx, y + dy);
                if nx < 0 || ny < 0 || nx as usize >= w || ny as usize >= h {
                    continue;
                }
                let j = ny as usize * w + nx as usize;
                if out[j] == 0 {
                    out[j] = out[i];
                    queue.push_back(j);
                }
            }
        }
    }
    out
}

/// Canny edges on a grayscale image in 0..255 (3x3 Sobel, L1 magnitude, no pre-blur).
pub fn canny(gray: &Grid, lo: f32, hi: f32) -> Mask {
    let (w, h) = (gray.w, gray.h);
    let px = |x: isize, y: isize| gray.at(reflect(x, w), reflect(y, h));
    let mut gx = vec![0f32; w * h];
    let mut gy = vec![0f32; w * h];
    let mut mag = vec![0f32; w * h];
    for y in 0..h as isize {
        for x in 0..w as isize {
            let i = y as usize * w + x as usize;
            gx[i] = px(x + 1, y - 1) + 2.0 * px(x + 1, y) + px(x + 1, y + 1)
                - px(x - 1, y - 1)
                - 2.0 * px(x - 1, y)
                - px(x - 1, y + 1);
            gy[i] = px(x - 1, y + 1) + 2.0 * px(x, y + 1) + px(x + 1, y + 1)
                - px(x - 1, y - 1)
                - 2.0 * px(x, y - 1)
                - px(x + 1, y - 1);
            mag[i] = gx[i].abs() + gy[i].abs();
        }
    }
    let m = |x: isize, y: isize| {
        if x < 0 || y < 0 || x as usize >= w || y as usize >= h {
            0.0
        } else {
            mag[y as usize * w + x as usize]
        }
    };
    // 0 = suppressed, 1 = weak, 2 = strong
    let mut state = vec![0u8; w * h];
    let tan22 = 22.5f32.to_radians().tan();
    for y in 0..h as isize {
        for x in 0..w as isize {
            let i = y as usize * w + x as usize;
            let v = mag[i];
            if v <= lo {
                continue;
            }
            let (ax, ay) = (gx[i].abs(), gy[i].abs());
            let (a, b) = if ay <= ax * tan22 {
                (m(x - 1, y), m(x + 1, y))
            } else if ay >= ax / tan22 {
                (m(x, y - 1), m(x, y + 1))
            } else if (gx[i] > 0.0) == (gy[i] > 0.0) {
                (m(x - 1, y - 1), m(x + 1, y + 1))
            } else {
                (m(x + 1, y - 1), m(x - 1, y + 1))
            };
            if v > a && v >= b {
                state[i] = if v > hi { 2 } else { 1 };
            }
        }
    }
    let mut stack: Vec<usize> = (0..w * h).filter(|&i| state[i] == 2).collect();
    while let Some(i) = stack.pop() {
        let (x, y) = ((i % w) as isize, (i / w) as isize);
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (nx, ny) = (x + dx, y + dy);
                if nx < 0 || ny < 0 || nx as usize >= w || ny as usize >= h {
                    continue;
                }
                let j = ny as usize * w + nx as usize;
                if state[j] == 1 {
                    state[j] = 2;
                    stack.push(j);
                }
            }
        }
    }
    Mask {
        w,
        h,
        data: state.iter().map(|&s| s == 2).collect(),
    }
}
