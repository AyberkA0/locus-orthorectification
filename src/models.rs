//! ONNX models (Depth Anything V2, LaMa) and loaders for precomputed inputs.

use anyhow::{Context, Result};
use image::{imageops::FilterType, RgbImage};
use ort::session::Session;
use ort::value::Tensor;

use crate::grid::{Grid, Mask};
use crate::warp::Rgb;

fn session(path: &str, env: &ort::environment::Environment, what: &str) -> Result<Session> {
    // CPU only: on macOS the default policy compiles the model for CoreML, which is
    // slow to load and runs in reduced precision
    Session::builder(env)?
        .with_auto_device(ort::session::builder::AutoDevicePolicy::PreferCPU)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .commit_from_file(path)
        .with_context(|| format!("cannot load {what} model {path}"))
}

pub struct Models {
    env: ort::environment::Environment,
    depth: Option<Session>,
    lama: Option<Session>,
}

impl Models {
    pub fn new() -> Result<Self> {
        Ok(Models {
            env: ort::init().with_name("locus-ortho").build()?,
            depth: None,
            lama: None,
        })
    }

    pub fn with_depth(mut self, path: &str) -> Result<Self> {
        self.depth = Some(session(path, &self.env, "depth")?);
        Ok(self)
    }

    pub fn with_lama(mut self, path: &str) -> Result<Self> {
        self.lama = Some(session(path, &self.env, "inpainting")?);
        Ok(self)
    }

    pub fn has_lama(&self) -> bool {
        self.lama.is_some()
    }

    /// Relative depth with Depth Anything V2; larger = closer.
    pub fn depth(&mut self, img: &RgbImage) -> Result<Grid> {
        const SIDE: u32 = 518;
        const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
        const STD: [f32; 3] = [0.229, 0.224, 0.225];
        let s = self.depth.as_mut().context("no depth model loaded")?;
        let x = image::imageops::resize(img, SIDE, SIDE, FilterType::CatmullRom);
        let n = (SIDE * SIDE) as usize;
        let mut data = vec![0f32; 3 * n];
        for (i, p) in x.pixels().enumerate() {
            for c in 0..3 {
                data[c * n + i] = (p[c] as f32 / 255.0 - MEAN[c]) / STD[c];
            }
        }
        let input = Tensor::from_array(([1usize, 3, SIDE as usize, SIDE as usize], data))?;
        let out = s.run(ort::inputs!["pixel_values" => input])?;
        let (shape, v) = out["predicted_depth"].try_extract_tensor::<f32>()?;
        let g = Grid {
            w: shape[2] as usize,
            h: shape[1] as usize,
            data: v.to_vec(),
        };
        Ok(g.resize(img.width() as usize, img.height() as usize))
    }

    /// Inpaint the pixels of `hole` with LaMa (run at 512 x 512); other pixels are kept.
    pub fn inpaint(&mut self, img: &Rgb, hole: &Mask) -> Result<Rgb> {
        const SIDE: usize = 512;
        let s = self.lama.as_mut().context("no inpainting model loaded")?;
        let (w, h) = (img[0].w, img[0].h);
        let small: Vec<Grid> = img.iter().map(|g| g.resize(SIDE, SIDE)).collect();
        let hm = Grid {
            w,
            h,
            data: hole.data.iter().map(|&b| b as u8 as f32).collect(),
        }
        .resize(SIDE, SIDE);
        let n = SIDE * SIDE;
        let mut x = vec![0f32; 3 * n];
        for c in 0..3 {
            for i in 0..n {
                x[c * n + i] = (small[c].data[i] / 255.0).clamp(0.0, 1.0);
            }
        }
        let m: Vec<f32> = hm.data.iter().map(|&v| (v > 0.0) as u8 as f32).collect();
        let out = s.run(ort::inputs![
            "image" => Tensor::from_array(([1usize, 3, SIDE, SIDE], x))?,
            "mask" => Tensor::from_array(([1usize, 1, SIDE, SIDE], m))?
        ])?;
        let (_, y) = out["output"].try_extract_tensor::<f32>()?;
        let k = if y.iter().fold(0f32, |a, &b| a.max(b)) <= 1.5 {
            255.0
        } else {
            1.0
        };
        Ok(std::array::from_fn(|c| {
            let g = Grid {
                w: SIDE,
                h: SIDE,
                data: y[c * n..(c + 1) * n].iter().map(|v| v * k).collect(),
            }
            .resize(w, h);
            let mut o = img[c].clone();
            for i in 0..w * h {
                if hole.data[i] {
                    o.data[i] = g.data[i].clamp(0.0, 255.0);
                }
            }
            o
        }))
    }
}

fn fit(g: Grid, w: u32, h: u32) -> Grid {
    g.resize(w as usize, h as usize)
}

/// Precomputed depth map (8- or 16-bit grayscale, larger = closer).
pub fn load_depth(path: &str, w: u32, h: u32) -> Result<Grid> {
    let im = image::open(path)
        .with_context(|| format!("cannot read depth {path}"))?
        .into_luma16();
    let g = Grid {
        w: im.width() as usize,
        h: im.height() as usize,
        data: im.pixels().map(|p| p[0] as f32 / 65535.0).collect(),
    };
    Ok(fit(g, w, h))
}

/// Unit normals from a 16-bit RGB PNG, n = v / 65535 · 2 − 1, camera coordinates
/// (x right, y up, z towards the camera).
pub fn load_normals(path: &str, w: u32, h: u32) -> Result<Vec<[f32; 3]>> {
    let im = image::open(path)
        .with_context(|| format!("cannot read normals {path}"))?
        .into_rgb16();
    let (iw, ih) = (im.width() as usize, im.height() as usize);
    let ch: Vec<Grid> = (0..3)
        .map(|c| {
            fit(
                Grid {
                    w: iw,
                    h: ih,
                    data: im
                        .pixels()
                        .map(|p| p[c] as f32 / 65535.0 * 2.0 - 1.0)
                        .collect(),
                },
                w,
                h,
            )
        })
        .collect();
    Ok((0..(w * h) as usize)
        .map(|i| {
            let v = [ch[0].data[i], ch[1].data[i], ch[2].data[i]];
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-12);
            [v[0] / l, v[1] / l, v[2] / l]
        })
        .collect())
}

/// Segment label map (16-bit grayscale PNG, 0 = no segment), nearest-neighbour resized.
pub fn load_segments(path: &str, w: u32, h: u32) -> Result<Vec<u32>> {
    let im = image::open(path)
        .with_context(|| format!("cannot read segments {path}"))?
        .into_luma16();
    let im = if im.width() == w && im.height() == h {
        im
    } else {
        image::imageops::resize(&im, w, h, FilterType::Nearest)
    };
    Ok(im.pixels().map(|p| p[0] as u32).collect())
}

pub fn to_planes(img: &RgbImage) -> Rgb {
    let (w, h) = (img.width() as usize, img.height() as usize);
    std::array::from_fn(|c| Grid {
        w,
        h,
        data: img.pixels().map(|p| p[c] as f32).collect(),
    })
}

pub fn to_image(p: &Rgb) -> RgbImage {
    let w = p[0].w;
    RgbImage::from_fn(p[0].w as u32, p[0].h as u32, |x, y| {
        let i = y as usize * w + x as usize;
        image::Rgb(std::array::from_fn(|c| {
            p[c].data[i].round().clamp(0.0, 255.0) as u8
        }))
    })
}
