use anyhow::{Context, Result};
use image::{imageops::FilterType, RgbImage};
use ort::session::Session;
use ort::value::Tensor;

use crate::grid::Grid;

const SIDE: u32 = 518;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Monocular relative depth with Depth Anything V2 (ONNX). Output: larger = closer.
pub struct DepthModel {
    session: Session,
    _env: ort::environment::Environment,
}

impl DepthModel {
    pub fn load(path: &str) -> Result<Self> {
        let env = ort::init().with_name("locus-ortho").build()?;
        let session = Session::builder(&env)?
            .commit_from_file(path)
            .with_context(|| format!("cannot load depth model {path}"))?;
        Ok(DepthModel { session, _env: env })
    }

    pub fn predict(&mut self, img: &RgbImage) -> Result<Grid> {
        let x = image::imageops::resize(img, SIDE, SIDE, FilterType::CatmullRom);
        let n = (SIDE * SIDE) as usize;
        let mut data = vec![0f32; 3 * n];
        for (i, p) in x.pixels().enumerate() {
            for c in 0..3 {
                data[c * n + i] = (p[c] as f32 / 255.0 - MEAN[c]) / STD[c];
            }
        }
        let input = Tensor::from_array(([1usize, 3, SIDE as usize, SIDE as usize], data))?;
        let outputs = self.session.run(ort::inputs!["pixel_values" => input])?;
        let (shape, out) = outputs["predicted_depth"].try_extract_tensor::<f32>()?;
        let (h, w) = (shape[1] as usize, shape[2] as usize);
        let g = Grid {
            w,
            h,
            data: out.to_vec(),
        };
        Ok(g.resize(img.width() as usize, img.height() as usize))
    }
}

/// Load a precomputed depth map (8- or 16-bit grayscale, larger = closer).
pub fn load_depth(path: &str, w: u32, h: u32) -> Result<Grid> {
    let im = image::open(path)
        .with_context(|| format!("cannot read depth {path}"))?
        .into_luma16();
    let g = Grid {
        w: im.width() as usize,
        h: im.height() as usize,
        data: im.pixels().map(|p| p[0] as f32 / 65535.0).collect(),
    };
    Ok(if g.w == w as usize && g.h == h as usize {
        g
    } else {
        g.resize(w as usize, h as usize)
    })
}

/// Load a surface-normal map stored as a 16-bit RGB PNG with n = v / 65535 · 2 − 1,
/// in camera coordinates (x right, y up, z towards the camera).
pub fn load_normals(path: &str, w: u32, h: u32) -> Result<crate::geometry::Normals> {
    let im = image::open(path)
        .with_context(|| format!("cannot read normals {path}"))?
        .into_rgb16();
    let (iw, ih) = (im.width() as usize, im.height() as usize);
    let ch: Vec<Grid> = (0..3)
        .map(|c| Grid {
            w: iw,
            h: ih,
            data: im
                .pixels()
                .map(|p| p[c] as f32 / 65535.0 * 2.0 - 1.0)
                .collect(),
        })
        .collect();
    let ch: Vec<Grid> = ch
        .into_iter()
        .map(|g| {
            if g.w == w as usize && g.h == h as usize {
                g
            } else {
                g.resize(w as usize, h as usize)
            }
        })
        .collect();
    let n = (0..(w * h) as usize)
        .map(|i| {
            let v = [ch[0].data[i], ch[1].data[i], ch[2].data[i]];
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-12);
            [v[0] / l, v[1] / l, v[2] / l]
        })
        .collect();
    Ok(crate::geometry::Normals { n })
}
