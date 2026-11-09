mod depth;
mod geometry;
mod grid;

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde::Serialize;

use geometry::{Estimate, Params};

/// Rectify oblique aerial images to a nadir view.
///
/// Level 1 corrects the ground plane (affine stretch by 1/cos(tilt) along the
/// tilt direction). Level 2 additionally moves above-ground pixels back to
/// their footprint using monocular depth. Tilt and azimuth are estimated per
/// image unless given.
#[derive(Parser)]
#[command(version, about)]
struct Args {
    /// Input images (PNG or JPEG).
    #[arg(required = true)]
    inputs: Vec<PathBuf>,

    /// Output directory.
    #[arg(short, long, default_value = "rectified")]
    out_dir: PathBuf,

    /// 1 = ground-plane rectification, 2 = with relief-displacement correction.
    #[arg(short, long, default_value_t = 2, value_parser = clap::value_parser!(u8).range(1..=2))]
    level: u8,

    /// Depth Anything V2 ONNX model.
    #[arg(
        short,
        long,
        env = "LOCUS_ORTHO_MODEL",
        default_value = "models/depth_anything_v2_small.onnx"
    )]
    model: String,

    /// Depth map(s) to use instead of the model: a 16-bit PNG (larger = closer) for a
    /// single input, or a directory with one `<input stem>.png` per input.
    #[arg(long)]
    depth: Option<PathBuf>,

    /// Surface-normal map(s) for tilt estimation: a 16-bit RGB PNG (see
    /// scripts/normals.py) for a single input, or a directory of `<input stem>.png`.
    #[arg(long)]
    normals: Option<PathBuf>,

    /// Tilt from nadir in degrees (skips estimation).
    #[arg(long)]
    tilt: Option<f32>,

    /// Direction in which the ground recedes, degrees clockwise from image up (skips estimation).
    #[arg(long)]
    azimuth: Option<f32>,

    /// Maximum relief shift as a fraction of image height (level 2).
    #[arg(long, default_value_t = 0.25)]
    max_shift: f32,

    /// Minimum facade fraction needed to estimate a non-zero tilt.
    #[arg(long, default_value_t = 0.005)]
    min_facade: f32,

    /// Write per-image estimates to this JSON file.
    #[arg(long)]
    report: Option<PathBuf>,
}

#[derive(Serialize)]
struct Row {
    input: String,
    output: String,
    #[serde(flatten)]
    estimate: Estimate,
    hole_frac: Option<f32>,
}

/// A file is used as is; for a directory, take `<input stem>.png` inside it.
fn companion(p: &Path, input: &Path) -> String {
    if p.is_dir() {
        p.join(Path::new(input.file_stem().unwrap()).with_extension("png"))
            .display()
            .to_string()
    } else {
        p.display().to_string()
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    for (flag, p) in [("--depth", &args.depth), ("--normals", &args.normals)] {
        if let Some(p) = p {
            if p.is_file() && args.inputs.len() != 1 {
                bail!(
                    "{flag} is a file but there are {} inputs; pass a directory",
                    args.inputs.len()
                );
            }
        }
    }
    std::fs::create_dir_all(&args.out_dir)?;
    let mut model = match args.depth {
        Some(_) => None,
        None => Some(depth::DepthModel::load(&args.model)?),
    };
    let params = Params {
        min_facade: args.min_facade,
        max_shift: args.max_shift,
    };
    let mut rows = Vec::new();
    for input in &args.inputs {
        let img = image::open(input)
            .with_context(|| format!("cannot read {}", input.display()))?
            .into_rgb8();
        let mut d = match (&args.depth, model.as_mut()) {
            (Some(p), _) => depth::load_depth(&companion(p, input), img.width(), img.height())?,
            (None, Some(m)) => m.predict(&img)?,
            _ => unreachable!(),
        };
        let normals = match &args.normals {
            Some(p) => Some(depth::load_normals(
                &companion(p, input),
                img.width(),
                img.height(),
            )?),
            None => None,
        };
        let est = geometry::estimate(&mut d, normals.as_ref(), &params, args.tilt, args.azimuth);
        let (out, holes) = match args.level {
            1 => (geometry::level1(&img, &est), None),
            _ => {
                let (o, f) = geometry::level2(&img, &d, &est, &params);
                (o, Some(f))
            }
        };
        let name = Path::new(input.file_name().unwrap()).with_extension("png");
        let path = args.out_dir.join(name);
        out.save(&path)?;
        println!(
            "{}  [{}]  tilt {:5.1}°  azimuth {:6.1}°  facade {:.3}{}",
            input.display(),
            serde_json::to_value(est.source)?.as_str().unwrap_or(""),
            est.tilt_deg,
            est.azimuth_deg,
            est.facade_frac,
            holes.map(|h| format!("  holes {h:.3}")).unwrap_or_default()
        );
        rows.push(Row {
            input: input.display().to_string(),
            output: path.display().to_string(),
            estimate: est,
            hole_frac: holes,
        });
    }
    if let Some(p) = args.report {
        std::fs::write(&p, serde_json::to_string_pretty(&rows)?)?;
    }
    Ok(())
}
