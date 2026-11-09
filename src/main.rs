mod grid;
mod models;
mod relief;
mod tilt;
mod warp;

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde::Serialize;

use models::Models;
use relief::Info;
use tilt::Estimate;

/// Rectify oblique aerial images to a nadir view.
///
/// Level 1 corrects the ground plane (stretch by 1/cos(tilt) along the tilt
/// direction). Level 2 also moves roofs and tree crowns back to their
/// footprints, removes facades and inpaints the ground they hid.
#[derive(Parser)]
#[command(version, about)]
struct Args {
    /// Input images (PNG or JPEG).
    #[arg(required = true)]
    inputs: Vec<PathBuf>,

    /// Output directory.
    #[arg(short, long, default_value = "rectified")]
    out_dir: PathBuf,

    /// 1 = ground plane only, 2 = with relief correction.
    #[arg(short, long, default_value_t = 2, value_parser = clap::value_parser!(u8).range(1..=2))]
    level: u8,

    /// Surface normals: a 16-bit RGB PNG (scripts/normals.py) for a single input,
    /// or a directory with one `<input stem>.png` per input.
    #[arg(long)]
    normals: Option<PathBuf>,

    /// Segment label maps (16-bit PNG, scripts/segments.py), file or directory.
    #[arg(long)]
    segments: Option<PathBuf>,

    /// Depth maps to use instead of the depth model (16-bit PNG, larger = closer).
    #[arg(long)]
    depth: Option<PathBuf>,

    /// Depth Anything V2 ONNX model.
    #[arg(
        short,
        long,
        env = "LOCUS_ORTHO_MODEL",
        default_value = "models/depth_anything_v2_small.onnx"
    )]
    model: String,

    /// LaMa ONNX model for the ground hidden behind buildings; without it,
    /// holes are filled by smooth interpolation.
    #[arg(
        long,
        env = "LOCUS_ORTHO_INPAINT",
        default_value = "models/lama_fp32.onnx"
    )]
    inpaint: String,

    /// Tilt from nadir in degrees (skips estimation).
    #[arg(long)]
    tilt: Option<f32>,

    /// Direction in which the ground recedes, degrees clockwise from image up.
    #[arg(long)]
    azimuth: Option<f32>,

    /// Write per-image estimates to this JSON file.
    #[arg(long)]
    report: Option<PathBuf>,
}

#[derive(Serialize)]
struct Row {
    input: String,
    output: String,
    level: u8,
    #[serde(flatten)]
    estimate: Estimate,
    #[serde(skip_serializing_if = "Option::is_none")]
    relief: Option<Info>,
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
    for (flag, p) in [
        ("--depth", &args.depth),
        ("--normals", &args.normals),
        ("--segments", &args.segments),
    ] {
        if let Some(p) = p {
            if p.is_file() && args.inputs.len() != 1 {
                bail!(
                    "{flag} is a file but there are {} inputs; pass a directory",
                    args.inputs.len()
                );
            }
        }
    }
    let manual = args.tilt.is_some() && args.azimuth.is_some();
    if args.normals.is_none() && (args.level == 2 || !manual) {
        bail!("--normals is required (level 1 without normals needs --tilt and --azimuth)");
    }
    std::fs::create_dir_all(&args.out_dir)?;
    let mut models = Models::new()?;
    if args.level == 2 {
        if args.depth.is_none() {
            models = models.with_depth(&args.model)?;
        }
        if Path::new(&args.inpaint).is_file() {
            models = models.with_lama(&args.inpaint)?;
        } else {
            eprintln!("note: {} not found, holes are interpolated", args.inpaint);
        }
    }
    let mut rows = Vec::new();
    for input in &args.inputs {
        let rgb = image::open(input)
            .with_context(|| format!("cannot read {}", input.display()))?
            .into_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        let img = models::to_planes(&rgb);
        let normals = match &args.normals {
            Some(p) => Some(models::load_normals(&companion(p, input), w, h)?),
            None => None,
        };
        let est = tilt::estimate(&img, normals.as_ref(), args.tilt, args.azimuth);
        let (out, info) = if args.level == 1 || est.tilt_deg < relief::MIN_TILT {
            (relief::level1(&img, &est), None)
        } else {
            let depth = match &args.depth {
                Some(p) => models::load_depth(&companion(p, input), w, h)?,
                None => models.depth(&rgb)?,
            };
            let segments = match &args.segments {
                Some(p) => Some(models::load_segments(&companion(p, input), w, h)?),
                None => None,
            };
            let (o, i) = relief::level2(
                &img,
                normals.as_ref().unwrap(),
                &depth,
                segments.as_deref(),
                &est,
                &mut models,
            )?;
            (o, Some(i))
        };
        let path = args
            .out_dir
            .join(Path::new(input.file_name().unwrap()).with_extension("png"));
        models::to_image(&out).save(&path)?;
        println!(
            "{}  [{}]  tilt {:5.1}°  azimuth {:6.1}°{}",
            input.display(),
            serde_json::to_value(est.source)?.as_str().unwrap_or(""),
            est.tilt_deg,
            est.azimuth_deg,
            info.as_ref()
                .map(|i| format!(
                    "  roofs {}  trees {}  holes {:.3}",
                    i.roofs, i.trees, i.hole_frac
                ))
                .unwrap_or_default()
        );
        rows.push(Row {
            input: input.display().to_string(),
            output: path.display().to_string(),
            level: if info.is_some() { 2 } else { 1 },
            estimate: est,
            relief: info,
        });
    }
    if let Some(p) = args.report {
        std::fs::write(&p, serde_json::to_string_pretty(&rows)?)?;
    }
    Ok(())
}
