//! # Which function runs at which resolution
//!
//! | function              | runs at          | cost                          |
//! |-----------------------|------------------|-------------------------------|
//! | `mask_small`          | 320x320          | model run + cheap             |
//! | `solid_mask_small`    | 320x320          | cheap                         |
//! | `instance_cutout`     | 320x320, then a  | cheap: scales with subject    |
//! |                       | crop per subject | size, not photo size          |
//! | `mask_full`           | full photo       | slow: Lanczos3 upscale        |
//! | `cutout_full`         | full photo       | slow                          |
//! | `solid_mask_full`     | full photo       | slowest: Otsu/close/fill on   |
//! |                       |                  | every pixel                   |
//!
//! The old names `mask`, `cutout`, `solid_mask` still work but are deprecated, so
//! the compiler flags every call site that is on a slow path.

pub mod decode;
mod resize;
pub mod variables;

pub use decode::{open_image, open_image_with};

use image::{DynamicImage, GrayImage, Luma, Rgb, RgbImage, Rgba, RgbaImage};
use imageproc::{
    contrast::{otsu_level, threshold},
    distance_transform::Norm,
    morphology::{close, dilate, erode},
    region_labelling::{Connectivity, connected_components},
};
use ort::{environment::Environment, inputs, session::Session, value::Value};
use resize::Filter;
use std::{collections::HashSet, path::Path};

use variables::*;

// U2Net constant, DON'T CHANGE THESE

pub const INPUT_SIZE: u32 = 320;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("inference error: {0}")]
    Inference(#[from] ort::Error),
    #[error("model {0:?} not embedded; enable the matching `embed-*` feature or use from_path")]
    NotEmbedded(Model),
    #[error("unexpected model output shape: {0:?}")]
    BadOutput(Vec<usize>),
}
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Model {
    U2Net,
    U2NetP,
}

#[allow(dead_code)]
pub struct Segmenter {
    model: Model,
    session: Session,
}

/// Bytes of the embedded model, or `NotEmbedded` if its `embed-*` feature is off.
fn embedded_bytes(model: Model) -> Result<&'static [u8]> {
    match model {
        #[cfg(feature = "embed-u2net")]
        Model::U2Net => Ok(include_bytes!("../models/u2net.onnx") as &[u8]),
        #[cfg(feature = "embed-u2netp")]
        Model::U2NetP => Ok(include_bytes!("../models/u2netp.onnx") as &[u8]),
        #[allow(unreachable_patterns)]
        _ => Err(Error::NotEmbedded(model)),
    }
}

impl Segmenter {
    fn default_env() -> Result<Environment> {
        Ok(ort::init().build()?)
    }

    // ---- from a file path ------------------------------------------------

    pub fn from_path(model: Model, path: impl AsRef<Path>) -> Result<Self> {
        Self::from_path_with_env(model, &Self::default_env()?, path)
    }

    pub fn from_path_with_env(
        model: Model,
        env: &Environment,
        path: impl AsRef<Path>,
    ) -> Result<Self> {
        let session = Session::builder(env)?.commit_from_file(path)?;
        Ok(Self { model, session })
    }

    // ---- from raw bytes --------------------------------------------------

    pub fn from_bytes(model: Model, bytes: &[u8]) -> Result<Self> {
        Self::from_bytes_with_env(model, &Self::default_env()?, bytes)
    }

    pub fn from_bytes_with_env(model: Model, env: &Environment, bytes: &[u8]) -> Result<Self> {
        let session = Session::builder(env)?.commit_from_memory(bytes)?;
        Ok(Self { model, session })
    }

    // ---- from the embedded model (needs the matching `embed-*` feature) --

    pub fn embedded(model: Model) -> Result<Self> {
        Self::embedded_with_env(model, &Self::default_env()?)
    }

    pub fn embedded_with_env(model: Model, env: &Environment) -> Result<Self> {
        Self::from_bytes_with_env(model, env, embedded_bytes(model)?)
    }

    // ---- custom session --------------------------------------------------

    pub fn new_with_env(
        model: Model,
        env: &Environment,
        build_fn: impl FnOnce(&Environment) -> ort::Result<Session>,
    ) -> Result<Self> {
        Ok(Self {
            model,
            session: build_fn(env)?,
        })
    }

    pub fn model(&self) -> Model {
        self.model
    }

    #[cfg(any(feature = "embed-u2net", feature = "embed-u2netp"))]
    pub fn new(model: Model) -> Result<Self> {
        match model {
            #[cfg(feature = "embed-u2net")]
            Model::U2Net => Self::from_bytes(Model::U2Net, include_bytes!("../models/u2net.onnx")),
            #[cfg(feature = "embed-u2netp")]
            Model::U2NetP => {
                Self::from_bytes(Model::U2NetP, include_bytes!("../models/u2netp.onnx"))
            }
            #[allow(unreachable_patterns)]
            _ => Err(Error::NotEmbedded(model)),
        }
    }

    /// Raw saliency map at model resolution (INPUT_SIZE x INPUT_SIZE), 0..=255.
    pub fn mask_small(&mut self, img: &DynamicImage) -> Result<GrayImage> {
        let _span = tracing::trace_span!("Segmenter::mask_small").entered();
        let start = std::time::Instant::now();
        // Triangle is far cheaper than Lanczos3 on big photos and plenty for a
        // 320x320 model input. (`fast-resize` feature: SIMD backend.)
        let resize_start = std::time::Instant::now();
        let small = resize::rgb(img, INPUT_SIZE, INPUT_SIZE, Filter::Triangle);
        let resize_ms = resize_start.elapsed().as_secs_f64() * 1000.0;

        let prep_start = std::time::Instant::now();
        let max = small.as_raw().iter().copied().max().unwrap_or(1).max(1) as f32;
        let n = (INPUT_SIZE * INPUT_SIZE) as usize;
        let mut data = vec![0f32; 3 * n];
        for (i, px) in small.pixels().enumerate() {
            for c in 0..3 {
                data[c * n + i] = (px[c] as f32 / max - MEAN[c]) / STD[c];
            }
        }
        let input_tensor =
            Value::from_array((vec![1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize], data))?;
        let prep_ms = prep_start.elapsed().as_secs_f64() * 1000.0;

        let infer_start = std::time::Instant::now();
        let outputs = self.session.run(inputs!["input.1" => input_tensor])?;
        let infer_ms = infer_start.elapsed().as_secs_f64() * 1000.0;

        let post_start = std::time::Instant::now();
        // Extract the first output tensor
        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        let shape_vec = shape.iter().map(|&s| s as usize).collect::<Vec<_>>();

        if data.len() != n {
            return Err(Error::BadOutput(shape_vec));
        }

        let (mn, mx) = data
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        // Flat output = no subject. Without this, the min-max below would stretch
        // noise into a fake subject.
        if mx - mn < MIN_SALIENCY_RANGE {
            return Ok(GrayImage::new(INPUT_SIZE, INPUT_SIZE));
        }
        let range = (mx - mn).max(1e-6);
        let bytes = data
            .iter()
            .map(|&v| (((v - mn) / range) * 255.0).round().clamp(0.0, 255.0) as u8)
            .collect();
        let post_ms = post_start.elapsed().as_secs_f64() * 1000.0;
        let total_ms = start.elapsed().as_secs_f64() * 1000.0;

        tracing::trace!(
            resize_ms = resize_ms,
            prep_ms = prep_ms,
            infer_ms = infer_ms,
            post_ms = post_ms,
            total_ms = total_ms,
            "Segmenter::mask_small detailed breakdown"
        );

        Ok(GrayImage::from_raw(INPUT_SIZE, INPUT_SIZE, bytes).unwrap())
    }

    /// 255 = subject, 0 = background, at the input image's size.
    /// SLOW on big photos: upscales the whole mask to full resolution.
    pub fn mask_full(&mut self, img: &DynamicImage) -> Result<GrayImage> {
        let small_mask = self.mask_small(img)?;
        Ok(resize::gray(
            &small_mask,
            img.width(),
            img.height(),
            Filter::Lanczos3,
        ))
    }

    /// Whole photo with the soft mask as alpha. SLOW on big photos (full-res work).
    pub fn cutout_full(&mut self, img: &DynamicImage) -> Result<RgbaImage> {
        let m = self.mask_full(img)?;
        Ok(apply_mask(img, &m))
    }

    /// Binary (0/255) subject mask with cracks bridged and enclosed holes filled, at
    /// model resolution (320x320). Use this for geometry (connectivity, bounding
    /// boxes), not for soft alpha. Cheap.
    pub fn solid_mask_small(&mut self, img: &DynamicImage) -> Result<GrayImage> {
        let soft = self.mask_small(img)?;
        Ok(solidify(&soft))
    }

    /// Same as `solid_mask_small` but at the photo's resolution. SLOWEST: threshold,
    /// closing and hole filling all run on every pixel. Prefer `solid_mask_small`
    /// (scale coordinates yourself) or `instance_cutout`.
    pub fn solid_mask_full(&mut self, img: &DynamicImage) -> Result<GrayImage> {
        let soft = self.mask_full(img)?;
        Ok(solidify(&soft))
    }

    // ---- deprecated aliases: old names, kept so existing callers still compile ----

    #[deprecated(note = "renamed `mask_full`; runs at full photo resolution (slow). \
                         Prefer `instance_cutout` or `mask_small`.")]
    pub fn mask(&mut self, img: &DynamicImage) -> Result<GrayImage> {
        self.mask_full(img)
    }

    #[deprecated(note = "renamed `cutout_full`; runs at full photo resolution (slow). \
                         Prefer `instance_cutout`.")]
    pub fn cutout(&mut self, img: &DynamicImage) -> Result<RgbaImage> {
        self.cutout_full(img)
    }

    #[deprecated(note = "renamed `solid_mask_full` (slowest). Prefer `solid_mask_small`.")]
    pub fn solid_mask(&mut self, img: &DynamicImage) -> Result<GrayImage> {
        self.solid_mask_full(img)
    }

    /// 1. salient mask (U2Net), kept at model resolution
    /// 2. threshold (Otsu, biased low), close small cracks, fill enclosed holes
    /// 3. split into connected subjects, drop specks under 0.5% of the image
    /// 4. per subject: fill holes, find bbox, crop the *photo* to it, build alpha from
    ///    the model's soft edge (confined to the subject, interior forced opaque),
    ///    upscale only that small crop and tighten the edge ramp
    /// 5. replace colors of semi-transparent edge pixels with nearby interior colors
    ///    (no background halo)
    ///
    /// Everything except the final per-subject crop runs on 320x320 data, so cost
    /// no longer scales with photo resolution x subject count.
    pub fn instance_cutout(&mut self, img: &DynamicImage) -> Result<Vec<RgbaImage>> {
        let (w, h) = (img.width(), img.height());
        let soft = self.mask_small(img)?;
        let solid = solidify(&soft);
        let min_area = (INPUT_SIZE * INPUT_SIZE) / MIN_AREA_DIVISOR;
        let s = INPUT_SIZE as u64;
        let lut = smoothstep_lut(EDGE_LO, EDGE_HI);
        // Approx. width of the soft edge in photo pixels (~2 model pixels).
        let band_px = 2 * w.max(h).div_ceil(INPUT_SIZE);

        let mut out = Vec::new();
        for part in split_instances(&solid, min_area) {
            let part = fill_holes(&part);
            let Some((x0, y0, x1, y1)) = mask_bbox(&part) else {
                continue;
            };
            // Pad by EDGE_RADIUS so the soft edge can fade out inside the crop.
            // (x1, y1) become exclusive bounds here.
            let pad = EDGE_RADIUS as u32;
            let (x0, y0) = (x0.saturating_sub(pad), y0.saturating_sub(pad));
            let (x1, y1) = (
                (x1 + 1 + pad).min(INPUT_SIZE),
                (y1 + 1 + pad).min(INPUT_SIZE),
            );

            // Model-space box -> photo-space box.
            let ox0 = (x0 as u64 * w as u64 / s) as u32;
            let oy0 = (y0 as u64 * h as u64 / s) as u32;
            let ox1 = ((x1 as u64 * w as u64).div_ceil(s) as u32).min(w);
            let oy1 = ((y1 as u64 * h as u64).div_ceil(s) as u32).min(h);
            let (cw, ch) = (ox1 - ox0, oy1 - oy0);
            if cw == 0 || ch == 0 {
                continue;
            }

            let crop = img.crop_imm(ox0, oy0, cw, ch);
            let alpha = instance_alpha(&soft, &part);
            let m = image::imageops::crop_imm(&alpha, x0, y0, x1 - x0, y1 - y0).to_image();
            let mut m = resize::gray(&m, cw, ch, Filter::CatmullRom);
            m.pixels_mut().for_each(|p| p[0] = lut[p[0] as usize]);
            let mut rgba = apply_mask(&crop, &m);
            bleed_edges(&mut rgba, band_px);
            out.push(rgba);
        }
        Ok(out)
    }

    /// `instance_cutout`, then each subject centred on a square `SQUARE_BG` canvas
    /// (RGB, aspect ratio preserved). `side = Some(224)` etc. resizes to a fixed
    /// input size for the extractor; `None` keeps natural size.
    pub fn instance_squares(
        &mut self,
        img: &DynamicImage,
        side: Option<u32>,
    ) -> Result<Vec<RgbImage>> {
        Ok(self
            .instance_cutout(img)?
            .iter()
            .map(|c| square_on(c, SQUARE_BG, side))
            .collect())
    }
}

/// Centre a cutout on a square canvas filled with `bg`, alpha-blending the subject
/// over it (no stretching, no distortion). The result is plain RGB, ready for a feature
/// extractor. `side = Some(n)` also resizes the square to n x n; `None` keeps the
/// natural size (the longer side of the cutout).
pub fn square_on(img: &RgbaImage, bg: [u8; 3], side: Option<u32>) -> RgbImage {
    let (w, h) = img.dimensions();
    let s = w.max(h).max(1);
    let mut out = RgbImage::from_pixel(s, s, Rgb(bg));
    let (ox, oy) = ((s - w) / 2, (s - h) / 2);
    for (x, y, p) in img.enumerate_pixels() {
        let a = p[3] as u32;
        let o = out.get_pixel_mut(x + ox, y + oy);
        for c in 0..3 {
            o[c] = ((p[c] as u32 * a + o[c] as u32 * (255 - a) + 127) / 255) as u8;
        }
    }
    match side {
        Some(t) if t != s && t > 0 => {
            resize::rgb(&DynamicImage::ImageRgb8(out), t, t, Filter::CatmullRom)
        }
        _ => out,
    }
}

/// Soft saliency map -> clean binary mask.
pub fn solidify(soft: &GrayImage) -> GrayImage {
    // Otsu adapts per image; the cap (OTSU_CAP) keeps weakly-salient highlights in.
    let level = otsu_level(soft).min(OTSU_CAP);
    let bin = threshold(soft, level, imageproc::contrast::ThresholdType::Binary);
    // Bridges thin cracks without merging distant subjects (see CLOSE_DIVISOR).
    let radius = ((soft.width().min(soft.height()) / CLOSE_DIVISOR).max(1)).min(255) as u8;
    fill_holes(&close(&bin, Norm::LInf, radius))
}

/// Any background region that doesn't touch the image border is a hole: make it foreground.
pub fn fill_holes(bin: &GrayImage) -> GrayImage {
    let (w, h) = bin.dimensions();
    if w == 0 || h == 0 {
        return bin.clone();
    }
    let mut inv = bin.clone();
    inv.pixels_mut().for_each(|p| p[0] = 255 - p[0]);
    let labels = connected_components(&inv, Connectivity::Four, Luma([0u8]));

    let mut border = HashSet::new();
    for x in 0..w {
        border.insert(labels.get_pixel(x, 0)[0]);
        border.insert(labels.get_pixel(x, h - 1)[0]);
    }
    for y in 0..h {
        border.insert(labels.get_pixel(0, y)[0]);
        border.insert(labels.get_pixel(w - 1, y)[0]);
    }

    let mut out = bin.clone();
    for (x, y, p) in labels.enumerate_pixels() {
        if p[0] != 0 && !border.contains(&p[0]) {
            out.put_pixel(x, y, Luma([255]));
        }
    }
    out
}

/// One full-size 0/255 mask per connected subject (8-connectivity), largest first.
pub fn split_instances(bin: &GrayImage, min_area: u32) -> Vec<GrayImage> {
    let (w, h) = bin.dimensions();
    let labels = connected_components(bin, Connectivity::Eight, Luma([0u8]));
    let n = labels.pixels().map(|p| p[0]).max().unwrap_or(0) as usize;

    let mut area = vec![0u32; n + 1];
    labels.pixels().for_each(|p| area[p[0] as usize] += 1);

    let mut keep: Vec<usize> = (1..=n).filter(|&l| area[l] >= min_area).collect();
    keep.sort_by_key(|&l| std::cmp::Reverse(area[l]));

    // label -> output slot, then a single pass over the image
    let mut slot = vec![usize::MAX; n + 1];
    for (i, &l) in keep.iter().enumerate() {
        slot[l] = i;
    }
    let mut masks: Vec<GrayImage> = keep.iter().map(|_| GrayImage::new(w, h)).collect();
    for (x, y, p) in labels.enumerate_pixels() {
        let s = slot[p[0] as usize];
        if p[0] != 0 && s != usize::MAX {
            masks[s].put_pixel(x, y, Luma([255]));
        }
    }
    masks
}

/// Alpha at model resolution for one subject: the model's own soft edge, confined
/// to that subject (so neighbors/background can't leak in), with the interior forced
/// opaque so low-scoring highlights can't punch holes.
fn instance_alpha(soft: &GrayImage, part: &GrayImage) -> GrayImage {
    let grown = dilate(part, Norm::LInf, EDGE_RADIUS);
    let core = erode(part, Norm::LInf, EDGE_RADIUS);
    let mut out = GrayImage::new(soft.width(), soft.height());
    for (((o, s), g), c) in out
        .pixels_mut()
        .zip(soft.pixels())
        .zip(grown.pixels())
        .zip(core.pixels())
    {
        o[0] = s[0].min(g[0]).max(c[0]);
    }
    out
}

/// Lookup table that tightens a blurry ramp: smoothstep between `lo` and `hi`.
fn smoothstep_lut(lo: f32, hi: f32) -> [u8; 256] {
    let mut lut = [0u8; 256];
    for (i, v) in lut.iter_mut().enumerate() {
        let t = ((i as f32 / 255.0 - lo) / (hi - lo)).clamp(0.0, 1.0);
        *v = (t * t * (3.0 - 2.0 * t) * 255.0).round() as u8;
    }
    lut
}

/// The (in-bounds) 8-neighborhood of a grid cell.
fn neighbors8(gx: u32, gy: u32, gw: u32, gh: u32) -> impl Iterator<Item = (u32, u32)> {
    (-1i32..=1)
        .flat_map(|dy| (-1i32..=1).map(move |dx| (dx, dy)))
        .filter(|&d| d != (0, 0))
        .map(move |(dx, dy)| (gx as i32 + dx, gy as i32 + dy))
        .filter(move |&(nx, ny)| nx >= 0 && ny >= 0 && nx < gw as i32 && ny < gh as i32)
        .map(|(nx, ny)| (nx as u32, ny as u32))
}

/// Replaces the RGB of semi-transparent edge pixels with colors propagated outward
/// from the subject's opaque interior, so nothing of the old background shows as a
/// halo when the cutout is placed on a different background. Alpha is untouched.
///
/// Works on a coarse grid, so cost depends on the edge width rather than the crop
/// size: sample the opaque interior, step one cell inward to dodge contaminated
/// rim pixels, grow that color outward a few cells, then blend it into edge pixels.
/// `band_px` is the approximate edge width in pixels of `img`.
fn bleed_edges(img: &mut RgbaImage, band_px: u32) {
    let (cw, ch) = img.dimensions();
    if cw == 0 || ch == 0 || img.pixels().all(|p| p[3] == 255) {
        return;
    }

    // Cell size: about half the edge width, but never so small the grid gets huge.
    let k = (band_px / 2)
        .max(cw.max(ch).div_ceil(BLEED_GRID_MAX))
        .max(1);
    let (gw, gh) = (cw.div_ceil(k), ch.div_ceil(k));
    let idx = |gx: u32, gy: u32| (gy * gw + gx) as usize;

    // 1. Sample the opaque interior; cell (gx, gy) sits at pixel (gx*k, gy*k).
    let mut color = vec![[0f32; 3]; (gw * gh) as usize];
    let mut opaque = vec![false; (gw * gh) as usize];
    for gy in 0..gh {
        for gx in 0..gw {
            let p = img.get_pixel((gx * k).min(cw - 1), (gy * k).min(ch - 1));
            if p[3] == 255 {
                opaque[idx(gx, gy)] = true;
                color[idx(gx, gy)] = [p[0] as f32, p[1] as f32, p[2] as f32];
            }
        }
    }

    // 2. Trusted sources: opaque cells whose neighbors are all opaque too, since
    //    opaque pixels right at the rim can still carry some background color.
    let mut filled: Vec<bool> = (0..gh)
        .flat_map(|gy| (0..gw).map(move |gx| (gx, gy)))
        .map(|(gx, gy)| {
            opaque[idx(gx, gy)] && neighbors8(gx, gy, gw, gh).all(|(nx, ny)| opaque[idx(nx, ny)])
        })
        .collect();
    if !filled.iter().any(|&f| f) {
        // Thin subject: no cell has a fully opaque neighborhood, so accept the rim.
        filled = opaque;
        if !filled.iter().any(|&f| f) {
            return;
        }
    }

    // 3. Grow colors outward just far enough to cover the edge band.
    let reach_px = band_px * 3 / 2 + 2 * k;
    for _ in 0..reach_px.div_ceil(k) + 1 {
        let mut grown = Vec::new();
        for gy in 0..gh {
            for gx in 0..gw {
                if filled[idx(gx, gy)] {
                    continue;
                }
                let (mut sum, mut n) = ([0f32; 3], 0f32);
                for (nx, ny) in neighbors8(gx, gy, gw, gh) {
                    let j = idx(nx, ny);
                    if filled[j] {
                        for c in 0..3 {
                            sum[c] += color[j][c];
                        }
                        n += 1.0;
                    }
                }
                if n > 0.0 {
                    grown.push((idx(gx, gy), [sum[0] / n, sum[1] / n, sum[2] / n]));
                }
            }
        }
        if grown.is_empty() {
            break;
        }
        for (i, c) in grown {
            color[i] = c;
            filled[i] = true;
        }
    }

    // 4. Blend into edge pixels. Fully transparent pixels are invisible, so skip them.
    for (x, y, p) in img.enumerate_pixels_mut() {
        let a = p[3];
        if a == 255 || a == 0 {
            continue;
        }
        let (fx, fy) = (x as f32 / k as f32, y as f32 / k as f32);
        let (x0, y0) = (fx as u32, fy as u32);
        let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
        let mut acc = [0f32; 3];
        let mut wsum = 0f32;
        for (dx, dy, w) in [
            (0u32, 0u32, (1.0 - tx) * (1.0 - ty)),
            (1, 0, tx * (1.0 - ty)),
            (0, 1, (1.0 - tx) * ty),
            (1, 1, tx * ty),
        ] {
            let i = idx((x0 + dx).min(gw - 1), (y0 + dy).min(gh - 1));
            if w > 0.0 && filled[i] {
                for c in 0..3 {
                    acc[c] += color[i][c] * w;
                }
                wsum += w;
            }
        }
        if wsum <= 0.0 {
            continue;
        }
        let keep = (a as f32 / 255.0).powi(BLEED_KEEP_POW);
        for c in 0..3 {
            let fill = acc[c] / wsum;
            p[c] = (fill * (1.0 - keep) + p[c] as f32 * keep)
                .round()
                .clamp(0.0, 255.0) as u8;
        }
    }
}

/// Inclusive bounding box (x0, y0, x1, y1) of non-zero pixels, if any.
pub fn mask_bbox(mask: &GrayImage) -> Option<(u32, u32, u32, u32)> {
    let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0u32, 0u32);
    for (x, y, p) in mask.enumerate_pixels() {
        if p[0] != 0 {
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
    }
    (x0 != u32::MAX).then_some((x0, y0, x1, y1))
}

pub fn apply_mask(img: &DynamicImage, mask: &GrayImage) -> RgbaImage {
    let mut rgba = img.to_rgba8();
    for (px, m) in rgba.pixels_mut().zip(mask.pixels()) {
        *px = Rgba([px[0], px[1], px[2], m[0]]);
    }
    rgba
}

pub fn crop_to_subject(img: &RgbaImage, threshold: u8) -> Option<RgbaImage> {
    let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0, 0);
    for (x, y, p) in img.enumerate_pixels() {
        if p[3] > threshold {
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
    }
    (x0 != u32::MAX)
        .then(|| image::imageops::crop_imm(img, x0, y0, x1 - x0 + 1, y1 - y0 + 1).to_image())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(w: u32, h: u32, x0: u32, y0: u32, x1: u32, y1: u32) -> GrayImage {
        let mut m = GrayImage::new(w, h);
        for y in y0..y1 {
            for x in x0..x1 {
                m.put_pixel(x, y, Luma([255]));
            }
        }
        m
    }

    #[test]
    fn fills_enclosed_hole() {
        let mut m = square(16, 16, 4, 4, 12, 12);
        m.put_pixel(8, 8, Luma([0])); // highlight hole
        assert_eq!(fill_holes(&m).get_pixel(8, 8)[0], 255);
    }

    #[test]
    fn keeps_border_connected_background() {
        let mut m = square(16, 16, 4, 4, 12, 12);
        // notch open to the outside must stay open
        for y in 4..8 {
            m.put_pixel(8, y, Luma([0]));
        }
        assert_eq!(fill_holes(&m).get_pixel(8, 5)[0], 0);
    }

    #[test]
    fn splits_two_subjects_and_drops_specks() {
        let mut m = square(32, 16, 2, 2, 10, 10);
        for y in 2..10 {
            for x in 20..30 {
                m.put_pixel(x, y, Luma([255]));
            }
        }
        m.put_pixel(15, 14, Luma([255])); // speck
        let parts = split_instances(&m, 4);
        assert_eq!(parts.len(), 2);
    }

    #[test]
    fn alpha_keeps_interior_opaque_and_confines_to_subject() {
        let part = square(16, 16, 4, 4, 12, 12);
        let mut soft = GrayImage::new(16, 16); // model scored everything 0
        soft.put_pixel(2, 2, Luma([255])); // leak far outside the subject
        soft.put_pixel(3, 8, Luma([200])); // soft edge just outside the subject
        let a = instance_alpha(&soft, &part);
        assert_eq!(a.get_pixel(8, 8)[0], 255); // highlight in interior stays opaque
        assert_eq!(a.get_pixel(2, 2)[0], 0); // leak is cut off
        assert_eq!(a.get_pixel(3, 8)[0], 200); // model's edge is kept
    }

    #[test]
    fn smoothstep_lut_is_monotonic_with_fixed_ends() {
        let lut = smoothstep_lut(EDGE_LO, EDGE_HI);
        assert_eq!(lut[0], 0);
        assert_eq!(lut[255], 255);
        assert!(lut.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn bleed_replaces_edge_color_with_interior_color() {
        let mut img = RgbaImage::from_pixel(32, 8, Rgba([255, 0, 0, 255])); // red subject
        for y in 0..8 {
            for x in 20..32 {
                img.put_pixel(x, y, Rgba([0, 0, 255, 0])); // blue, fully transparent
            }
            img.put_pixel(19, y, Rgba([128, 0, 128, 128])); // edge tainted by blue
        }
        bleed_edges(&mut img, 4);
        let p = img.get_pixel(19, 4);
        assert!(p[0] > 200 && p[2] < 40, "edge not bled: {p:?}");
        assert_eq!(p[3], 128); // alpha untouched
        assert_eq!(img.get_pixel(5, 4).0, [255, 0, 0, 255]); // interior untouched
        assert_eq!(img.get_pixel(25, 4).0, [0, 0, 255, 0]); // invisible pixels skipped
    }

    #[test]
    fn square_on_centres_without_distortion() {
        let img = RgbaImage::from_pixel(10, 4, Rgba([255, 0, 0, 255])); // wide red strip
        let sq = square_on(&img, [10, 20, 30], None);
        assert_eq!(sq.dimensions(), (10, 10));
        assert_eq!(sq.get_pixel(5, 5).0, [255, 0, 0]); // subject in the middle
        assert_eq!(sq.get_pixel(5, 0).0, [10, 20, 30]); // padding = bg
        assert_eq!(sq.get_pixel(5, 9).0, [10, 20, 30]);
    }

    #[test]
    fn square_on_blends_alpha_and_resizes() {
        let mut img = RgbaImage::from_pixel(4, 8, Rgba([200, 200, 200, 255]));
        img.put_pixel(0, 0, Rgba([200, 200, 200, 0])); // transparent -> pure bg
        let sq = square_on(&img, [0, 0, 0], None);
        assert_eq!(sq.dimensions(), (8, 8));
        assert_eq!(sq.get_pixel(2, 0).0, [0, 0, 0]); // (0,0) lands at (2,0)
        assert_eq!(square_on(&img, [0, 0, 0], Some(16)).dimensions(), (16, 16));
    }

    #[test]
    fn bbox_is_inclusive_and_none_when_empty() {
        let m = square(16, 16, 4, 5, 12, 9);
        assert_eq!(mask_bbox(&m), Some((4, 5, 11, 8)));
        assert_eq!(mask_bbox(&GrayImage::new(8, 8)), None);
    }
}
