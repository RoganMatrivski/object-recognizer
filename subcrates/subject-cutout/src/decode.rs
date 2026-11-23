//! Image loading.
//!
//! `jpeg-scaled` feature: JPEGs are decoded at 1/2, 1/4 or 1/8 size straight from the
//! DCT coefficients (much cheaper than full decode + resize). The returned image is
//! then *smaller than the file*, but never shorter than `min_side` on its shorter side.
//! Anything that isn't a plain RGB/gray JPEG falls back to `image::open`.
//!
//! Without the feature this is just `image::open`.

use image::{DynamicImage, ImageResult};
use std::path::Path;

use crate::variables::JPEG_MIN_SIDE;

/// Load an image using [`JPEG_MIN_SIDE`] as the size floor for scaled JPEG decoding.
pub fn open_image(path: impl AsRef<Path>) -> ImageResult<DynamicImage> {
    open_image_with(path, JPEG_MIN_SIDE)
}

/// Like [`open_image`] with an explicit floor (ignored unless `jpeg-scaled` is on).
#[allow(unused_variables)]
pub fn open_image_with(path: impl AsRef<Path>, min_side: u16) -> ImageResult<DynamicImage> {
    let path = path.as_ref();
    #[cfg(feature = "jpeg-scaled")]
    if let Some(img) = scaled_jpeg(path, min_side) {
        return Ok(img);
    }
    image::open(path)
}

#[cfg(feature = "jpeg-scaled")]
fn scaled_jpeg(path: &Path, min_side: u16) -> Option<DynamicImage> {
    use image::{GrayImage, RgbImage};
    use jpeg_decoder::{Decoder, PixelFormat};
    use std::{fs::File, io::BufReader};

    let mut dec = Decoder::new(BufReader::new(File::open(path).ok()?));
    dec.read_info().ok()?; // not a JPEG -> None -> fallback
    let info = dec.info()?;
    let (iw, ih) = (info.width as f32, info.height as f32);
    let k = (min_side as f32 / iw.min(ih)).min(1.0); // never upscale
    let req_w = (iw * k).ceil().max(1.0) as u16;
    let req_h = (ih * k).ceil().max(1.0) as u16;
    // Picks the largest power-of-two reduction that is still >= the request.
    let (w, h) = dec.scale(req_w, req_h).ok()?;
    let pixels = dec.decode().ok()?;
    match dec.info()?.pixel_format {
        PixelFormat::RGB24 => {
            RgbImage::from_raw(w as u32, h as u32, pixels).map(DynamicImage::ImageRgb8)
        }
        PixelFormat::L8 => {
            GrayImage::from_raw(w as u32, h as u32, pixels).map(DynamicImage::ImageLuma8)
        }
        _ => None, // CMYK, 16-bit: let `image` handle it
    }
}
