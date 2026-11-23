//! Resize backends shared by the cutout and feature-extraction crates.
//!
//! Default: the `image` crate's scalar resizer.
//! `fast-resize` feature: SIMD `fast_image_resize` (SSE4.1/AVX2/NEON), same filters.
//!
//! Callers only use [`rgb`] and [`gray`], so the feature is a pure drop-in toggle.

use image::{DynamicImage, GrayImage, RgbImage, imageops::FilterType as ImgFilter};

#[derive(Clone, Copy, Debug)]
pub enum Filter {
    /// Bilinear. Cheapest; fine for the model input and soft masks.
    Triangle,
    CatmullRom,
    Lanczos3,
}

impl Filter {
    fn to_image(self) -> ImgFilter {
        match self {
            Filter::Triangle => ImgFilter::Triangle,
            Filter::CatmullRom => ImgFilter::CatmullRom,
            Filter::Lanczos3 => ImgFilter::Lanczos3,
        }
    }
}

/// Resize any image to exactly `w` x `h`, returned as RGB8.
pub fn rgb(img: &DynamicImage, w: u32, h: u32, f: Filter) -> RgbImage {
    #[cfg(feature = "fast-resize")]
    if let Some(out) = fast::rgb(img, w, h, f) {
        return out;
    }
    img.resize_exact(w, h, f.to_image()).to_rgb8()
}

/// Resize an 8-bit gray image to exactly `w` x `h`.
pub fn gray(img: &GrayImage, w: u32, h: u32, f: Filter) -> GrayImage {
    #[cfg(feature = "fast-resize")]
    if let Some(out) = fast::gray(img, w, h, f) {
        return out;
    }
    image::imageops::resize(img, w, h, f.to_image())
}

#[cfg(feature = "fast-resize")]
mod fast {
    use super::Filter;
    use fast_image_resize::{
        FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer,
        images::{Image, ImageRef},
    };
    use image::{DynamicImage, GrayImage, RgbImage};
    use std::cell::RefCell;

    thread_local! {
        // Created once per thread: avoids re-detecting CPU features and
        // re-allocating coefficient buffers on every call.
        static RESIZER: RefCell<Resizer> = RefCell::new(Resizer::new());
    }

    fn alg(f: Filter) -> ResizeAlg {
        ResizeAlg::Convolution(match f {
            Filter::Triangle => FilterType::Bilinear,
            Filter::CatmullRom => FilterType::CatmullRom,
            Filter::Lanczos3 => FilterType::Lanczos3,
        })
    }

    // None = couldn't (zero-sized etc.): caller falls back to the `image` crate.
    pub fn rgb(img: &DynamicImage, w: u32, h: u32, f: Filter) -> Option<RgbImage> {
        let owned;
        let src_img: &RgbImage = match img {
            DynamicImage::ImageRgb8(i) => i, // borrow, no copy
            _ => {
                owned = img.to_rgb8();
                &owned
            }
        };
        let src = ImageRef::new(src_img.width(), src_img.height(), src_img.as_raw(), PixelType::U8x3).ok()?;
        let mut dst = Image::new(w, h, PixelType::U8x3);
        RESIZER
            .with(|r| r.borrow_mut().resize(&src, &mut dst, &ResizeOptions::new().resize_alg(alg(f))))
            .ok()?;
        RgbImage::from_raw(w, h, dst.into_vec())
    }

    pub fn gray(img: &GrayImage, w: u32, h: u32, f: Filter) -> Option<GrayImage> {
        let src = ImageRef::new(img.width(), img.height(), img.as_raw(), PixelType::U8).ok()?;
        let mut dst = Image::new(w, h, PixelType::U8);
        RESIZER
            .with(|r| r.borrow_mut().resize(&src, &mut dst, &ResizeOptions::new().resize_alg(alg(f))))
            .ok()?;
        GrayImage::from_raw(w, h, dst.into_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Luma, Rgb};

    fn gradient_rgb(w: u32, h: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_fn(w, h, |x, y| {
            Rgb([(x * 255 / w) as u8, (y * 255 / h) as u8, ((x + y) * 255 / (w + h)) as u8])
        }))
    }

    #[test]
    fn rgb_has_requested_size() {
        for f in [Filter::Triangle, Filter::CatmullRom, Filter::Lanczos3] {
            assert_eq!(rgb(&gradient_rgb(333, 211), 64, 64, f).dimensions(), (64, 64));
        }
    }

    #[test]
    fn rgba_input_is_accepted() {
        let rgba = DynamicImage::ImageRgba8(gradient_rgb(50, 40).to_rgba8());
        assert_eq!(rgb(&rgba, 16, 16, Filter::Triangle).dimensions(), (16, 16));
    }

    #[test]
    fn gray_up_and_down() {
        let g = GrayImage::from_fn(40, 30, |x, _| Luma([(x * 6) as u8]));
        assert_eq!(gray(&g, 160, 120, Filter::Lanczos3).dimensions(), (160, 120));
        assert_eq!(gray(&g, 10, 8, Filter::Triangle).dimensions(), (10, 8));
    }

    /// The SIMD path must agree with the scalar one (tiny rounding differences only).
    #[cfg(feature = "fast-resize")]
    #[test]
    fn fast_matches_scalar() {
        let src = gradient_rgb(640, 480);
        for f in [Filter::Triangle, Filter::CatmullRom, Filter::Lanczos3] {
            let a = rgb(&src, 320, 320, f);
            let b = src.resize_exact(320, 320, f.to_image()).to_rgb8();
            let mad: f64 = a.as_raw().iter().zip(b.as_raw()).map(|(x, y)| (*x as f64 - *y as f64).abs()).sum::<f64>() / a.as_raw().len() as f64;
            assert!(mad < 1.5, "{f:?}: mean abs diff {mad}");
        }
    }
}
