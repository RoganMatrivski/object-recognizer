mod config;

use image::imageops::FilterType;
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ExtractionError {
    #[error("ONNX error: {0}")]
    Ort(#[from] ort::Error),
    #[error("Image error: {0}")]
    Image(#[from] image::ImageError),
    #[error("Invalid output shape or type from model")]
    InvalidOutput,
}

pub struct FeatureExtractor {
    session: Session,
}

impl FeatureExtractor {
    pub fn new_with_env(
        env: &ort::environment::Environment,
        model_path: impl AsRef<Path>,
    ) -> Result<Self, ExtractionError> {
        let session = Session::builder(env)?.commit_from_file(model_path)?;
        Ok(Self { session })
    }

    #[cfg(feature = "embed-model")]
    pub fn new_embedded_with_env(
        env: &ort::environment::Environment,
    ) -> Result<Self, ExtractionError> {
        let session = Session::builder(env)?.commit_from_memory(config::EMBEDDED_MODEL_BYTES)?;
        Ok(Self { session })
    }

    pub fn extract(&mut self, img: &image::DynamicImage) -> Result<Vec<f32>, ExtractionError> {
        let (w, h) = (config::MODEL_WIDTH, config::MODEL_HEIGHT);
        let rgb = img.resize_exact(w, h, FilterType::Triangle).to_rgb8();

        let area = (w as usize) * (h as usize);
        let mut data = vec![0.0f32; 1 * 3 * area];
        for (x, y, pixel) in rgb.enumerate_pixels() {
            let [r, g, b] = pixel.0;
            let idx = (y as usize) * (w as usize) + (x as usize);
            data[idx] = r as f32 / 255.0;
            data[area + idx] = g as f32 / 255.0;
            data[2 * area + idx] = b as f32 / 255.0;
        }

        let input_tensor = Tensor::from_array(([1, 3, h as i64, w as i64], data))?;
        let outputs = self.session.run(ort::inputs!["image" => input_tensor])?;

        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;

        let slice = shape.as_ref();
        let expected_dim = config::EMBEDDING_DIM;
        if slice != [1, expected_dim as i64] && slice != [expected_dim as i64] {
            return Err(ExtractionError::InvalidOutput);
        }

        Ok(data.to_vec())
    }
}
