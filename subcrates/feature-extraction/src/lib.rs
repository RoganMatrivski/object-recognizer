mod config;

use image::imageops::FilterType;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ExtractionError {
    #[error("ONNX error: {0}")]
    Ort(#[from] ort::Error),
    #[error("ONNX builder error: {0}")]
    OrtBuilder(String),
    #[error("Image error: {0}")]
    Image(#[from] image::ImageError),
    #[error("Invalid output shape or type from model")]
    InvalidOutput,
}

pub struct FeatureExtractor {
    session: Session,
}

impl FeatureExtractor {
    fn build_session(
        env: &ort::environment::Environment,
        commit_fn: impl FnOnce(&mut ort::session::builder::SessionBuilder) -> ort::Result<Session>,
    ) -> Result<Self, ExtractionError> {
        let intra_threads = std::env::var("ORT_INTRA_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| num_cpus::get().min(4));
        let mut builder = Session::builder(env)?
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(|e| ExtractionError::OrtBuilder(e.to_string()))?
            .with_intra_threads(intra_threads)
            .map_err(|e| ExtractionError::OrtBuilder(e.to_string()))?;
        let session = commit_fn(&mut builder)?;
        Ok(Self { session })
    }

    pub fn new_with_env(
        env: &ort::environment::Environment,
        model_path: impl AsRef<Path>,
    ) -> Result<Self, ExtractionError> {
        let path = model_path.as_ref().to_path_buf();
        Self::build_session(env, |builder| builder.commit_from_file(path))
    }

    #[cfg(feature = "embed-model")]
    pub fn new_embedded_with_env(
        env: &ort::environment::Environment,
    ) -> Result<Self, ExtractionError> {
        Self::build_session(env, |builder| {
            builder.commit_from_memory(config::EMBEDDED_MODEL_BYTES)
        })
    }

    pub fn prepare_input(img: &image::DynamicImage) -> Vec<f32> {
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
        data
    }

    pub fn run(&mut self, input: Vec<f32>) -> Result<Vec<f32>, ExtractionError> {
        let (w, h) = (config::MODEL_WIDTH, config::MODEL_HEIGHT);
        let input_tensor = Tensor::from_array(([1, 3, h as i64, w as i64], input))?;
        let outputs = self.session.run(ort::inputs!["image" => input_tensor])?;

        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;

        let slice = shape.as_ref();
        let expected_dim = config::EMBEDDING_DIM;
        if slice != [1, expected_dim as i64] && slice != [expected_dim as i64] {
            return Err(ExtractionError::InvalidOutput);
        }

        Ok(data.to_vec())
    }

    pub fn extract(&mut self, img: &image::DynamicImage) -> Result<Vec<f32>, ExtractionError> {
        let input = Self::prepare_input(img);
        self.run(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepare_input_produces_expected_dimensions() {
        let img = image::DynamicImage::ImageRgb8(image::RgbImage::new(100, 100));
        let data = FeatureExtractor::prepare_input(&img);
        let expected_size = 1 * 3 * config::MODEL_WIDTH as usize * config::MODEL_HEIGHT as usize;
        assert_eq!(data.len(), expected_size);
    }
}
