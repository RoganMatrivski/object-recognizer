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
        const MODEL_BYTES: &[u8] = include_bytes!("../model/mobilenetv3_embedding.onnx");
        let session = Session::builder(env)?.commit_from_memory(MODEL_BYTES)?;
        Ok(Self { session })
    }

    pub fn extract(&mut self, img: &image::DynamicImage) -> Result<Vec<f32>, ExtractionError> {
        let rgb = img.resize_exact(224, 224, FilterType::Triangle).to_rgb8();

        let mut data = vec![0.0f32; 1 * 3 * 224 * 224];
        for (x, y, pixel) in rgb.enumerate_pixels() {
            let [r, g, b] = pixel.0;
            let idx = (y as usize) * 224 + (x as usize);
            data[idx] = r as f32 / 255.0;
            data[224 * 224 + idx] = g as f32 / 255.0;
            data[2 * 224 * 224 + idx] = b as f32 / 255.0;
        }

        let input_tensor = Tensor::from_array(([1, 3, 224, 224], data))?;
        let outputs = self.session.run(ort::inputs!["image" => input_tensor])?;

        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;

        let slice = shape.as_ref();
        if slice != [1, 1280] && slice != [1280] {
            return Err(ExtractionError::InvalidOutput);
        }

        Ok(data.to_vec())
    }
}
