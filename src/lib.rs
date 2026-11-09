use object_recognizer_feature_extraction;
use object_recognizer_subject_cutout;
use object_recognizer_vecdb;

use object_recognizer_subject_cutout::{Model, Segmenter};

boltffi::scaffolding!();

pub mod ffi;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("ort error: {0}")]
    Ort(#[from] ort::Error),
    #[error("feature extraction error: {0}")]
    FeatureExtraction(#[from] object_recognizer_feature_extraction::ExtractionError),
    #[error("segmenter error: {0}")]
    Segmenter(#[from] object_recognizer_subject_cutout::Error),
    #[error("vecdb error: {0}")]
    VecDb(#[from] anyhow::Error),
    #[error("image error: {0}")]
    Image(#[from] image::ImageError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

pub struct Recognizer {
    pub seg: object_recognizer_subject_cutout::Segmenter,
    pub extractor: object_recognizer_feature_extraction::FeatureExtractor,
    pub vecdb: object_recognizer_vecdb::VecDb,
    pub env: ort::environment::Environment,
}

impl Recognizer {
    fn from_parts(
        env: ort::environment::Environment,
        seg: Segmenter,
        vecdb_path: impl AsRef<std::path::Path>,
    ) -> Result<Self> {
        let extractor = create_feature_extractor(&env)?;
        let vecdb = object_recognizer_vecdb::VecDb::open(vecdb_path)?;
        Ok(Self {
            env,
            seg,
            extractor,
            vecdb,
        })
    }

    pub fn new(
        segmenter_model_path: impl AsRef<std::path::Path>,
        vecdb_path: impl AsRef<std::path::Path>,
    ) -> Result<Self> {
        let env = ort::init().build()?;
        let seg = create_segmenter_from_path(&env, segmenter_model_path)?;
        Self::from_parts(env, seg, vecdb_path)
    }

    pub fn new_default(segmenter_model_path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::new(segmenter_model_path, object_recognizer_vecdb::DEFAULT_DIR)
    }

    pub fn new_embedded(vecdb_path: impl AsRef<std::path::Path>) -> Result<Self> {
        let env = ort::init().build()?;
        let seg = Segmenter::embedded_with_env(Model::U2Net, &env)?;
        Self::from_parts(env, seg, vecdb_path)
    }

    pub fn new_embedded_lite(vecdb_path: impl AsRef<std::path::Path>) -> Result<Self> {
        let env = ort::init().build()?;
        let seg = Segmenter::embedded_with_env(Model::U2NetP, &env)?;
        Self::from_parts(env, seg, vecdb_path)
    }

    pub fn new_embedded_default() -> Result<Self> {
        Self::new_embedded(object_recognizer_vecdb::DEFAULT_DIR)
    }

    pub fn segment_subject(&mut self, img: &image::DynamicImage) -> Result<image::DynamicImage> {
        let rgba = self.seg.cutout(img)?;
        let dyn_img = image::DynamicImage::ImageRgba8(rgba);
        Ok(dyn_img)
    }

    pub fn preprocess_image(&mut self, img: &image::DynamicImage) -> Result<image::DynamicImage> {
        self.segment_subject(img)
    }

    pub fn generate_embedding(&mut self, img: &image::DynamicImage) -> Result<Vec<f32>> {
        let embedding = self.extractor.extract(img)?;
        Ok(embedding)
    }

    pub fn vector_db_insert<P: serde::Serialize>(
        &mut self,
        vector: &[f32],
        metadata: &P,
    ) -> Result<u64> {
        let ids = self
            .vecdb
            .add_with(std::iter::once((vector.to_vec(), metadata)))?;
        Ok(ids[0])
    }

    pub fn vector_db_search(
        &self,
        query_vector: &[f32],
        k: usize,
    ) -> Result<Vec<object_recognizer_vecdb::Hit<serde_json::Value>>> {
        let results = self.vecdb.recognize(query_vector.to_vec(), k, 0.0)?;
        Ok(results)
    }

    pub fn vector_db_save(&self, _path: &str) -> Result<()> {
        self.vecdb.optimize()?;
        Ok(())
    }

    pub fn vector_db_load(
        &mut self,
        path: &str,
        mode: object_recognizer_vecdb::ImportMode,
    ) -> Result<usize> {
        let count = self.vecdb.import(path, mode)?;
        Ok(count)
    }

    pub fn add_training_sample<P: serde::Serialize>(
        &mut self,
        img: &image::DynamicImage,
        metadata: &P,
    ) -> Result<u64> {
        let segmented = self.segment_subject(img)?;
        let embedding = self.generate_embedding(&segmented)?;
        let id = self.vector_db_insert(&embedding, metadata)?;
        Ok(id)
    }

    pub fn classify_image<T: serde::de::DeserializeOwned>(
        &mut self,
        img: &image::DynamicImage,
        top_n: usize,
        min_score: f32,
    ) -> Result<Vec<object_recognizer_vecdb::Hit<T>>> {
        let segmented = self.segment_subject(img)?;
        let embedding = self.generate_embedding(&segmented)?;
        let hits = self.vecdb.recognize(embedding, top_n, min_score)?;
        Ok(hits)
    }
}

pub fn create_segmenter_from_path(
    env: &ort::environment::Environment,
    path: impl AsRef<std::path::Path>,
) -> Result<object_recognizer_subject_cutout::Segmenter> {
    object_recognizer_subject_cutout::Segmenter::from_path_with_env(
        object_recognizer_subject_cutout::Model::U2Net,
        env,
        path,
    )
    .map_err(Error::Segmenter)
}

pub fn create_feature_extractor(
    env: &ort::environment::Environment,
) -> Result<object_recognizer_feature_extraction::FeatureExtractor> {
    object_recognizer_feature_extraction::FeatureExtractor::new_embedded_with_env(env)
        .map_err(Error::FeatureExtraction)
}
