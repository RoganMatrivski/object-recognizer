pub use boltffi::*;

#[cfg(test)]
mod concurrency_tests;

boltffi::scaffolding!();

#[data]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HitDto {
    pub id: u64,
    pub score: f32,
    pub metadata_json: String,
}

#[data]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct KeyCount {
    pub value: String,
    pub count: usize,
}

#[error]
#[derive(Clone, Debug)]
pub enum RecognizerError {
    Initialization { message: String },
    ImageDecode { message: String },
    InvalidMetadata { message: String },
    Inference { message: String },
    VectorDb { message: String },
    Serialization { message: String },
    NoSubjectFound,
}

impl RecognizerError {
    fn initialization(error: impl std::fmt::Display) -> Self {
        Self::Initialization {
            message: error.to_string(),
        }
    }

    fn image_decode(error: impl std::fmt::Display) -> Self {
        Self::ImageDecode {
            message: error.to_string(),
        }
    }

    fn invalid_metadata(error: impl std::fmt::Display) -> Self {
        Self::InvalidMetadata {
            message: error.to_string(),
        }
    }

    fn inference(error: impl std::fmt::Display) -> Self {
        Self::Inference {
            message: error.to_string(),
        }
    }

    fn vector_db(error: impl std::fmt::Display) -> Self {
        Self::VectorDb {
            message: error.to_string(),
        }
    }

    fn serialization(error: impl std::fmt::Display) -> Self {
        Self::Serialization {
            message: error.to_string(),
        }
    }
}

#[export]
pub struct RecognizerWrapper {
    seg: parking_lot::Mutex<object_recognizer::subject_cutout::Segmenter>,
    extractor: parking_lot::Mutex<object_recognizer::feature_extraction::FeatureExtractor>,
    vecdb: object_recognizer::vecdb::VecDb,
    _env: ort::environment::Environment,
}

fn assert_send_sync<T: Send + Sync>() {}

#[export]
impl RecognizerWrapper {
    fn from_recognizer(rec: object_recognizer::Recognizer) -> Self {
        let (seg, extractor, vecdb, _env) = rec.into_parts();
        Self {
            seg: parking_lot::Mutex::new(seg),
            extractor: parking_lot::Mutex::new(extractor),
            vecdb,
            _env,
        }
    }

    pub fn new(segmenter_model_path: String, vecdb_path: String) -> Result<Self, RecognizerError> {
        assert_send_sync::<RecognizerWrapper>();
        let rec = object_recognizer::Recognizer::new(segmenter_model_path, vecdb_path)
            .map_err(RecognizerError::initialization)?;
        Ok(Self::from_recognizer(rec))
    }

    pub fn new_embedded(vecdb_path: String) -> Result<Self, RecognizerError> {
        assert_send_sync::<RecognizerWrapper>();
        let rec = object_recognizer::Recognizer::new_embedded(vecdb_path)
            .map_err(RecognizerError::initialization)?;
        Ok(Self::from_recognizer(rec))
    }

    pub fn new_embedded_lite(vecdb_path: String) -> Result<Self, RecognizerError> {
        assert_send_sync::<RecognizerWrapper>();
        let rec = object_recognizer::Recognizer::new_embedded_lite(vecdb_path)
            .map_err(RecognizerError::initialization)?;
        Ok(Self::from_recognizer(rec))
    }

    pub fn new_embedded_default() -> Result<Self, RecognizerError> {
        assert_send_sync::<RecognizerWrapper>();
        let rec = object_recognizer::Recognizer::new_embedded_default()
            .map_err(RecognizerError::initialization)?;
        Ok(Self::from_recognizer(rec))
    }

    pub fn new_default(segmenter_model_path: String) -> Result<Self, RecognizerError> {
        assert_send_sync::<RecognizerWrapper>();
        let rec = object_recognizer::Recognizer::new_default(segmenter_model_path)
            .map_err(RecognizerError::initialization)?;
        Ok(Self::from_recognizer(rec))
    }

    pub fn segment_subject_raw(
        &self,
        image_bytes: Vec<u8>,
    ) -> Result<Vec<Vec<u8>>, RecognizerError> {
        let img = image::load_from_memory(&image_bytes).map_err(RecognizerError::image_decode)?;
        let input = object_recognizer::subject_cutout::Segmenter::prepare_input(&img);
        let raw = {
            let mut seg = self.seg.lock();
            seg.run_raw(input).map_err(RecognizerError::inference)?
        };
        let soft = object_recognizer::subject_cutout::Segmenter::saliency_from_raw(&raw)
            .map_err(RecognizerError::inference)?;
        let cutouts =
            object_recognizer::subject_cutout::squares_from_saliency(&img, &soft, Some(224))
                .map_err(RecognizerError::inference)?;
        Ok(cutouts.into_iter().map(|c| c.into_raw()).collect())
    }

    pub fn generate_embedding(&self, image_bytes: Vec<u8>) -> Result<Vec<f32>, RecognizerError> {
        let img = image::load_from_memory(&image_bytes).map_err(RecognizerError::image_decode)?;
        let input = object_recognizer::feature_extraction::FeatureExtractor::prepare_input(&img);
        let mut ext = self.extractor.lock();
        ext.run(input).map_err(RecognizerError::inference)
    }

    pub fn vector_db_insert(
        &self,
        vector: Vec<f32>,
        metadata_json: String,
    ) -> Result<u64, RecognizerError> {
        let meta: serde_json::Value =
            serde_json::from_str(&metadata_json).map_err(RecognizerError::invalid_metadata)?;
        let ids = self
            .vecdb
            .add_with(std::iter::once((vector, meta)))
            .map_err(RecognizerError::vector_db)?;
        Ok(ids[0])
    }

    pub fn vector_db_search(
        &self,
        query_vector: Vec<f32>,
        k: usize,
    ) -> Result<Vec<HitDto>, RecognizerError> {
        let hits = self
            .vecdb
            .recognize::<serde_json::Value>(query_vector, k, 0.0)
            .map_err(RecognizerError::vector_db)?;
        let mut dtos = Vec::new();
        for h in hits {
            dtos.push(HitDto {
                id: h.id,
                score: h.score,
                metadata_json: serde_json::to_string(&h.meta)
                    .map_err(RecognizerError::serialization)?,
            });
        }
        Ok(dtos)
    }

    pub fn vector_db_save(&self, path: String) -> Result<(), String> {
        let _ = path;
        self.vecdb.optimize().map_err(|e| e.to_string())
    }

    pub fn vector_db_export(&self, path: String) -> Result<usize, String> {
        let count = self.vecdb.export(&path).map_err(|e| e.to_string())?;
        Ok(count)
    }

    pub fn vector_db_import(&self, path: String, replace: bool) -> Result<usize, RecognizerError> {
        let mode = if replace {
            object_recognizer::vecdb::ImportMode::Replace
        } else {
            object_recognizer::vecdb::ImportMode::Merge
        };
        let count = self
            .vecdb
            .import(&path, mode)
            .map_err(RecognizerError::vector_db)?;
        Ok(count)
    }

    pub fn vector_db_clear(&self) -> Result<(), RecognizerError> {
        self.vecdb.clear().map_err(RecognizerError::vector_db)
    }

    pub fn add_training_sample(
        &self,
        image_bytes: Vec<u8>,
        metadata_json: String,
    ) -> Result<Vec<u64>, RecognizerError> {
        let img = image::load_from_memory(&image_bytes).map_err(RecognizerError::image_decode)?;
        let meta: serde_json::Value =
            serde_json::from_str(&metadata_json).map_err(RecognizerError::invalid_metadata)?;

        let input = object_recognizer::subject_cutout::Segmenter::prepare_input(&img);
        let raw = {
            let mut seg = self.seg.lock();
            seg.run_raw(input).map_err(RecognizerError::inference)?
        };
        let soft = object_recognizer::subject_cutout::Segmenter::saliency_from_raw(&raw)
            .map_err(RecognizerError::inference)?;
        let cutouts =
            object_recognizer::subject_cutout::squares_from_saliency(&img, &soft, Some(224))
                .map_err(RecognizerError::inference)?;

        let mut embeddings = Vec::new();
        for cutout in cutouts {
            let dyn_img = image::DynamicImage::ImageRgb8(cutout);
            let ext_input =
                object_recognizer::feature_extraction::FeatureExtractor::prepare_input(&dyn_img);
            let embedding = {
                let mut ext = self.extractor.lock();
                ext.run(ext_input).map_err(RecognizerError::inference)?
            };
            embeddings.push(embedding);
        }

        if embeddings.is_empty() {
            return Err(RecognizerError::NoSubjectFound);
        }

        let ids = self
            .vecdb
            .add_with(embeddings.into_iter().map(|emb| (emb, meta.clone())))
            .map_err(RecognizerError::vector_db)?;
        Ok(ids)
    }

    pub fn classify_image(
        &self,
        image_bytes: Vec<u8>,
        top_n: usize,
        min_score: f32,
    ) -> Result<Vec<HitDto>, RecognizerError> {
        let img = image::load_from_memory(&image_bytes).map_err(RecognizerError::image_decode)?;

        let input = object_recognizer::subject_cutout::Segmenter::prepare_input(&img);
        let raw = {
            let mut seg = self.seg.lock();
            seg.run_raw(input).map_err(RecognizerError::inference)?
        };
        let soft = object_recognizer::subject_cutout::Segmenter::saliency_from_raw(&raw)
            .map_err(RecognizerError::inference)?;
        let cutouts =
            object_recognizer::subject_cutout::squares_from_saliency(&img, &soft, Some(224))
                .map_err(RecognizerError::inference)?;

        let mut best_hits = Vec::new();
        for cutout in cutouts {
            let dyn_img = image::DynamicImage::ImageRgb8(cutout);
            let ext_input =
                object_recognizer::feature_extraction::FeatureExtractor::prepare_input(&dyn_img);
            let embedding = {
                let mut ext = self.extractor.lock();
                ext.run(ext_input).map_err(RecognizerError::inference)?
            };
            let hits = self
                .vecdb
                .recognize::<serde_json::Value>(embedding, top_n, min_score)
                .map_err(RecognizerError::vector_db)?;
            best_hits.extend(hits);
        }

        let merged = object_recognizer::merge_hits(best_hits, top_n);
        let mut dtos = Vec::new();
        for h in merged {
            dtos.push(HitDto {
                id: h.id,
                score: h.score,
                metadata_json: serde_json::to_string(&h.meta)
                    .map_err(RecognizerError::serialization)?,
            });
        }
        Ok(dtos)
    }
}
