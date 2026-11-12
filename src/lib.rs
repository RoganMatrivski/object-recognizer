pub use object_recognizer_feature_extraction as feature_extraction;
pub use object_recognizer_subject_cutout as subject_cutout;
pub use object_recognizer_vecdb as vecdb;

use object_recognizer_subject_cutout::{Model, Segmenter};

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
        _segmenter_model_path: impl AsRef<std::path::Path>,
        vecdb_path: impl AsRef<std::path::Path>,
    ) -> Result<Self> {
        let _span = tracing::trace_span!("Recognizer::from_parts").entered();
        let start = std::time::Instant::now();
        let extractor = create_feature_extractor(&env)?;
        let vecdb = object_recognizer_vecdb::VecDb::open(vecdb_path)?;
        tracing::trace!(elapsed_ms = ?start.elapsed().as_secs_f64() * 1000.0, "Recognizer::from_parts completed");
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
        let _span = tracing::trace_span!("Recognizer::new").entered();
        let start = std::time::Instant::now();
        let env = ort::init().build()?;
        let seg = create_segmenter_from_path(&env, &segmenter_model_path)?;
        let res = Self::from_parts(env, seg, segmenter_model_path, vecdb_path);
        tracing::trace!(elapsed_ms = ?start.elapsed().as_secs_f64() * 1000.0, "Recognizer::new completed");
        res
    }

    pub fn new_default(segmenter_model_path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::new(segmenter_model_path, object_recognizer_vecdb::DEFAULT_DIR)
    }

    pub fn new_embedded(vecdb_path: impl AsRef<std::path::Path>) -> Result<Self> {
        let _span = tracing::trace_span!("Recognizer::new_embedded").entered();
        let start = std::time::Instant::now();
        let env = ort::init().build()?;
        let seg = Segmenter::embedded_with_env(Model::U2Net, &env)?;
        let res = Self::from_parts(env, seg, "", vecdb_path);
        tracing::trace!(elapsed_ms = ?start.elapsed().as_secs_f64() * 1000.0, "Recognizer::new_embedded completed");
        res
    }

    pub fn new_embedded_lite(vecdb_path: impl AsRef<std::path::Path>) -> Result<Self> {
        let _span = tracing::trace_span!("Recognizer::new_embedded_lite").entered();
        let start = std::time::Instant::now();
        let env = ort::init().build()?;
        let seg = Segmenter::embedded_with_env(Model::U2NetP, &env)?;
        let res = Self::from_parts(env, seg, "", vecdb_path);
        tracing::trace!(elapsed_ms = ?start.elapsed().as_secs_f64() * 1000.0, "Recognizer::new_embedded_lite completed");
        res
    }

    pub fn new_embedded_default() -> Result<Self> {
        Self::new_embedded(object_recognizer_vecdb::DEFAULT_DIR)
    }

    pub fn segment_subject(
        &mut self,
        img: &image::DynamicImage,
    ) -> Result<Vec<image::DynamicImage>> {
        let _span = tracing::trace_span!("Recognizer::segment_subject").entered();
        let start = std::time::Instant::now();
        let cutouts = self.seg.instance_squares(img, Some(224))?;
        let dyn_imgs = cutouts
            .into_iter()
            .map(image::DynamicImage::ImageRgb8)
            .collect();
        tracing::trace!(elapsed_ms = ?start.elapsed().as_secs_f64() * 1000.0, "Recognizer::segment_subject completed");
        Ok(dyn_imgs)
    }

    pub fn preprocess_image(
        &mut self,
        img: &image::DynamicImage,
    ) -> Result<Vec<image::DynamicImage>> {
        self.segment_subject(img)
    }

    pub fn generate_embedding(&mut self, img: &image::DynamicImage) -> Result<Vec<f32>> {
        let _span = tracing::trace_span!("Recognizer::generate_embedding").entered();
        let start = std::time::Instant::now();
        let embedding = self.extractor.extract(img)?;
        tracing::trace!(elapsed_ms = ?start.elapsed().as_secs_f64() * 1000.0, "Recognizer::generate_embedding completed");
        Ok(embedding)
    }

    pub fn vector_db_insert<P: serde::Serialize>(
        &mut self,
        vector: &[f32],
        metadata: &P,
    ) -> Result<u64> {
        let _span = tracing::trace_span!("Recognizer::vector_db_insert").entered();
        let start = std::time::Instant::now();
        let ids = self
            .vecdb
            .add_with(std::iter::once((vector.to_vec(), metadata)))?;
        tracing::trace!(elapsed_ms = ?start.elapsed().as_secs_f64() * 1000.0, "Recognizer::vector_db_insert completed");
        Ok(ids[0])
    }

    pub fn vector_db_search(
        &self,
        query_vector: &[f32],
        k: usize,
    ) -> Result<Vec<object_recognizer_vecdb::Hit<serde_json::Value>>> {
        let _span = tracing::trace_span!("Recognizer::vector_db_search").entered();
        let start = std::time::Instant::now();
        let results = self.vecdb.recognize(query_vector.to_vec(), k, 0.0)?;
        tracing::trace!(elapsed_ms = ?start.elapsed().as_secs_f64() * 1000.0, "Recognizer::vector_db_search completed");
        Ok(results)
    }

    pub fn vector_db_save(&self, _path: &str) -> Result<()> {
        let _span = tracing::trace_span!("Recognizer::vector_db_save").entered();
        let start = std::time::Instant::now();
        self.vecdb.optimize()?;
        tracing::trace!(elapsed_ms = ?start.elapsed().as_secs_f64() * 1000.0, "Recognizer::vector_db_save completed");
        Ok(())
    }

    pub fn vector_db_load(
        &mut self,
        path: &str,
        mode: object_recognizer_vecdb::ImportMode,
    ) -> Result<usize> {
        let _span = tracing::trace_span!("Recognizer::vector_db_load").entered();
        let start = std::time::Instant::now();
        let count = self.vecdb.import(path, mode)?;
        tracing::trace!(elapsed_ms = ?start.elapsed().as_secs_f64() * 1000.0, "Recognizer::vector_db_load completed");
        Ok(count)
    }

    pub fn add_training_sample<P: serde::Serialize>(
        &mut self,
        img: &image::DynamicImage,
        metadata: &P,
    ) -> Result<Vec<u64>> {
        let _span = tracing::trace_span!("Recognizer::add_training_sample").entered();
        let start = std::time::Instant::now();
        let segmented = self.segment_subject(img)?;
        let mut ids = Vec::new();
        for seg in segmented {
            let embedding = self.generate_embedding(&seg)?;
            let id = self.vector_db_insert(&embedding, metadata)?;
            ids.push(id);
        }
        tracing::trace!(elapsed_ms = ?start.elapsed().as_secs_f64() * 1000.0, "Recognizer::add_training_sample completed");
        Ok(ids)
    }

    pub fn into_parts(
        self,
    ) -> (
        Segmenter,
        object_recognizer_feature_extraction::FeatureExtractor,
        object_recognizer_vecdb::VecDb,
        ort::environment::Environment,
    ) {
        (self.seg, self.extractor, self.vecdb, self.env)
    }

    pub fn classify_image<T: serde::de::DeserializeOwned + Send + Sync + 'static + PartialEq>(
        &mut self,
        img: &image::DynamicImage,
        top_n: usize,
        min_score: f32,
    ) -> Result<Vec<object_recognizer_vecdb::Hit<T>>> {
        let segmented = self.segment_subject(img)?;
        let mut best_hits = Vec::new();
        for seg in segmented {
            let embedding = self.generate_embedding(&seg)?;
            let hits = self.vecdb.recognize::<T>(embedding, top_n, min_score)?;
            best_hits.extend(hits);
        }
        Ok(merge_hits(best_hits, top_n))
    }
}

pub fn merge_hits<T>(
    mut hits: Vec<object_recognizer_vecdb::Hit<T>>,
    top_n: usize,
) -> Vec<object_recognizer_vecdb::Hit<T>>
where
    T: PartialEq,
{
    // Sort by score desc
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Dedupe by id, keeping the best score (which is the first one found)
    let mut unique_ids = std::collections::HashSet::new();
    let mut deduped = Vec::new();
    for hit in hits {
        if unique_ids.insert(hit.id) {
            deduped.push(hit);
        }
    }

    deduped.truncate(top_n);
    deduped
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_recognizer_vecdb::Hit;

    #[test]
    fn test_merge_hits() {
        let h1 = Hit {
            id: 1,
            score: 0.9,
            meta: "a",
        };
        let h2 = Hit {
            id: 2,
            score: 0.8,
            meta: "b",
        };
        let h3 = Hit {
            id: 1,
            score: 0.95,
            meta: "a",
        };
        let h4 = Hit {
            id: 3,
            score: 0.7,
            meta: "c",
        };

        let merged = merge_hits(vec![h1, h2, h3, h4], 2);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].id, 1);
        assert_eq!(merged[0].score, 0.95);
        assert_eq!(merged[1].id, 2);
        assert_eq!(merged[1].score, 0.8);
    }
}

pub fn create_segmenter_from_path(
    env: &ort::environment::Environment,
    path: impl AsRef<std::path::Path>,
) -> Result<object_recognizer_subject_cutout::Segmenter> {
    let _span = tracing::trace_span!("create_segmenter_from_path").entered();
    let start = std::time::Instant::now();
    let res = object_recognizer_subject_cutout::Segmenter::from_path_with_env(
        object_recognizer_subject_cutout::Model::U2Net,
        env,
        path,
    )
    .map_err(Error::Segmenter);
    tracing::trace!(elapsed_ms = ?start.elapsed().as_secs_f64() * 1000.0, "create_segmenter_from_path completed");
    res
}

pub fn create_feature_extractor(
    env: &ort::environment::Environment,
) -> Result<object_recognizer_feature_extraction::FeatureExtractor> {
    let _span = tracing::trace_span!("create_feature_extractor").entered();
    let start = std::time::Instant::now();
    let res = object_recognizer_feature_extraction::FeatureExtractor::new_embedded_with_env(env)
        .map_err(Error::FeatureExtraction);
    tracing::trace!(elapsed_ms = ?start.elapsed().as_secs_f64() * 1000.0, "create_feature_extractor completed");
    res
}
