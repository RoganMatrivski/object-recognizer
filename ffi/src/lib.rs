pub use boltffi::*;
use image::{DynamicImage, RgbImage};
use std::{collections::HashMap, time::Instant};

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

#[data]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct OperationTiming {
    pub name: String,
    pub elapsed_ms: f64,
}

const TIMED_OPERATIONS: &[&str] = &[
    "new",
    "new_embedded",
    "new_embedded_lite",
    "new_embedded_default",
    "new_default",
    "segment_subject_raw",
    "segment_subject_pixels",
    "generate_embedding",
    "generate_embedding_pixels",
    "vector_db_insert",
    "vector_db_search",
    "vector_db_len",
    "vector_db_count_by",
    "vector_db_delete_by_id",
    "vector_db_delete_where",
    "vector_db_save",
    "vector_db_export",
    "vector_db_import",
    "vector_db_clear",
    "add_training_sample",
    "add_training_sample_pixels",
    "classify_image",
    "classify_pixels",
];

struct TimingGuard<'a> {
    timings: &'a parking_lot::Mutex<HashMap<&'static str, f64>>,
    operation: &'static str,
    started: Instant,
}

impl Drop for TimingGuard<'_> {
    fn drop(&mut self) {
        self.timings.lock().insert(
            self.operation,
            self.started.elapsed().as_secs_f64() * 1000.0,
        );
    }
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

    fn serialization(error: impl std::fmt::Display) -> Self {
        Self::Serialization {
            message: error.to_string(),
        }
    }
}

/// Maps each source error type to its FFI category once, so call sites can use plain `?`.
/// The message has to be flattened to a `String`: boltffi errors cross the boundary as data.
macro_rules! from_source {
    ($($source:ty => $variant:ident),* $(,)?) => {$(
        impl From<$source> for RecognizerError {
            fn from(error: $source) -> Self {
                Self::$variant { message: error.to_string() }
            }
        }
    )*};
}

from_source! {
    object_recognizer::vecdb::Error => VectorDb,
    object_recognizer::subject_cutout::Error => Inference,
    object_recognizer::feature_extraction::ExtractionError => Inference,
    image::ImageError => ImageDecode,
}

#[export]
pub struct RecognizerWrapper {
    seg: parking_lot::Mutex<object_recognizer::subject_cutout::Segmenter>,
    extractor: parking_lot::Mutex<object_recognizer::feature_extraction::FeatureExtractor>,
    vecdb: object_recognizer::vecdb::VecDb,
    _env: ort::environment::Environment,
    timings: parking_lot::Mutex<HashMap<&'static str, f64>>,
}

fn assert_send_sync<T: Send + Sync>() {}

#[export]
impl RecognizerWrapper {
    fn from_recognizer(
        rec: object_recognizer::Recognizer,
        operation: &'static str,
        started: Instant,
    ) -> Self {
        let (seg, extractor, vecdb, _env) = rec.into_parts();
        let mut timings: HashMap<_, _> =
            TIMED_OPERATIONS.iter().map(|&name| (name, -1.0)).collect();
        timings.insert(operation, started.elapsed().as_secs_f64() * 1000.0);
        Self {
            seg: parking_lot::Mutex::new(seg),
            extractor: parking_lot::Mutex::new(extractor),
            vecdb,
            _env,
            timings: parking_lot::Mutex::new(timings),
        }
    }

    pub fn new(segmenter_model_path: String, vecdb_path: String) -> Result<Self, RecognizerError> {
        let started = Instant::now();
        assert_send_sync::<RecognizerWrapper>();
        let rec = object_recognizer::Recognizer::new(segmenter_model_path, vecdb_path)
            .map_err(RecognizerError::initialization)?;
        Ok(Self::from_recognizer(rec, "new", started))
    }

    pub fn new_embedded(vecdb_path: String) -> Result<Self, RecognizerError> {
        let started = Instant::now();
        assert_send_sync::<RecognizerWrapper>();
        let rec = object_recognizer::Recognizer::new_embedded(vecdb_path)
            .map_err(RecognizerError::initialization)?;
        Ok(Self::from_recognizer(rec, "new_embedded", started))
    }

    pub fn new_embedded_lite(vecdb_path: String) -> Result<Self, RecognizerError> {
        let started = Instant::now();
        assert_send_sync::<RecognizerWrapper>();
        let rec = object_recognizer::Recognizer::new_embedded_lite(vecdb_path)
            .map_err(RecognizerError::initialization)?;
        Ok(Self::from_recognizer(rec, "new_embedded_lite", started))
    }

    pub fn new_embedded_default() -> Result<Self, RecognizerError> {
        let started = Instant::now();
        assert_send_sync::<RecognizerWrapper>();
        let rec = object_recognizer::Recognizer::new_embedded_default()
            .map_err(RecognizerError::initialization)?;
        Ok(Self::from_recognizer(rec, "new_embedded_default", started))
    }

    pub fn new_default(segmenter_model_path: String) -> Result<Self, RecognizerError> {
        let started = Instant::now();
        assert_send_sync::<RecognizerWrapper>();
        let rec = object_recognizer::Recognizer::new_default(segmenter_model_path)
            .map_err(RecognizerError::initialization)?;
        Ok(Self::from_recognizer(rec, "new_default", started))
    }

    fn start_timing(&self, operation: &'static str) -> TimingGuard<'_> {
        TimingGuard {
            timings: &self.timings,
            operation,
            started: Instant::now(),
        }
    }

    pub fn get_timings(&self) -> Vec<OperationTiming> {
        let timings = self.timings.lock();
        TIMED_OPERATIONS
            .iter()
            .map(|&name| OperationTiming {
                name: name.to_string(),
                elapsed_ms: timings[name],
            })
            .collect()
    }

    pub fn segment_subject_raw(
        &self,
        image_bytes: Vec<u8>,
    ) -> Result<Vec<Vec<u8>>, RecognizerError> {
        let _timing = self.start_timing("segment_subject_raw");
        let cutouts = self.cutouts(&decode(&image_bytes)?)?;
        Ok(cutouts.into_iter().map(|c| c.into_raw()).collect())
    }

    /// Same as `segment_subject_raw`, from BGR24 pixels (see `image_from_bgr24`).
    pub fn segment_subject_pixels(
        &self,
        pixels: Vec<u8>,
        width: u32,
        height: u32,
        stride: u32,
    ) -> Result<Vec<Vec<u8>>, RecognizerError> {
        let _timing = self.start_timing("segment_subject_pixels");
        let cutouts = self.cutouts(&image_from_bgr24(&pixels, width, height, stride)?)?;
        Ok(cutouts.into_iter().map(|c| c.into_raw()).collect())
    }

    pub fn generate_embedding(&self, image_bytes: Vec<u8>) -> Result<Vec<f32>, RecognizerError> {
        let _timing = self.start_timing("generate_embedding");
        self.embed(&decode(&image_bytes)?)
    }

    /// Same as `generate_embedding`, from BGR24 pixels (see `image_from_bgr24`).
    pub fn generate_embedding_pixels(
        &self,
        pixels: Vec<u8>,
        width: u32,
        height: u32,
        stride: u32,
    ) -> Result<Vec<f32>, RecognizerError> {
        let _timing = self.start_timing("generate_embedding_pixels");
        self.embed(&image_from_bgr24(&pixels, width, height, stride)?)
    }

    pub fn vector_db_insert(
        &self,
        vector: Vec<f32>,
        metadata_json: String,
    ) -> Result<u64, RecognizerError> {
        let _timing = self.start_timing("vector_db_insert");
        let meta: serde_json::Value =
            serde_json::from_str(&metadata_json).map_err(RecognizerError::invalid_metadata)?;
        let ids = self.vecdb.add_with(std::iter::once((vector, meta)))?;
        Ok(ids[0])
    }

    pub fn vector_db_search(
        &self,
        query_vector: Vec<f32>,
        k: usize,
    ) -> Result<Vec<HitDto>, RecognizerError> {
        let _timing = self.start_timing("vector_db_search");
        let hits = self
            .vecdb
            .recognize::<serde_json::Value>(query_vector, k, 0.0)?;
        hit_dtos(hits)
    }

    pub fn vector_db_len(&self) -> Result<usize, RecognizerError> {
        let _timing = self.start_timing("vector_db_len");
        Ok(self.vecdb.count()?)
    }

    pub fn vector_db_count_by(&self, key: String) -> Result<Vec<KeyCount>, RecognizerError> {
        let _timing = self.start_timing("vector_db_count_by");
        let counts = self.vecdb.count_by_field(&key)?;
        Ok(counts
            .into_iter()
            .map(|(value, count)| KeyCount { value, count })
            .collect())
    }

    pub fn vector_db_delete_by_id(&self, id: u64) -> Result<bool, RecognizerError> {
        let _timing = self.start_timing("vector_db_delete_by_id");
        Ok(self.vecdb.delete_by_id(id)?)
    }

    pub fn vector_db_delete_where(
        &self,
        key: String,
        value: String,
    ) -> Result<usize, RecognizerError> {
        let _timing = self.start_timing("vector_db_delete_where");
        Ok(self.vecdb.delete_where(&key, value)?)
    }

    pub fn vector_db_save(&self, path: String) -> Result<(), RecognizerError> {
        let _timing = self.start_timing("vector_db_save");
        let _ = path;
        Ok(self.vecdb.optimize()?)
    }

    pub fn vector_db_export(&self, path: String) -> Result<usize, RecognizerError> {
        let _timing = self.start_timing("vector_db_export");
        Ok(self.vecdb.export(&path)?)
    }

    pub fn vector_db_import(&self, path: String, replace: bool) -> Result<usize, RecognizerError> {
        let _timing = self.start_timing("vector_db_import");
        let mode = if replace {
            object_recognizer::vecdb::ImportMode::Replace
        } else {
            object_recognizer::vecdb::ImportMode::Merge
        };
        Ok(self.vecdb.import(&path, mode)?)
    }

    pub fn vector_db_clear(&self) -> Result<(), RecognizerError> {
        let _timing = self.start_timing("vector_db_clear");
        Ok(self.vecdb.clear()?)
    }

    pub fn add_training_sample(
        &self,
        image_bytes: Vec<u8>,
        metadata_json: String,
    ) -> Result<Vec<u64>, RecognizerError> {
        let _timing = self.start_timing("add_training_sample");
        self.add_sample(&decode(&image_bytes)?, &metadata_json)
    }

    /// Same as `add_training_sample`, from BGR24 pixels (see `image_from_bgr24`).
    pub fn add_training_sample_pixels(
        &self,
        pixels: Vec<u8>,
        width: u32,
        height: u32,
        stride: u32,
        metadata_json: String,
    ) -> Result<Vec<u64>, RecognizerError> {
        let _timing = self.start_timing("add_training_sample_pixels");
        self.add_sample(
            &image_from_bgr24(&pixels, width, height, stride)?,
            &metadata_json,
        )
    }

    pub fn classify_image(
        &self,
        image_bytes: Vec<u8>,
        top_n: usize,
        min_score: f32,
    ) -> Result<Vec<HitDto>, RecognizerError> {
        let _timing = self.start_timing("classify_image");
        self.classify(&decode(&image_bytes)?, top_n, min_score)
    }

    /// Same as `classify_image`, from BGR24 pixels (see `image_from_bgr24`).
    pub fn classify_pixels(
        &self,
        pixels: Vec<u8>,
        width: u32,
        height: u32,
        stride: u32,
        top_n: usize,
        min_score: f32,
    ) -> Result<Vec<HitDto>, RecognizerError> {
        let _timing = self.start_timing("classify_pixels");
        self.classify(
            &image_from_bgr24(&pixels, width, height, stride)?,
            top_n,
            min_score,
        )
    }
}

/// Pipeline steps shared by the byte (encoded image) and pixel entry points.
impl RecognizerWrapper {
    /// Subject cutouts as 224x224 RGB squares, ready for the embedder.
    fn cutouts(&self, img: &DynamicImage) -> Result<Vec<RgbImage>, RecognizerError> {
        use object_recognizer::subject_cutout::{Segmenter, squares_from_saliency};
        let input = Segmenter::prepare_input(img);
        let raw = self.seg.lock().run_raw(input)?;
        let soft = Segmenter::saliency_from_raw(&raw)?;
        Ok(squares_from_saliency(img, &soft, Some(224))?)
    }

    fn embed(&self, img: &DynamicImage) -> Result<Vec<f32>, RecognizerError> {
        let input = object_recognizer::feature_extraction::FeatureExtractor::prepare_input(img);
        Ok(self.extractor.lock().run(input)?)
    }

    fn add_sample(
        &self,
        img: &DynamicImage,
        metadata_json: &str,
    ) -> Result<Vec<u64>, RecognizerError> {
        let meta: serde_json::Value =
            serde_json::from_str(metadata_json).map_err(RecognizerError::invalid_metadata)?;
        let embeddings = self
            .cutouts(img)?
            .into_iter()
            .map(|cutout| self.embed(&DynamicImage::ImageRgb8(cutout)))
            .collect::<Result<Vec<_>, _>>()?;
        if embeddings.is_empty() {
            return Err(RecognizerError::NoSubjectFound);
        }
        Ok(self
            .vecdb
            .add_with(embeddings.into_iter().map(|emb| (emb, meta.clone())))?)
    }

    fn classify(
        &self,
        img: &DynamicImage,
        top_n: usize,
        min_score: f32,
    ) -> Result<Vec<HitDto>, RecognizerError> {
        let mut best_hits = Vec::new();
        for cutout in self.cutouts(img)? {
            let embedding = self.embed(&DynamicImage::ImageRgb8(cutout))?;
            let hits = self
                .vecdb
                .recognize::<serde_json::Value>(embedding, top_n, min_score)?;
            best_hits.extend(hits);
        }
        hit_dtos(object_recognizer::merge_hits(best_hits, top_n))
    }
}

fn hit_dtos(
    hits: Vec<object_recognizer::vecdb::Hit<serde_json::Value>>,
) -> Result<Vec<HitDto>, RecognizerError> {
    hits.into_iter()
        .map(|h| {
            Ok(HitDto {
                id: h.id,
                score: h.score,
                metadata_json: serde_json::to_string(&h.meta)
                    .map_err(RecognizerError::serialization)?,
            })
        })
        .collect()
}

fn decode(image_bytes: &[u8]) -> Result<DynamicImage, RecognizerError> {
    Ok(image::load_from_memory(image_bytes)?)
}

/// RGB image from BGR24 pixels laid out like a .NET `Bitmap` locked as
/// `Format24bppRgb`: `stride` bytes per row, padding at the end of each row ignored.
/// The last row may omit its padding.
fn image_from_bgr24(
    pixels: &[u8],
    width: u32,
    height: u32,
    stride: u32,
) -> Result<DynamicImage, RecognizerError> {
    let (w, h, stride) = (width as usize, height as usize, stride as usize);
    let row = w * 3;
    let needed = stride
        .checked_mul(h.saturating_sub(1))
        .and_then(|n| n.checked_add(row));
    match needed {
        _ if w == 0 || h == 0 => {
            return Err(RecognizerError::image_decode(format!(
                "empty image {width}x{height}"
            )));
        }
        _ if stride < row => {
            return Err(RecognizerError::image_decode(format!(
                "stride {stride} is smaller than a {width}px BGR24 row ({row} bytes)"
            )));
        }
        Some(n) if pixels.len() >= n => {}
        _ => {
            return Err(RecognizerError::image_decode(format!(
                "{} bytes is too short for {width}x{height} BGR24 with stride {stride}",
                pixels.len()
            )));
        }
    }
    let mut rgb = Vec::with_capacity(row * h);
    for y in 0..h {
        for px in pixels[y * stride..y * stride + row].chunks_exact(3) {
            rgb.extend_from_slice(&[px[2], px[1], px[0]]);
        }
    }
    let img = RgbImage::from_raw(width, height, rgb).expect("buffer sized to width*height*3");
    Ok(DynamicImage::ImageRgb8(img))
}

#[cfg(test)]
mod pixel_tests {
    use super::*;

    #[test]
    fn bgr24_is_swapped_to_rgb_and_row_padding_skipped() {
        // 2x2 image, stride 8 (6 bytes of pixels + 2 padding), last row unpadded.
        let pixels = vec![
            1, 2, 3, 4, 5, 6, 0xEE, 0xEE, //
            7, 8, 9, 10, 11, 12,
        ];
        let img = image_from_bgr24(&pixels, 2, 2, 8).unwrap().into_rgb8();
        assert_eq!(img.as_raw(), &[3, 2, 1, 6, 5, 4, 9, 8, 7, 12, 11, 10]);
    }

    #[test]
    fn bad_buffers_are_rejected() {
        let ok = vec![0u8; 2 * 8];
        assert!(image_from_bgr24(&ok, 2, 2, 8).is_ok());
        assert!(image_from_bgr24(&ok, 0, 2, 8).is_err(), "zero width");
        assert!(image_from_bgr24(&ok, 2, 0, 8).is_err(), "zero height");
        assert!(
            image_from_bgr24(&ok, 3, 2, 8).is_err(),
            "stride smaller than row"
        );
        assert!(
            image_from_bgr24(&ok[..13], 2, 2, 8).is_err(),
            "short buffer"
        );
        assert!(
            image_from_bgr24(&ok, 2, 2, u32::MAX).is_err(),
            "huge stride"
        );
    }

    /// Needs ONNX Runtime (like `concurrency_tests`): the pixel entry points must give
    /// exactly what the byte entry points give for the same (losslessly encoded) frame.
    #[test]
    fn pixel_path_matches_byte_path() {
        let (w, h) = (320u32, 240u32);
        // Gray background with a colored block, so the segmenter has a subject.
        let rgb = RgbImage::from_fn(w, h, |x, y| {
            let inside = (100..220).contains(&x) && (60..180).contains(&y);
            image::Rgb(if inside {
                [200, 40, 30]
            } else {
                [120, 120, 120]
            })
        });
        let mut png = Vec::new();
        rgb.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        let stride = w * 3 + 4; // padded rows, like a .NET Bitmap
        let mut bgr = vec![0u8; (stride * h) as usize];
        for (x, y, p) in rgb.enumerate_pixels() {
            let i = (y * stride + x * 3) as usize;
            bgr[i..i + 3].copy_from_slice(&[p[2], p[1], p[0]]);
        }

        let dir = std::env::temp_dir().join(format!(
            "object-recognizer-ffi-pixels-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let wrapper = RecognizerWrapper::new_embedded(dir.to_string_lossy().into_owned()).unwrap();

        assert_eq!(
            wrapper.segment_subject_raw(png.clone()).unwrap(),
            wrapper
                .segment_subject_pixels(bgr.clone(), w, h, stride)
                .unwrap()
        );
        assert_eq!(
            wrapper.generate_embedding(png).unwrap(),
            wrapper
                .generate_embedding_pixels(bgr, w, h, stride)
                .unwrap()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
