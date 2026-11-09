pub use boltffi::*;

boltffi::scaffolding!();

#[data]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HitDto {
    pub id: u64,
    pub score: f32,
    pub metadata_json: String,
}

#[export]
pub struct RecognizerWrapper {
    inner: std::sync::Mutex<object_recognizer::Recognizer>,
}

#[export]
impl RecognizerWrapper {
    pub fn new(segmenter_model_path: String, vecdb_path: String) -> Result<Self, String> {
        let inner = object_recognizer::Recognizer::new(segmenter_model_path, vecdb_path)
            .map_err(|e| e.to_string())?;
        Ok(Self {
            inner: std::sync::Mutex::new(inner),
        })
    }

    pub fn new_embedded(vecdb_path: String) -> Result<Self, String> {
        let inner =
            object_recognizer::Recognizer::new_embedded(vecdb_path).map_err(|e| e.to_string())?;
        Ok(Self {
            inner: std::sync::Mutex::new(inner),
        })
    }

    pub fn new_embedded_lite(vecdb_path: String) -> Result<Self, String> {
        let inner = object_recognizer::Recognizer::new_embedded_lite(vecdb_path)
            .map_err(|e| e.to_string())?;
        Ok(Self {
            inner: std::sync::Mutex::new(inner),
        })
    }

    pub fn new_embedded_default() -> Result<Self, String> {
        let inner =
            object_recognizer::Recognizer::new_embedded_default().map_err(|e| e.to_string())?;
        Ok(Self {
            inner: std::sync::Mutex::new(inner),
        })
    }

    pub fn new_default(segmenter_model_path: String) -> Result<Self, String> {
        let inner = object_recognizer::Recognizer::new_default(segmenter_model_path)
            .map_err(|e| e.to_string())?;
        Ok(Self {
            inner: std::sync::Mutex::new(inner),
        })
    }

    pub fn segment_subject(&self, image_bytes: Vec<u8>) -> Result<Vec<u8>, String> {
        let img = image::load_from_memory(&image_bytes).map_err(|e| e.to_string())?;
        let mut inner = self.inner.lock().map_err(|e| e.to_string())?;
        let cutout = inner.segment_subject(&img).map_err(|e| e.to_string())?;
        let mut bytes = Vec::new();
        cutout
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Png,
            )
            .map_err(|e| e.to_string())?;
        Ok(bytes)
    }

    pub fn generate_embedding(&self, image_bytes: Vec<u8>) -> Result<Vec<f32>, String> {
        let img = image::load_from_memory(&image_bytes).map_err(|e| e.to_string())?;
        let mut inner = self.inner.lock().map_err(|e| e.to_string())?;
        let embedding = inner.generate_embedding(&img).map_err(|e| e.to_string())?;
        Ok(embedding)
    }

    pub fn vector_db_insert(&self, vector: Vec<f32>, metadata_json: String) -> Result<u64, String> {
        let mut inner = self.inner.lock().map_err(|e| e.to_string())?;
        let meta: serde_json::Value =
            serde_json::from_str(&metadata_json).map_err(|e| e.to_string())?;
        let id = inner
            .vector_db_insert(&vector, &meta)
            .map_err(|e| e.to_string())?;
        Ok(id)
    }

    pub fn vector_db_search(
        &self,
        query_vector: Vec<f32>,
        k: usize,
    ) -> Result<Vec<HitDto>, String> {
        let inner = self.inner.lock().map_err(|e| e.to_string())?;
        let hits = inner
            .vector_db_search(&query_vector, k)
            .map_err(|e| e.to_string())?;
        let mut dtos = Vec::new();
        for h in hits {
            dtos.push(HitDto {
                id: h.id,
                score: h.score,
                metadata_json: serde_json::to_string(&h.meta).unwrap_or_default(),
            });
        }
        Ok(dtos)
    }

    pub fn vector_db_save(&self, path: String) -> Result<(), String> {
        let inner = self.inner.lock().map_err(|e| e.to_string())?;
        inner.vector_db_save(&path).map_err(|e| e.to_string())
    }

    pub fn vector_db_export(&self, path: String) -> Result<usize, String> {
        let inner = self.inner.lock().map_err(|e| e.to_string())?;
        let count = inner.vecdb.export(&path).map_err(|e| e.to_string())?;
        Ok(count)
    }

    pub fn vector_db_import(&self, path: String, replace: bool) -> Result<usize, String> {
        let inner = self.inner.lock().map_err(|e| e.to_string())?;
        let mode = if replace {
            object_recognizer::object_recognizer_vecdb::ImportMode::Replace
        } else {
            object_recognizer::object_recognizer_vecdb::ImportMode::Merge
        };
        let count = inner.vecdb.import(&path, mode).map_err(|e| e.to_string())?;
        Ok(count)
    }

    pub fn add_training_sample(
        &self,
        image_bytes: Vec<u8>,
        metadata_json: String,
    ) -> Result<u64, String> {
        let img = image::load_from_memory(&image_bytes).map_err(|e| e.to_string())?;
        let meta: serde_json::Value =
            serde_json::from_str(&metadata_json).map_err(|e| e.to_string())?;
        let mut inner = self.inner.lock().map_err(|e| e.to_string())?;
        let id = inner
            .add_training_sample(&img, &meta)
            .map_err(|e| e.to_string())?;
        Ok(id)
    }

    pub fn classify_image(
        &self,
        image_bytes: Vec<u8>,
        top_n: usize,
        min_score: f32,
    ) -> Result<Vec<HitDto>, String> {
        let img = image::load_from_memory(&image_bytes).map_err(|e| e.to_string())?;
        let mut inner = self.inner.lock().map_err(|e| e.to_string())?;
        let hits: Vec<object_recognizer::object_recognizer_vecdb::Hit<serde_json::Value>> = inner
            .classify_image(&img, top_n, min_score)
            .map_err(|e| e.to_string())?;
        let mut dtos = Vec::new();
        for h in hits {
            dtos.push(HitDto {
                id: h.id,
                score: h.score,
                metadata_json: serde_json::to_string(&h.meta).unwrap_or_default(),
            });
        }
        Ok(dtos)
    }
}
