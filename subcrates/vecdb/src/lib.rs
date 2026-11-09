use anyhow::Result;
use flate2::{Compression, read::GzDecoder, write::GzEncoder};
use qdrant_edge::*;
pub use qdrant_edge::{Condition, Filter, Match};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

// ---- Define your schema once, here ----
pub const DEFAULT_DIR: &str = "./vecdb";
pub const VECTOR_NAME: &str = "embedding";
pub const VECTOR_DIM: usize = 1280; // match your embedding model
const DISTANCE: Distance = Distance::Cosine;
/// Identifies the embedding model. Bump/rename it whenever you change the model:
/// exported files are only importable where this string matches exactly.
pub const EMBEDDING_MODEL: &str = "my-embedder-v1";
const EXPORT_FORMAT: u32 = 1;
/// Metadata fields that get a keyword index when a new db is created. Filters work on
/// any field, but indexed ones are much faster on large dbs. Use `index_field` to add
/// more to an existing db.
pub const INDEXED_FIELDS: &[&str] = &["name", "category"];

// ---- filter helpers: eq("category", "kitchen"), all_of([...]) ----

/// Anything usable as the value in `eq(..)` / `delete_where(..)`:
/// `&str`, `String`, integers and `bool`.
pub trait IntoMatch {
    fn into_match(self) -> Match;
}
impl IntoMatch for &str {
    fn into_match(self) -> Match {
        Match::from(self.to_string())
    }
}
impl IntoMatch for String {
    fn into_match(self) -> Match {
        Match::from(self)
    }
}
impl IntoMatch for i64 {
    fn into_match(self) -> Match {
        Match::from(self)
    }
}
impl IntoMatch for i32 {
    fn into_match(self) -> Match {
        Match::from(self as i64)
    }
}
impl IntoMatch for bool {
    fn into_match(self) -> Match {
        Match::from(self)
    }
}

/// Condition: metadata field equals a value (string, integer or bool).
pub fn eq(key: &str, value: impl IntoMatch) -> Result<Condition> {
    let k = key
        .try_into()
        .map_err(|_| anyhow::anyhow!("invalid payload key '{key}'"))?;
    Ok(Condition::Field(FieldCondition::new_match(
        k,
        value.into_match(),
    )))
}

/// All conditions must match (AND).
pub fn all_of(conds: impl IntoIterator<Item = Condition>) -> Filter {
    Filter {
        must: Some(conds.into_iter().collect()),
        ..Filter::new()
    }
}

/// At least one condition must match (OR).
pub fn any_of(conds: impl IntoIterator<Item = Condition>) -> Filter {
    Filter {
        should: Some(conds.into_iter().collect()),
        ..Filter::new()
    }
}

/// None of the conditions may match (NOT).
pub fn none_of(conds: impl IntoIterator<Item = Condition>) -> Filter {
    Filter {
        must_not: Some(conds.into_iter().collect()),
        ..Filter::new()
    }
}

fn create_keyword_index(shard: &EdgeShard, key: &str) -> Result<()> {
    let field_name = key
        .try_into()
        .map_err(|_| anyhow::anyhow!("invalid payload key '{key}'"))?;
    shard.update(UpdateOperation::FieldIndexOperation(
        FieldIndexOperations::CreateIndex(CreateIndex {
            field_name,
            field_schema: Some(PayloadSchemaType::Keyword.into()),
        }),
    ))?;
    Ok(())
}

#[derive(Serialize, Deserialize)]
struct ExportHeader {
    format: u32,
    model: String,
    vector_name: String,
    dim: usize,
    distance: String,
}

#[derive(Serialize, Deserialize)]
struct ExportRow {
    id: u64,
    vector: Vec<f32>,
    payload: serde_json::Value,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImportMode {
    /// Add imported points next to the local ones, with fresh ids (no collisions).
    Merge,
    /// Delete all local points first, then load the file keeping its original ids.
    Replace,
}

pub struct Hit<T> {
    pub id: u64,
    pub score: f32,
    pub meta: T,
}

pub struct VecDb {
    shard: EdgeShard,
    next_id: AtomicU64,
    id_file: PathBuf,
}

impl VecDb {
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        fs::create_dir_all(dir)?;
        let id_file = dir.join("next_id");

        let shard = if id_file.exists() {
            EdgeShard::load(dir, None)?
        } else {
            let config = EdgeConfigBuilder::new()
                .payload_memory(Memory::Cold)
                .vector(
                    VECTOR_NAME,
                    EdgeVectorParamsBuilder::new(VECTOR_DIM, DISTANCE)
                        .memory(Memory::Cold)
                        .build(),
                )
                .build();
            let s = EdgeShard::new(dir, config)?;
            for f in INDEXED_FIELDS {
                create_keyword_index(&s, f)?;
            }
            fs::write(&id_file, "0")?;
            s
        };

        let next = fs::read_to_string(&id_file)?.trim().parse().unwrap_or(0);
        Ok(Self {
            shard,
            next_id: AtomicU64::new(next),
            id_file,
        })
    }

    pub fn open_default() -> Result<Self> {
        Self::open(DEFAULT_DIR)
    }

    pub fn add(&self, vectors: Vec<Vec<f32>>) -> Result<Vec<u64>> {
        self.add_with(vectors.into_iter().map(|v| (v, serde_json::json!({}))))
    }

    pub fn add_with<P: Serialize>(
        &self,
        items: impl IntoIterator<Item = (Vec<f32>, P)>,
    ) -> Result<Vec<u64>> {
        let mut ids = Vec::new();
        let mut points: Vec<PointStructPersisted> = Vec::new();
        for (vec, payload) in items {
            check_dim(&vec)?;
            let id = self.next_id.fetch_add(1, Ordering::SeqCst);
            ids.push(id);
            points.push(
                PointStruct::new(
                    id,
                    Vectors::new_named([(VECTOR_NAME, vec)]),
                    serde_json::to_value(payload)?,
                )
                .into(),
            );
        }
        self.upsert_points(points)?;
        fs::write(
            &self.id_file,
            self.next_id.load(Ordering::SeqCst).to_string(),
        )?;
        Ok(ids)
    }

    pub fn upsert<P: Serialize>(&self, id: u64, vec: Vec<f32>, payload: P) -> Result<()> {
        check_dim(&vec)?;
        let p: PointStructPersisted = PointStruct::new(
            id,
            Vectors::new_named([(VECTOR_NAME, vec)]),
            serde_json::to_value(payload)?,
        )
        .into();
        self.upsert_points(vec![p])
    }

    pub fn search(&self, query: Vec<f32>, limit: usize) -> Result<Vec<ScoredPoint>> {
        self.search_filtered(query, limit, None)
    }

    pub fn search_filtered(
        &self,
        query: Vec<f32>,
        limit: usize,
        filter: Option<Filter>,
    ) -> Result<Vec<ScoredPoint>> {
        check_dim(&query)?;
        let mut req = QueryRequestBuilder::new(limit)
            .query(ScoringQuery::Vector(QueryEnum::Nearest(NamedQuery {
                query: query.into(),
                using: Some(VECTOR_NAME.to_string()),
            })))
            .with_payload(WithPayloadInterface::Bool(true));
        if let Some(f) = filter {
            req = req.filter(f);
        }
        Ok(self.shard.query(req.build())?)
    }

    /// Nearest matches above `min_score`. Empty => "unknown object".
    pub fn recognize<T: DeserializeOwned>(
        &self,
        query: Vec<f32>,
        limit: usize,
        min_score: f32,
    ) -> Result<Vec<Hit<T>>> {
        self.recognize_filtered(query, limit, min_score, None)
    }

    /// Like `recognize`, but only considers points matching `filter`
    /// (e.g. only objects of one category).
    pub fn recognize_filtered<T: DeserializeOwned>(
        &self,
        query: Vec<f32>,
        limit: usize,
        min_score: f32,
        filter: Option<Filter>,
    ) -> Result<Vec<Hit<T>>> {
        let mut out = Vec::new();
        for p in self.search_filtered(query, limit, filter)? {
            if p.score < min_score {
                continue;
            }
            let id = match p.id {
                PointId::NumId(n) => n,
                _ => continue,
            };
            let payload = p
                .payload
                .map(serde_json::to_value)
                .transpose()?
                .unwrap_or(serde_json::json!({}));
            out.push(Hit {
                id,
                score: p.score,
                meta: serde_json::from_value(payload)?,
            });
        }
        Ok(out)
    }

    pub fn delete(&self, ids: &[u64]) -> Result<()> {
        self.shard.update(UpdateOperation::PointOperation(
            PointOperations::DeletePoints {
                ids: ids.iter().map(|i| PointId::NumId(*i)).collect(),
            },
        ))?;
        Ok(())
    }

    /// Deletes every point matching `filter`; returns how many were deleted.
    /// (Space is reclaimed on the next `optimize`.)
    pub fn delete_by_filter(&self, filter: Filter) -> Result<usize> {
        let n = self.count_by_filter(filter.clone())?;
        self.shard.update(UpdateOperation::PointOperation(
            PointOperations::DeletePointsByFilter(filter),
        ))?;
        Ok(n)
    }

    /// Shortcut: delete all points whose metadata `key` equals `value`.
    ///   db.delete_where("name", "Red mug")?
    pub fn delete_where(&self, key: &str, value: impl IntoMatch) -> Result<usize> {
        self.delete_by_filter(all_of([eq(key, value)?]))
    }

    /// Number of points matching `filter` (exact).
    pub fn count_by_filter(&self, filter: Filter) -> Result<usize> {
        Ok(self.shard.count(
            CountRequestBuilder::new()
                .filter(filter)
                .exact(true)
                .build(),
        )?)
    }

    /// Total number of points.
    pub fn count(&self) -> Result<usize> {
        Ok(self
            .shard
            .count(CountRequestBuilder::new().exact(true).build())?)
    }

    /// Adds a keyword index on a metadata field of an existing db (speeds up filters).
    pub fn index_field(&self, key: &str) -> Result<()> {
        create_keyword_index(&self.shard, key)
    }

    pub fn optimize(&self) -> Result<()> {
        self.shard.optimize()?;
        Ok(())
    }

    /// Writes every point (vector + metadata) to a portable `.vecdb` file
    /// (gzipped JSON lines). Returns the number of points exported.
    pub fn export(&self, path: impl AsRef<Path>) -> Result<usize> {
        let path = path.as_ref();
        let tmp = path.with_extension("tmp");
        let mut w = GzEncoder::new(BufWriter::new(File::create(&tmp)?), Compression::default());

        let header = ExportHeader {
            format: EXPORT_FORMAT,
            model: EMBEDDING_MODEL.into(),
            vector_name: VECTOR_NAME.into(),
            dim: VECTOR_DIM,
            distance: format!("{DISTANCE:?}"),
        };
        writeln!(w, "{}", serde_json::to_string(&header)?)?;

        let mut offset: Option<PointId> = None;
        let mut n = 0;
        loop {
            let mut req = ScrollRequestBuilder::new()
                .limit(256)
                .with_payload(WithPayloadInterface::Bool(true))
                .with_vector(WithVector::Bool(true));
            if let Some(o) = offset.take() {
                req = req.offset(o);
            }
            let (records, next) = self.shard.scroll(req.build())?;
            for r in records {
                let id = match r.id {
                    PointId::NumId(n) => n,
                    other => anyhow::bail!("unsupported non-numeric point id: {other:?}"),
                };
                let vector = match r.vector {
                    Some(VectorStructInternal::Named(mut m)) => match m.remove(VECTOR_NAME) {
                        Some(VectorInternal::Dense(v)) => v,
                        _ => anyhow::bail!("point {id} has no dense '{VECTOR_NAME}' vector"),
                    },
                    _ => anyhow::bail!("point {id} has unexpected vector layout"),
                };
                let payload = r
                    .payload
                    .map(serde_json::to_value)
                    .transpose()?
                    .unwrap_or(serde_json::json!({}));
                writeln!(
                    w,
                    "{}",
                    serde_json::to_string(&ExportRow {
                        id,
                        vector,
                        payload
                    })?
                )?;
                n += 1;
            }
            match next {
                Some(o) => offset = Some(o),
                None => break,
            }
        }
        let mut f = w.finish()?;
        f.flush()?;
        drop(f);
        fs::rename(&tmp, path)?; // never leave a half-written export at `path`
        Ok(n)
    }

    /// Loads a `.vecdb` file. The whole file is validated before anything is
    /// modified, so a bad or incompatible file leaves the local db untouched.
    /// Returns the number of points imported.
    pub fn import(&self, path: impl AsRef<Path>, mode: ImportMode) -> Result<usize> {
        let mut lines = BufReader::new(GzDecoder::new(File::open(path)?)).lines();
        let header: ExportHeader = serde_json::from_str(
            &lines
                .next()
                .ok_or_else(|| anyhow::anyhow!("empty file"))??,
        )?;
        anyhow::ensure!(
            header.format == EXPORT_FORMAT,
            "unsupported export format {}",
            header.format
        );
        anyhow::ensure!(
            header.model == EMBEDDING_MODEL,
            "embedding model mismatch: file '{}' vs local '{}'",
            header.model,
            EMBEDDING_MODEL
        );
        anyhow::ensure!(
            header.dim == VECTOR_DIM,
            "dim mismatch: file {} vs local {VECTOR_DIM}",
            header.dim
        );
        anyhow::ensure!(
            header.distance == format!("{DISTANCE:?}"),
            "distance mismatch: file {} vs local {DISTANCE:?}",
            header.distance
        );

        let mut rows = Vec::new();
        for line in lines {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let row: ExportRow = serde_json::from_str(&line)?;
            check_dim(&row.vector)?;
            rows.push(row);
        }
        let n = rows.len();

        match mode {
            ImportMode::Merge => {
                self.add_with(rows.into_iter().map(|r| (r.vector, r.payload)))?;
            }
            ImportMode::Replace => {
                self.clear()?;
                let max_id = rows.iter().map(|r| r.id).max();
                let points: Vec<PointStructPersisted> = rows
                    .into_iter()
                    .map(|r| {
                        PointStruct::new(
                            r.id,
                            Vectors::new_named([(VECTOR_NAME, r.vector)]),
                            r.payload,
                        )
                        .into()
                    })
                    .collect();
                if !points.is_empty() {
                    self.upsert_points(points)?;
                }
                if let Some(m) = max_id {
                    self.next_id.fetch_max(m + 1, Ordering::SeqCst);
                    fs::write(
                        &self.id_file,
                        self.next_id.load(Ordering::SeqCst).to_string(),
                    )?;
                }
            }
        }
        self.optimize()?;
        Ok(n)
    }

    /// Deletes every point.
    pub fn clear(&self) -> Result<()> {
        let mut offset: Option<PointId> = None;
        loop {
            let mut req = ScrollRequestBuilder::new()
                .limit(1024)
                .with_payload(WithPayloadInterface::Bool(false))
                .with_vector(WithVector::Bool(false));
            if let Some(o) = offset.take() {
                req = req.offset(o);
            }
            let (records, next) = self.shard.scroll(req.build())?;
            let ids: Vec<PointId> = records.into_iter().map(|r| r.id).collect();
            if !ids.is_empty() {
                self.shard.update(UpdateOperation::PointOperation(
                    PointOperations::DeletePoints { ids },
                ))?;
            }
            match next {
                Some(o) => offset = Some(o),
                None => break,
            }
        }
        self.optimize()
    }

    fn upsert_points(&self, points: Vec<PointStructPersisted>) -> Result<()> {
        self.shard.update(UpdateOperation::PointOperation(
            PointOperations::UpsertPoints(PointInsertOperations::PointsList(points)),
        ))?;
        Ok(())
    }
}

fn check_dim(v: &[f32]) -> Result<()> {
    anyhow::ensure!(
        v.len() == VECTOR_DIM,
        "expected dim {VECTOR_DIM}, got {}",
        v.len()
    );
    Ok(())
}
