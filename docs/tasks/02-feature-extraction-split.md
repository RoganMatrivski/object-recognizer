# Task 2: feature-extraction: shrink the critical section

**Depends on:** nothing. Can run in parallel with Tasks 1 and 3.

## Context

Workspace crates:
- `object-recognizer`: aggregates `Recognizer { seg, extractor, vecdb, env }`
- `object-recognizer-subject-cutout`: `Segmenter`, U2Net via `ort` 2.0.0-rc.14
- `object-recognizer-feature-extraction`: `FeatureExtractor`
- `object-recognizer-vecdb`: `VecDb` over `qdrant_edge::EdgeShard`
- FFI crate: `RecognizerWrapper` (boltffi)

**Goal:** replace the single `Mutex<Recognizer>` with fine-grained locking so searches and DB ops never wait on inference, and so locks cover only `Session::run`.

**Constraints:**
- In `ort` rc.14, `Session::run` takes `&mut self`.
- Do not change public behavior or model constants.
- Keep existing tests green.
- Do not introduce a session pool.
- Never nest model locks.

## Steps

1. Split `extract` into:
   - `pub fn prepare_input(img: &DynamicImage) -> Vec<f32>`: `resize_exact` 256x256 Triangle, NCHW, /255. Pure.
   - `pub fn run(&mut self, input: Vec<f32>) -> Result<Vec<f32>, ExtractionError>`: tensor, run, shape check vs `EMBEDDING_DIM`, copy out.
2. Keep `extract(&mut self, img)` as `run(prepare_input(img))`.
3. Gate the `include_bytes!` in `config.rs` behind `feature = "embed-model"` so the crate builds without the model file when the feature is off.
4. Deduplicate the two constructors via a private builder helper.

## Acceptance

- `extract` returns the same vector as before.
- The shape-validation path is covered by a test where feasible.
