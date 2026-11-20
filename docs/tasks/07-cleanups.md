# Task 7: cleanups (low priority, separate PRs)

**Depends on:** nothing. Can be done anytime.

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

1. Stale docs: `variables.rs` and the `lib.rs` table say 320x320, but `INPUT_SIZE` is 256.
2. **DECISION, ask the owner first:** `SQUARE_BG` is the ImageNet mean (assumes mean/std normalization), but `FeatureExtractor::extract` only divides by 255. Check whether the ONNX graph bakes in normalization. If not, either add it in `prepare_input` or change `SQUARE_BG`. Either choice changes embeddings and invalidates existing DBs.
3. Remove the unused `_segmenter_model_path` param on `Recognizer::from_parts`.
4. `create_segmenter_from_path` hardcodes `Model::U2Net`; accept a `Model` argument.
5. Intra-threads env parsing is duplicated in three places; extract a shared helper.

## Acceptance

- Each item is its own small PR.
- Step 2 is not done without an explicit answer from the owner.
