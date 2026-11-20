# Task 4: object-recognizer aggregator crate

**Depends on:** Tasks 1, 2, 3 (uses their new APIs).

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

1. Add `Recognizer::into_parts(self) -> (Segmenter, FeatureExtractor, VecDb, Environment)`.
2. Mark `classify_image` / `add_training_sample` (the full-pipeline `&mut self` versions) as single-threaded convenience, or remove them.
3. **DECISION, ask the owner first:** `vector_db_save(&self, _path)` ignores `path` and just calls `optimize()`. Either rename to `vector_db_optimize` or make it call `export(path)`. Update the FFI to match.
4. `classify_image` concatenates hits from all cutouts, so `top_n` applies per cutout. Add a pure, unit-tested merge fn: sort by score desc, dedupe by id keeping the best score, truncate to `top_n`.
5. `add_training_sample` returns all ids (`Vec<u64>`).

## Acceptance

- Merge fn has unit tests (duplicates, ties, fewer hits than `top_n`).
- `into_parts` compiles with the `pub` fields intact.
- Step 3 is not done without an explicit answer from the owner.
