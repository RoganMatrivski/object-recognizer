# Task 6: concurrency tests and benchmark

**Depends on:** Task 5.

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

1. While one thread loops `classify_image`, another thread's `vector_db_search` stays under ~20 ms p99 (previously blocked for the full inference time).
2. Two threads calling `classify_image` simultaneously give results identical to serial calls.
3. 30-second no-deadlock soak with 4 threads mixing `generate_embedding`, `segment_subject_raw`, `add_training_sample`, `vector_db_import(replace)`.
4. Benchmark images/s with 1, 2, 4 caller threads, before vs after, with `ORT_INTRA_THREADS` unset, 2, and 4. (Both sessions default to `min(cores, 4)` threads, so overlapping calls can oversubscribe cores.)
5. Report whether inference is the bottleneck. Only then consider anything beyond one mutex per model.

## Acceptance

- Tests 1-3 pass reliably (run each 20 times).
- Benchmark results delivered as a short table plus a one-paragraph conclusion.
