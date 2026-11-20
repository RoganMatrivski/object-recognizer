# Task 5: FFI crate (RecognizerWrapper)

**Depends on:** Tasks 1, 2, 3, 4.

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

1. New struct:

```rust
pub struct RecognizerWrapper {
    seg: Mutex<Segmenter>,
    extractor: Mutex<FeatureExtractor>,
    vecdb: VecDb,                          // no lock, unless Task 3.4 says otherwise
    _env: ort::environment::Environment,   // declared LAST so it drops last
}
```

2. Constructors call `Recognizer::new*`, then `into_parts()`, then wrap.
3. Use `parking_lot::Mutex` for `seg` and `extractor` (no poisoning) unless boltffi needs std types.
4. Rewrite methods so a lock guard lives for one statement only:
   - `segment_subject_raw`: decode, `prepare_input`, **lock** `seg.run_raw`, then `saliency_from_raw` + `squares_from_saliency` unlocked, then `into_raw()` each.
   - `generate_embedding`: decode, `prepare_input` unlocked, **lock** `extractor.run`.
   - `classify_image`: stage segment, then per cutout embed (staged), then `vecdb.recognize` with no model lock held, then the Task 4.4 merge.
   - `add_training_sample`: same staging, collect all embeddings, one `vecdb.add_with` call, return `Vec<u64>`.
   - `vector_db_*`: plain calls on `self.vecdb`, no model locks.
5. Rules:
   - Never hold two model locks at once.
   - Never call `vecdb` while holding a model lock.
6. Remove the `unwrap_or(0)` id hack.
7. Remove `unwrap_or_default()` on `serde_json::to_string(&h.meta)`; propagate the error.
8. Add a compile-time `Send + Sync` assertion for `RecognizerWrapper`.

## Acceptance

- Compiles, FFI bindings regenerate.
- No method holds a model lock across a `vecdb` call or a second model lock.
- Return type of `add_training_sample` is `Vec<u64>` on the FFI side; note the API change for the app-side callers.
