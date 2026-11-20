# Task 3: vecdb: concurrency correctness

**Depends on:** nothing. Can run in parallel with Tasks 1 and 2.

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

1. **Id counter race.** In `add_with`, `fetch_add` + `upsert_points` + `fs::write(id_file)` is not atomic across threads; an older counter value can overwrite a newer one on disk. Add a `write_lock: Mutex<()>` held for the whole of `add_with`, and write the file value under it. Cover the Replace branch of `import` (`fetch_max` + file write) with the same lock.
2. **Atomic id file.** Write `next_id.tmp`, then `fs::rename` over `next_id`.
3. **Replace-import visibility.** `clear()` then `upsert` exposes an empty or partial DB to concurrent readers. Add an `RwLock<()>` gate: `recognize*`/`search*` take read; Replace-import and `clear` take write. Merge-import and `add_with` do not need the write side.
4. **`Sync` check.** Add a test calling `fn assert_sync<T: Send + Sync>() {}` with `VecDb`. If `EdgeShard` is not `Sync`, report it and wrap the shard in an internal `RwLock`, keeping the public API `&self`.
5. **Tests:**
   - 8 threads calling `add_with` concurrently produce unique ids, and after reopen `next_id >= max_id + 1`.
   - Concurrent `recognize` during `import(Replace)` never sees an empty result when the file is non-empty.
6. **DECISION, ask the owner first:** `EMBEDDING_MODEL = "my-embedder-v1"` is a placeholder baked into exports. Renaming (e.g. `mobilenetv4-embedding-256-1280-v1`) breaks compatibility with existing exports; bump `EXPORT_FORMAT` or add a migration note.

## Acceptance

- All new tests pass; existing tests (`roundtrip`, `export`, `filter`) stay green.
- Step 6 is not done without an explicit answer from the owner.
