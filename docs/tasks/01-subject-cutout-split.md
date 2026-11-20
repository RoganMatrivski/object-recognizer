# Task 1: subject-cutout: split model call from post-processing

**Depends on:** nothing. Can run in parallel with Tasks 2 and 3.

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

## Problem

`instance_squares(&mut self, ..)` runs resize, normalize, `session.run`, Otsu, closing, connected components, hole fill, alpha, bleed and squaring all under the lock. Only `session.run` needs `&mut`.

## Steps

1. Read `mask_small`, `instance_squares`, `instance_cutout` fully (lib.rs lines ~223-539 were not reviewed yet).
2. Split `mask_small` into:
   - `fn prepare_input(img: &DynamicImage) -> Vec<f32>`: resize to `INPUT_SIZE`, divide by max, MEAN/STD, NCHW. Pure, no `self`.
   - `pub fn run_raw(&mut self, input: Vec<f32>) -> Result<Vec<f32>>`: build tensor, `session.run`, validate `len == n`, copy output into an owned `Vec<f32>`. The borrow from `outputs` must end inside this fn. Only part needing `&mut`.
   - `fn saliency_from_raw(raw: &[f32]) -> Result<GrayImage>`: min-max normalize plus the `MIN_SALIENCY_RANGE` early-out. Pure.
3. Keep `mask_small(&mut self, ..)` as a thin composition of the three (keep the tracing breakdown).
4. Refactor post-processing into pure associated/free fns, e.g. `pub fn squares_from_saliency(img: &DynamicImage, soft: &GrayImage, side: Option<u32>) -> Result<Vec<RgbImage>>`. Keep `instance_squares` and `instance_cutout` as wrappers.
5. Make the new fns public and document which one needs the session.

## Acceptance

- Existing tests pass.
- New test: `instance_squares(img)` equals `squares_from_saliency(img, &mask_small(img)?)` on a synthetic image.
