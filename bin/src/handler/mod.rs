//! HTTP API. Each endpoint group lives in its own module and exposes
//! `router()`; this module wires them together with shared state and the
//! OpenAPI docs.
//!
//! Adding an endpoint group: create `<name>.rs` with a `TAG` and `router()`,
//! then add `mod <name>;`, its tag in [`ApiDoc`], and `.merge(<name>::router())`
//! in [`get_router`].

mod common;
mod cutout;
mod embedding;
mod error;
mod pipeline;
mod vecdb;

use axum::{Json, Router, extract::DefaultBodyLimit, routing::get};
use std::sync::{Arc, Mutex};
use utoipa::OpenApi;
use utoipa_axum::router::OpenApiRouter;
use utoipa_scalar::{Scalar, Servable};

use object_recognizer::{
    feature_extraction::FeatureExtractor,
    subject_cutout::{Model, Segmenter},
    vecdb::VecDb,
};

#[derive(Clone)]
pub(super) struct HandlerState {
    // Inference takes `&mut self`, so the models are shared behind mutexes.
    pub(super) seg: Arc<Mutex<Segmenter>>,
    pub(super) extractor: Arc<Mutex<FeatureExtractor>>,
    // `VecDb` locks internally.
    pub(super) vecdb: Arc<VecDb>,
}

#[derive(OpenApi)]
#[openapi(
    info(
        title = "object-recognizer",
        description = "Subject segmentation and object recognition API."
    ),
    tags(
        (name = cutout::TAG, description = "Subject segmentation and cutouts"),
        (name = embedding::TAG, description = "Image feature vectors"),
        (name = pipeline::TAG, description = "Several models chained in one request"),
        (name = vecdb::TAG, description = "Manual vector database access"),
    )
)]
struct ApiDoc;

pub fn get_router(vecdb_dir: impl AsRef<std::path::Path>) -> eyre::Result<Router> {
    let env = ort::init().build()?;

    let state = HandlerState {
        seg: Arc::new(Mutex::new(Segmenter::embedded_with_env(Model::U2Net, &env)?)),
        extractor: Arc::new(Mutex::new(FeatureExtractor::new_embedded_with_env(&env)?)),
        vecdb: Arc::new(VecDb::open(vecdb_dir)?),
    };

    let (router, api) = OpenApiRouter::with_openapi(ApiDoc::openapi())
        .merge(cutout::router())
        .merge(embedding::router())
        .merge(pipeline::router())
        .merge(vecdb::router())
        .split_for_parts();

    Ok(router
        .route(
            "/openapi.json",
            get({
                let api = api.clone();
                move || async move { Json(api) }
            }),
        )
        .merge(Scalar::with_url("/docs", api))
        .layer(DefaultBodyLimit::max(common::MAX_UPLOAD_BYTES))
        .with_state(state))
}
