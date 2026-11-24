use axum::{
    Json,
    extract::{DefaultBodyLimit, Multipart, Query, State},
    http::header,
    response::IntoResponse,
};
use object_recognizer::vecdb::ImportMode;
use serde::{Deserialize, Serialize};
use std::io::Write;
use utoipa::{IntoParams, ToSchema};
use utoipa_axum::{router::OpenApiRouter, routes};

use super::{TAG, db_err};
use crate::handler::{
    HandlerState,
    common,
    error::{ApiError, ApiResult},
};

/// Max accepted `.vecdb` upload size. Keep the 413 description in sync.
const MAX_IMPORT_BYTES: usize = 1024 * 1024 * 1024;

pub(super) fn router() -> OpenApiRouter<HandlerState> {
    OpenApiRouter::new().routes(routes!(export)).merge(
        // Innermost body limit wins, so this overrides the global one.
        OpenApiRouter::new()
            .routes(routes!(import))
            .layer(DefaultBodyLimit::max(MAX_IMPORT_BYTES)),
    )
}

#[derive(Deserialize, ToSchema, Default, Clone, Copy)]
#[serde(rename_all = "lowercase")]
enum Mode {
    /// Keep local points; add imported ones with fresh ids.
    #[default]
    Merge,
    /// Delete all local points first, keeping the file's ids.
    Replace,
}

#[derive(Deserialize, IntoParams)]
#[into_params(parameter_in = Query)]
struct ImportParams {
    #[serde(default)]
    #[param(inline)]
    mode: Mode,
}

// Documentation only, like `common::ImageUpload`.
#[derive(ToSchema)]
#[allow(dead_code)]
struct VecDbUpload {
    /// A `.vecdb` file from `GET /vecdb/export`.
    #[schema(value_type = String, format = Binary, content_media_type = "application/gzip")]
    file: Vec<u8>,
}

#[derive(Serialize, ToSchema)]
struct ImportResponse {
    /// Number of points imported.
    imported: usize,
}

/// Export the database.
///
/// Downloads every point (vector + payload) as a portable `.vecdb` file
/// (gzipped JSON lines), importable with `POST /vecdb/import`.
#[utoipa::path(
    get,
    path = "/vecdb/export",
    tag = TAG,
    responses(
        (status = 200, description = "The `.vecdb` file", content_type = "application/gzip", body = String),
        (status = 500, description = "Database or I/O error", body = String, content_type = "text/plain"),
    )
)]
async fn export(State(state): State<HandlerState>) -> ApiResult<impl IntoResponse> {
    let bytes = common::blocking(move || {
        let dir = tempfile::tempdir().map_err(ApiError::internal)?;
        let path = dir.path().join("export.vecdb");
        state.vecdb.export(&path).map_err(db_err)?;
        std::fs::read(&path).map_err(ApiError::internal)
    })
    .await?;

    Ok((
        [
            (header::CONTENT_TYPE, "application/gzip"),
            (
                header::CONTENT_DISPOSITION,
                "attachment; filename=\"export.vecdb\"",
            ),
        ],
        bytes,
    ))
}

/// Import a `.vecdb` file.
///
/// The whole file is validated before anything is written, so a bad or
/// incompatible file leaves the database untouched.
#[utoipa::path(
    post,
    path = "/vecdb/import",
    tag = TAG,
    params(ImportParams),
    request_body(content = inline(VecDbUpload), content_type = "multipart/form-data"),
    responses(
        (status = 200, description = "File imported", body = ImportResponse),
        (status = 400, description = "Missing `file` field, or not a valid/compatible `.vecdb` file", body = String, content_type = "text/plain"),
        (status = 413, description = "Upload exceeds the 1 GiB limit", body = String, content_type = "text/plain"),
        (status = 500, description = "Database or I/O error", body = String, content_type = "text/plain"),
    )
)]
async fn import(
    State(state): State<HandlerState>,
    Query(params): Query<ImportParams>,
    mut multipart: Multipart,
) -> ApiResult<Json<ImportResponse>> {
    let bytes = common::read_file_field(&mut multipart, "file").await?;
    let mode = match params.mode {
        Mode::Merge => ImportMode::Merge,
        Mode::Replace => ImportMode::Replace,
    };

    let imported = common::blocking(move || {
        let mut tmp = tempfile::NamedTempFile::new().map_err(ApiError::internal)?;
        tmp.write_all(&bytes).map_err(ApiError::internal)?;
        tmp.flush().map_err(ApiError::internal)?;
        state.vecdb.import(tmp.path(), mode).map_err(db_err)
    })
    .await?;

    Ok(Json(ImportResponse { imported }))
}
