use axum::{
    extract::{Path, State},
    http::StatusCode,
};
use utoipa_axum::{router::OpenApiRouter, routes};

use super::{TAG, db_err};
use crate::handler::{HandlerState, common, error::ApiResult};

pub(super) fn router() -> OpenApiRouter<HandlerState> {
    OpenApiRouter::new()
        .routes(routes!(optimize))
        .routes(routes!(index_field))
}

/// Optimize storage.
///
/// Compacts the database and reclaims space left by deleted points.
#[utoipa::path(
    post,
    path = "/vecdb/optimize",
    tag = TAG,
    responses(
        (status = 204, description = "Database optimized"),
        (status = 500, description = "Database error", body = String, content_type = "text/plain"),
    )
)]
async fn optimize(State(state): State<HandlerState>) -> ApiResult<StatusCode> {
    common::blocking(move || state.vecdb.optimize().map_err(db_err)).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Index a payload field.
///
/// Adds a keyword index so filters on this field are fast on large
/// databases. `name` and `category` are indexed by default.
#[utoipa::path(
    post,
    path = "/vecdb/index/{field}",
    tag = TAG,
    params(("field" = String, Path, description = "Top-level payload field to index")),
    responses(
        (status = 204, description = "Index created"),
        (status = 400, description = "Invalid field name", body = String, content_type = "text/plain"),
        (status = 500, description = "Database error", body = String, content_type = "text/plain"),
    )
)]
async fn index_field(
    State(state): State<HandlerState>,
    Path(field): Path<String>,
) -> ApiResult<StatusCode> {
    common::blocking(move || state.vecdb.index_field(&field).map_err(db_err)).await?;
    Ok(StatusCode::NO_CONTENT)
}
