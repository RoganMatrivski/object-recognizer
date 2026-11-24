use axum::{
    Json,
    extract::{Path, State},
};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use utoipa_axum::{router::OpenApiRouter, routes};

use super::{FieldFilter, TAG, db_err, to_filter};
use crate::handler::{
    HandlerState,
    common,
    error::{ApiError, ApiResult},
};

pub(super) fn router() -> OpenApiRouter<HandlerState> {
    OpenApiRouter::new()
        .routes(routes!(search))
        .routes(routes!(count))
        .routes(routes!(count_by_field))
}

fn default_limit() -> usize {
    5
}

#[derive(Deserialize, ToSchema)]
struct SearchRequest {
    /// Query vector (1280 floats).
    vector: Vec<f32>,
    /// Max number of hits.
    #[serde(default = "default_limit")]
    #[schema(default = 5, minimum = 1)]
    limit: usize,
    /// Drop hits scoring below this (cosine similarity, -1..1).
    #[serde(default)]
    #[schema(default = 0.0)]
    min_score: f32,
    /// Only consider points whose payload matches every field.
    #[serde(default)]
    #[schema(value_type = HashMap<String, super::FilterValue>)]
    filter: FieldFilter,
}

#[derive(Serialize, ToSchema)]
struct SearchHit {
    id: u64,
    /// Cosine similarity, higher is closer.
    score: f32,
    #[schema(value_type = Object)]
    payload: serde_json::Value,
}

#[derive(Serialize, ToSchema)]
struct SearchResponse {
    /// Best match first. Empty means nothing scored above `min_score`.
    hits: Vec<SearchHit>,
}

#[derive(Serialize, ToSchema)]
struct CountResponse {
    count: usize,
}

#[derive(Serialize, ToSchema)]
struct FieldCount {
    /// A distinct string value of the field.
    value: String,
    /// Number of points with that value.
    count: usize,
}

/// Search nearest points.
#[utoipa::path(
    post,
    path = "/vecdb/search",
    tag = TAG,
    request_body = SearchRequest,
    responses(
        (status = 200, description = "Nearest points, best first", body = SearchResponse),
        (status = 400, description = "Wrong vector length, `limit` = 0, or invalid filter field", body = String, content_type = "text/plain"),
        (status = 422, description = "Body is not a valid request", body = String, content_type = "text/plain"),
        (status = 500, description = "Database error", body = String, content_type = "text/plain"),
    )
)]
async fn search(
    State(state): State<HandlerState>,
    Json(req): Json<SearchRequest>,
) -> ApiResult<Json<SearchResponse>> {
    if req.limit == 0 {
        return Err(ApiError::bad_request("`limit` must be at least 1"));
    }
    let filter = to_filter(req.filter)?;
    let hits = common::blocking(move || {
        state
            .vecdb
            .recognize_filtered::<serde_json::Value>(req.vector, req.limit, req.min_score, filter)
            .map_err(db_err)
    })
    .await?;

    Ok(Json(SearchResponse {
        hits: hits
            .into_iter()
            .map(|h| SearchHit {
                id: h.id,
                score: h.score,
                payload: h.meta,
            })
            .collect(),
    }))
}

/// Count all points.
#[utoipa::path(
    get,
    path = "/vecdb/count",
    tag = TAG,
    responses(
        (status = 200, description = "Total number of points", body = CountResponse),
        (status = 500, description = "Database error", body = String, content_type = "text/plain"),
    )
)]
async fn count(State(state): State<HandlerState>) -> ApiResult<Json<CountResponse>> {
    let count = common::blocking(move || state.vecdb.count().map_err(db_err)).await?;
    Ok(Json(CountResponse { count }))
}

/// Count points per value of a payload field.
///
/// Only string values are counted. Scans the whole database.
#[utoipa::path(
    get,
    path = "/vecdb/count/{field}",
    tag = TAG,
    params(("field" = String, Path, description = "Top-level payload field, e.g. `name`")),
    responses(
        (status = 200, description = "Counts sorted by value", body = [FieldCount]),
        (status = 500, description = "Database error", body = String, content_type = "text/plain"),
    )
)]
async fn count_by_field(
    State(state): State<HandlerState>,
    Path(field): Path<String>,
) -> ApiResult<Json<Vec<FieldCount>>> {
    let counts =
        common::blocking(move || state.vecdb.count_by_field(&field).map_err(db_err)).await?;
    Ok(Json(
        counts
            .into_iter()
            .map(|(value, count)| FieldCount { value, count })
            .collect(),
    ))
}
