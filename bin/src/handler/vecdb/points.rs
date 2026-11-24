use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use utoipa_axum::{router::OpenApiRouter, routes};

use super::{FieldFilter, TAG, db_err, empty_object, to_filter};
use crate::handler::{
    HandlerState,
    common,
    error::{ApiError, ApiResult},
};

pub(super) fn router() -> OpenApiRouter<HandlerState> {
    OpenApiRouter::new()
        .routes(routes!(add_points, clear_points))
        .routes(routes!(upsert_point, delete_point))
        .routes(routes!(delete_by_filter))
}

#[derive(Deserialize, ToSchema)]
struct PointInput {
    /// Feature vector (1280 floats), e.g. from `/embedding`.
    vector: Vec<f32>,
    /// Arbitrary JSON metadata. `name` and `category` are indexed.
    #[serde(default = "empty_object")]
    #[schema(value_type = Object)]
    payload: serde_json::Value,
}

#[derive(Deserialize, ToSchema)]
struct AddPointsRequest {
    points: Vec<PointInput>,
}

#[derive(Serialize, ToSchema)]
struct AddPointsResponse {
    /// Ids assigned to the new points, in request order.
    ids: Vec<u64>,
}

#[derive(Deserialize, ToSchema)]
struct DeleteByFilterRequest {
    /// Must not be empty; use `DELETE /vecdb/points` to delete everything.
    #[schema(value_type = HashMap<String, super::FilterValue>)]
    filter: FieldFilter,
}

#[derive(Serialize, ToSchema)]
struct DeletedResponse {
    /// Number of points deleted.
    deleted: usize,
}

/// Add points.
///
/// Stores each vector with its payload under a newly assigned id.
#[utoipa::path(
    post,
    path = "/vecdb/points",
    tag = TAG,
    request_body = AddPointsRequest,
    responses(
        (status = 200, description = "Points added", body = AddPointsResponse),
        (status = 400, description = "A vector has the wrong length", body = String, content_type = "text/plain"),
        (status = 422, description = "Body is not a valid request", body = String, content_type = "text/plain"),
        (status = 500, description = "Database error", body = String, content_type = "text/plain"),
    )
)]
async fn add_points(
    State(state): State<HandlerState>,
    Json(req): Json<AddPointsRequest>,
) -> ApiResult<Json<AddPointsResponse>> {
    let ids = common::blocking(move || {
        state
            .vecdb
            .add_with(req.points.into_iter().map(|p| (p.vector, p.payload)))
            .map_err(db_err)
    })
    .await?;
    Ok(Json(AddPointsResponse { ids }))
}

/// Delete all points.
#[utoipa::path(
    delete,
    path = "/vecdb/points",
    tag = TAG,
    responses(
        (status = 204, description = "Database cleared"),
        (status = 500, description = "Database error", body = String, content_type = "text/plain"),
    )
)]
async fn clear_points(State(state): State<HandlerState>) -> ApiResult<StatusCode> {
    common::blocking(move || state.vecdb.clear().map_err(db_err)).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Insert or replace a point.
///
/// Writes the vector and payload under `id`, overwriting any existing point.
#[utoipa::path(
    put,
    path = "/vecdb/points/{id}",
    tag = TAG,
    params(("id" = u64, Path, description = "Point id")),
    request_body = PointInput,
    responses(
        (status = 204, description = "Point written"),
        (status = 400, description = "Vector has the wrong length", body = String, content_type = "text/plain"),
        (status = 422, description = "Body is not a valid point", body = String, content_type = "text/plain"),
        (status = 500, description = "Database error", body = String, content_type = "text/plain"),
    )
)]
async fn upsert_point(
    State(state): State<HandlerState>,
    Path(id): Path<u64>,
    Json(point): Json<PointInput>,
) -> ApiResult<StatusCode> {
    common::blocking(move || {
        state
            .vecdb
            .upsert(id, point.vector, point.payload)
            .map_err(db_err)
    })
    .await?;
    Ok(StatusCode::NO_CONTENT)
}

/// Delete a point by id.
#[utoipa::path(
    delete,
    path = "/vecdb/points/{id}",
    tag = TAG,
    params(("id" = u64, Path, description = "Point id")),
    responses(
        (status = 204, description = "Point deleted"),
        (status = 404, description = "No point with this id", body = String, content_type = "text/plain"),
        (status = 500, description = "Database error", body = String, content_type = "text/plain"),
    )
)]
async fn delete_point(
    State(state): State<HandlerState>,
    Path(id): Path<u64>,
) -> ApiResult<StatusCode> {
    let deleted = common::blocking(move || state.vecdb.delete_by_id(id).map_err(db_err)).await?;
    if deleted {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(ApiError::not_found(format!("no point with id {id}")))
    }
}

/// Delete points matching a payload filter.
#[utoipa::path(
    post,
    path = "/vecdb/points/delete",
    tag = TAG,
    request_body = DeleteByFilterRequest,
    responses(
        (status = 200, description = "Matching points deleted", body = DeletedResponse),
        (status = 400, description = "Empty filter or invalid field name", body = String, content_type = "text/plain"),
        (status = 422, description = "Body is not a valid request", body = String, content_type = "text/plain"),
        (status = 500, description = "Database error", body = String, content_type = "text/plain"),
    )
)]
async fn delete_by_filter(
    State(state): State<HandlerState>,
    Json(req): Json<DeleteByFilterRequest>,
) -> ApiResult<Json<DeletedResponse>> {
    let filter = to_filter(req.filter)?.ok_or_else(|| {
        ApiError::bad_request("`filter` must not be empty; use DELETE /vecdb/points to clear")
    })?;
    let deleted =
        common::blocking(move || state.vecdb.delete_by_filter(filter).map_err(db_err)).await?;
    Ok(Json(DeletedResponse { deleted }))
}
