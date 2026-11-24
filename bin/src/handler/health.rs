//! Liveness check for clients that offload work to this server.

use axum::Json;
use serde::Serialize;
use utoipa::ToSchema;
use utoipa_axum::{router::OpenApiRouter, routes};

use super::HandlerState;

pub const TAG: &str = "health";

pub(super) fn router() -> OpenApiRouter<HandlerState> {
    OpenApiRouter::new().routes(routes!(health))
}

#[derive(Serialize, ToSchema)]
struct HealthResponse {
    /// Always `"ok"`.
    status: &'static str,
}

/// Check that the server is up.
///
/// Touches no model and no database, so it answers immediately even while
/// other requests are running inference.
#[utoipa::path(
    get,
    path = "/health",
    tag = TAG,
    responses(
        (status = 200, description = "Server is up", body = HealthResponse),
    )
)]
async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}
