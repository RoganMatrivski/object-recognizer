use axum::{
    Json,
    extract::{Multipart, State},
};
use image::DynamicImage;
use serde::Serialize;
use utoipa::ToSchema;
use utoipa_axum::{router::OpenApiRouter, routes};

use super::{
    HandlerState,
    common::{self, ImageUpload},
    error::{ApiError, ApiResult},
};

pub const TAG: &str = "embedding";

pub(super) fn router() -> OpenApiRouter<HandlerState> {
    OpenApiRouter::new().routes(routes!(embedding))
}

#[derive(Serialize, ToSchema)]
struct EmbeddingResponse {
    /// Feature vector (1280 floats).
    embedding: Vec<f32>,
}

/// Compute the feature vector of `img`.
///
/// Blocking; call from inside [`common::blocking`]. Reusable by combined endpoints.
pub(super) fn embed(state: &HandlerState, img: &DynamicImage) -> ApiResult<Vec<f32>> {
    common::lock(&state.extractor)?
        .extract(img)
        .map_err(ApiError::internal)
}

/// Embed a whole image.
///
/// Computes the feature vector of the uploaded image as-is, without cutting
/// out subjects first. Use `/cutout-embed` to embed each subject separately.
#[utoipa::path(
    post,
    path = "/embedding",
    tag = TAG,
    request_body(content = inline(ImageUpload), content_type = "multipart/form-data"),
    responses(
        (status = 200, description = "Feature vector of the image", body = EmbeddingResponse),
        (status = 400, description = "Malformed multipart body, missing `image` field, or undecodable image", body = String, content_type = "text/plain"),
        (status = 413, description = "Upload exceeds the 32 MiB limit", body = String, content_type = "text/plain"),
        (status = 500, description = "Embedding failed", body = String, content_type = "text/plain"),
    )
)]
async fn embedding(
    State(state): State<HandlerState>,
    mut multipart: Multipart,
) -> ApiResult<Json<EmbeddingResponse>> {
    let bytes = common::read_image_field(&mut multipart).await?;

    let embedding = common::blocking(move || {
        let img = common::decode_image(&bytes)?;
        embed(&state, &img)
    })
    .await?;

    Ok(Json(EmbeddingResponse { embedding }))
}
