//! Endpoints that chain several models in one request.

use axum::{
    Json,
    extract::{Multipart, Query, State},
};
use image::DynamicImage;
use serde::{Deserialize, Serialize};
use utoipa::{IntoParams, ToSchema};
use utoipa_axum::{router::OpenApiRouter, routes};

use super::{
    HandlerState,
    common::{self, ImageUpload},
    cutout, embedding,
    error::{ApiError, ApiResult},
};

pub const TAG: &str = "pipeline";

pub(super) fn router() -> OpenApiRouter<HandlerState> {
    OpenApiRouter::new().routes(routes!(cutout_embed))
}

fn default_side() -> u32 {
    224
}

#[derive(Deserialize, IntoParams)]
#[into_params(parameter_in = Query)]
struct CutoutEmbedParams {
    /// Side of each subject square, in pixels, before embedding.
    #[serde(default = "default_side")]
    #[param(default = 224, minimum = 1)]
    side: u32,
    /// Also return each subject's cutout as a base64 PNG.
    #[serde(default)]
    #[param(default = false)]
    images: bool,
}

#[derive(Serialize, ToSchema)]
struct Subject {
    /// Feature vector of this subject (1280 floats).
    embedding: Vec<f32>,
    /// Squared cutout as a base64 PNG. Only present with `images=true`.
    #[serde(skip_serializing_if = "Option::is_none")]
    image: Option<String>,
}

#[derive(Serialize, ToSchema)]
struct CutoutEmbedResponse {
    /// One entry per detected subject.
    subjects: Vec<Subject>,
}

/// Cut out every subject, then embed each one.
///
/// Equivalent to `/cutout` followed by `/embedding` on every returned square,
/// in a single request.
#[utoipa::path(
    post,
    path = "/cutout-embed",
    tag = TAG,
    params(CutoutEmbedParams),
    request_body(content = inline(ImageUpload), content_type = "multipart/form-data"),
    responses(
        (status = 200, description = "One embedding per detected subject", body = CutoutEmbedResponse),
        (status = 400, description = "Malformed multipart body, missing `image` field, undecodable image, or `side` = 0", body = String, content_type = "text/plain"),
        (status = 413, description = "Upload exceeds the 32 MiB limit", body = String, content_type = "text/plain"),
        (status = 500, description = "Segmentation, embedding or encoding failed", body = String, content_type = "text/plain"),
    )
)]
async fn cutout_embed(
    State(state): State<HandlerState>,
    Query(params): Query<CutoutEmbedParams>,
    mut multipart: Multipart,
) -> ApiResult<Json<CutoutEmbedResponse>> {
    if params.side == 0 {
        return Err(ApiError::bad_request("`side` must be at least 1"));
    }
    let bytes = common::read_image_field(&mut multipart).await?;

    let subjects = common::blocking(move || {
        let img = common::decode_image(&bytes)?;
        // The segmenter lock is released here, before any embedding runs.
        let squares = cutout::subject_squares(&state, &img, Some(params.side))?;

        squares
            .into_iter()
            .map(|sq| {
                let image = params.images.then(|| common::png_base64(&sq)).transpose()?;
                let embedding = embedding::embed(&state, &DynamicImage::ImageRgb8(sq))?;
                Ok(Subject { embedding, image })
            })
            .collect()
    })
    .await?;

    Ok(Json(CutoutEmbedResponse { subjects }))
}
