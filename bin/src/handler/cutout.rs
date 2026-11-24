use axum::{
    Json,
    extract::{Multipart, Query, State},
};
use image::{DynamicImage, RgbImage};
use serde::{Deserialize, Serialize};
use utoipa::{IntoParams, ToSchema};
use utoipa_axum::{router::OpenApiRouter, routes};

use super::{
    HandlerState,
    common::{self, ImageUpload},
    error::{ApiError, ApiResult},
};

pub const TAG: &str = "cutout";

pub(super) fn router() -> OpenApiRouter<HandlerState> {
    OpenApiRouter::new().routes(routes!(cutout))
}

#[derive(Deserialize, IntoParams)]
#[into_params(parameter_in = Query)]
struct CutoutParams {
    /// Output square side in pixels. Omit to keep the subject's native size.
    side: Option<u32>,
}

#[derive(Serialize, ToSchema)]
struct CutoutResponse {
    /// One base64-encoded PNG per detected subject.
    images: Vec<String>,
}

/// Segment `img` into one squared image per detected subject.
///
/// Blocking; call from inside [`common::blocking`]. Reusable by combined endpoints.
pub(super) fn subject_squares(
    state: &HandlerState,
    img: &DynamicImage,
    side: Option<u32>,
) -> ApiResult<Vec<RgbImage>> {
    common::lock(&state.seg)?
        .instance_squares(img, side)
        .map_err(ApiError::internal)
}

/// Cut out every subject in an image.
///
/// Segments the uploaded image, splits it into individual subjects and returns
/// each one as a squared PNG.
#[utoipa::path(
    post,
    path = "/cutout",
    tag = TAG,
    params(CutoutParams),
    request_body(content = inline(ImageUpload), content_type = "multipart/form-data"),
    responses(
        (status = 200, description = "One squared PNG per detected subject", body = CutoutResponse),
        (status = 400, description = "Malformed multipart body, missing `image` field, or undecodable image", body = String, content_type = "text/plain"),
        (status = 413, description = "Upload exceeds the 32 MiB limit", body = String, content_type = "text/plain"),
        (status = 500, description = "Segmentation or encoding failed", body = String, content_type = "text/plain"),
    )
)]
async fn cutout(
    State(state): State<HandlerState>,
    Query(params): Query<CutoutParams>,
    mut multipart: Multipart,
) -> ApiResult<Json<CutoutResponse>> {
    let bytes = common::read_image_field(&mut multipart).await?;

    let images = common::blocking(move || {
        let img = common::decode_image(&bytes)?;
        subject_squares(&state, &img, params.side)?
            .iter()
            .map(common::png_base64)
            .collect()
    })
    .await?;

    Ok(Json(CutoutResponse { images }))
}
