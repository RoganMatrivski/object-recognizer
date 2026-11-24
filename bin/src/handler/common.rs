//! Helpers shared by every image endpoint.

use axum::{
    body::Bytes,
    extract::{Multipart, multipart::MultipartError},
    http::StatusCode,
};
use base64::{Engine, engine::general_purpose::STANDARD as B64};
use image::{DynamicImage, RgbImage};
use std::{
    io::Cursor,
    sync::{Mutex, MutexGuard},
};
use utoipa::ToSchema;

use super::error::{ApiError, ApiResult};

/// Max accepted request body size. Keep the 413 descriptions in sync.
pub const MAX_UPLOAD_BYTES: usize = 32 * 1024 * 1024;

// Documentation only: handlers read the body via `Multipart`, which has no
// schema of its own. Use as
// `request_body(content = inline(ImageUpload), content_type = "multipart/form-data")`.
#[derive(ToSchema)]
#[allow(dead_code)]
pub struct ImageUpload {
    /// Image file (JPEG, PNG or WebP).
    #[schema(value_type = String, format = Binary, content_media_type = "application/octet-stream")]
    image: Vec<u8>,
}

/// Read the raw bytes of the `image` field from a multipart body.
pub async fn read_image_field(multipart: &mut Multipart) -> ApiResult<Bytes> {
    read_file_field(multipart, "image").await
}

/// Read the raw bytes of the field called `name` from a multipart body.
pub async fn read_file_field(multipart: &mut Multipart, name: &str) -> ApiResult<Bytes> {
    while let Some(field) = multipart.next_field().await.map_err(multipart_err)? {
        if field.name() == Some(name) {
            return field.bytes().await.map_err(multipart_err);
        }
    }
    Err(ApiError::bad_request(format!("missing `{name}` field")))
}

/// Keep the 413 for oversized uploads; any other multipart error is a 400.
fn multipart_err(e: MultipartError) -> ApiError {
    if e.status() == StatusCode::PAYLOAD_TOO_LARGE {
        ApiError::PayloadTooLarge(e.body_text())
    } else {
        ApiError::bad_request(e.body_text())
    }
}

/// Run CPU-bound work (decoding, inference, encoding) off the async workers.
pub async fn blocking<T, F>(f: F) -> ApiResult<T>
where
    F: FnOnce() -> ApiResult<T> + Send + 'static,
    T: Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(ApiError::internal)?
}

pub fn decode_image(bytes: &[u8]) -> ApiResult<DynamicImage> {
    image::load_from_memory(bytes).map_err(ApiError::bad_request)
}

pub fn png_base64(img: &RgbImage) -> ApiResult<String> {
    let mut buf = Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Png)
        .map_err(ApiError::internal)?;
    Ok(B64.encode(buf.into_inner()))
}

/// Lock a shared model, mapping a poisoned mutex to a 500.
pub fn lock<T>(m: &Mutex<T>) -> ApiResult<MutexGuard<'_, T>> {
    m.lock()
        .map_err(|_| ApiError::internal("model mutex poisoned"))
}
