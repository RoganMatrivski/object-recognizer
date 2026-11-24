use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};

/// Error returned by every handler. Rendered as a `text/plain` body.
#[derive(Debug)]
pub enum ApiError {
    BadRequest(String),
    NotFound(String),
    PayloadTooLarge(String),
    Internal(String),
}

pub type ApiResult<T> = Result<T, ApiError>;

impl ApiError {
    /// For use in `map_err`: the client sent something unusable.
    pub fn bad_request(msg: impl ToString) -> Self {
        Self::BadRequest(msg.to_string())
    }

    pub fn not_found(msg: impl ToString) -> Self {
        Self::NotFound(msg.to_string())
    }

    /// For use in `map_err`: the server failed to process a valid request.
    pub fn internal(msg: impl ToString) -> Self {
        Self::Internal(msg.to_string())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        match self {
            Self::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
            Self::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
            Self::PayloadTooLarge(msg) => (StatusCode::PAYLOAD_TOO_LARGE, msg),
            Self::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
        }
        .into_response()
    }
}
