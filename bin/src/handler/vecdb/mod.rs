//! Manual vector DB access: clients supply vectors and payloads themselves.
//! Nothing here runs a model.

mod maintenance;
mod points;
mod search;
mod transfer;

use object_recognizer::vecdb::{self, Error as DbError, Filter};
use serde::Deserialize;
use std::{collections::HashMap, io::ErrorKind};
use utoipa::ToSchema;
use utoipa_axum::router::OpenApiRouter;

use super::{HandlerState, error::ApiError};

pub const TAG: &str = "vecdb";

pub(super) fn router() -> OpenApiRouter<HandlerState> {
    OpenApiRouter::new()
        .merge(points::router())
        .merge(search::router())
        .merge(transfer::router())
        .merge(maintenance::router())
}

/// A value a payload field must equal.
#[derive(Deserialize, ToSchema)]
#[serde(untagged)]
enum FilterValue {
    Bool(bool),
    Int(i64),
    Str(String),
}

/// Payload equality filter: every listed field must equal its value (AND).
/// Example: `{ "name": "apple", "category": "fruit" }`.
type FieldFilter = HashMap<String, FilterValue>;

/// Convert a [`FieldFilter`] to a DB filter. `None` for an empty map.
fn to_filter(fields: FieldFilter) -> Result<Option<Filter>, ApiError> {
    if fields.is_empty() {
        return Ok(None);
    }
    let conds = fields
        .into_iter()
        .map(|(key, value)| match value {
            FilterValue::Bool(b) => vecdb::eq(&key, b),
            FilterValue::Int(i) => vecdb::eq(&key, i),
            FilterValue::Str(s) => vecdb::eq(&key, s),
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(db_err)?;
    Ok(Some(vecdb::all_of(conds)))
}

fn empty_object() -> serde_json::Value {
    serde_json::json!({})
}

/// Map DB errors caused by bad input to 400, everything else to 500.
fn db_err(e: DbError) -> ApiError {
    match &e {
        DbError::DimMismatch { .. }
        | DbError::InvalidKey(_)
        | DbError::Incompatible(_)
        | DbError::EmptyFile
        | DbError::Json(_) => ApiError::bad_request(e),
        // Corrupt/non-gzip import files surface as I/O errors of these kinds.
        DbError::Io(io)
            if matches!(
                io.kind(),
                ErrorKind::InvalidData | ErrorKind::InvalidInput | ErrorKind::UnexpectedEof
            ) =>
        {
            ApiError::bad_request(e)
        }
        _ => ApiError::internal(e),
    }
}
