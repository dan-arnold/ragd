//! HTTP API: health, resource management, and querying.

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use crate::db::{Database, IndexingStatus, ResourceRecord, ResourceStatus};
use crate::error::RagdError;

#[derive(Clone)]
struct AppState {
    db: Arc<Database>,
}

/// Builds the axum [`Router`] for the daemon's HTTP API.
///
/// Route handlers for querying are added once the indexing pipeline lands
/// (see the project plan); resource management is fully wired here, though
/// `POST /resources` does not yet kick off indexing — that arrives with the
/// indexing pipeline task.
pub fn router(db: Arc<Database>) -> Router {
    let state = AppState { db };
    Router::new()
        .route("/health", get(health))
        .route("/resources", get(list_resources).post(add_resource))
        .route("/resources/{name}", axum::routing::delete(remove_resource))
        .with_state(state)
}

/// Wraps [`RagdError`] so it can be returned directly from axum handlers.
struct ApiError(RagdError);

impl From<RagdError> for ApiError {
    fn from(err: RagdError) -> Self {
        ApiError(err)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match &self.0 {
            RagdError::ResourceNotFound(_) => StatusCode::NOT_FOUND,
            RagdError::ResourceAlreadyExists(_) => StatusCode::CONFLICT,
            RagdError::Config(_) => StatusCode::BAD_REQUEST,
            RagdError::Io(_) | RagdError::Storage(_) | RagdError::ModelEndpoint(_) => {
                StatusCode::INTERNAL_SERVER_ERROR
            }
        };
        (
            status,
            Json(serde_json::json!({ "error": self.0.to_string() })),
        )
            .into_response()
    }
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

#[derive(Debug, Deserialize)]
struct AddResourceRequest {
    name: String,
    uri: String,
}

#[derive(Debug, Serialize)]
struct StatusMessage {
    status: &'static str,
    message: String,
}

async fn add_resource(
    State(state): State<AppState>,
    Json(request): Json<AddResourceRequest>,
) -> Result<Json<StatusMessage>, ApiError> {
    if let Some(existing) = state.db.get_resource(&request.uri).await? {
        if existing.name != request.name {
            return Err(RagdError::ResourceAlreadyExists(format!(
                "uri `{}` is already registered under a different name (`{}`)",
                request.uri, existing.name
            ))
            .into());
        }
        return Ok(Json(StatusMessage {
            status: "ok",
            message: format!("resource `{}` already registered", request.name),
        }));
    }

    if state
        .db
        .get_resource_by_name(&request.name)
        .await?
        .is_some()
    {
        return Err(RagdError::ResourceAlreadyExists(format!(
            "name `{}` is already in use",
            request.name
        ))
        .into());
    }

    let resource = ResourceRecord {
        uri: request.uri,
        name: request.name.clone(),
        status: ResourceStatus::Active,
        indexing_status: IndexingStatus::Pending,
        indexing_status_message: None,
        created_at: chrono::Utc::now().to_rfc3339(),
        indexing_started_at: None,
        last_indexed_at: None,
        last_error: None,
    };
    state.db.upsert_resource(&resource).await?;

    Ok(Json(StatusMessage {
        status: "ok",
        message: format!("resource `{}` added", request.name),
    }))
}

async fn remove_resource(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<StatusMessage>, ApiError> {
    let Some(mut resource) = state.db.get_resource_by_name(&name).await? else {
        return Err(RagdError::ResourceNotFound(name).into());
    };

    resource.status = ResourceStatus::Inactive;
    state.db.upsert_resource(&resource).await?;

    Ok(Json(StatusMessage {
        status: "ok",
        message: format!("resource `{name}` removed"),
    }))
}

#[derive(Debug, Serialize)]
struct ResourceListResponse {
    resources: Vec<ResourceRecord>,
    total_count: usize,
    status_summary: HashMap<String, usize>,
}

async fn list_resources(
    State(state): State<AppState>,
) -> Result<Json<ResourceListResponse>, ApiError> {
    let resources = state.db.list_resources().await?;

    let mut status_summary = HashMap::new();
    for resource in &resources {
        *status_summary
            .entry(resource.status.as_str().to_string())
            .or_insert(0) += 1;
    }

    Ok(Json(ResourceListResponse {
        total_count: resources.len(),
        resources,
        status_summary,
    }))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    async fn test_router() -> Router {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = Database::connect(dir.keep().as_path(), 4)
            .await
            .expect("connect");
        router(Arc::new(db))
    }

    async fn body_json(response: Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        serde_json::from_slice(&bytes).expect("valid json")
    }

    fn post_resources(name: &str, uri: &str) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri("/resources")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::json!({ "name": name, "uri": uri }).to_string(),
            ))
            .expect("valid request")
    }

    #[tokio::test]
    async fn health_returns_ok() {
        let app = test_router().await;
        let request = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .expect("valid request");
        let response = app.oneshot(request).await.expect("request should succeed");

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn add_resource_then_list_returns_it() {
        let app = test_router().await;
        let response = app
            .clone()
            .oneshot(post_resources("proj", "file:///tmp/proj/"))
            .await
            .expect("post");
        assert_eq!(response.status(), StatusCode::OK);

        let list_request = Request::builder()
            .uri("/resources")
            .body(Body::empty())
            .expect("valid request");
        let response = app.oneshot(list_request).await.expect("get");
        assert_eq!(response.status(), StatusCode::OK);

        let json = body_json(response).await;
        assert_eq!(json["total_count"], 1);
        assert_eq!(json["resources"][0]["name"], "proj");
        assert_eq!(json["resources"][0]["status"], "active");
    }

    #[tokio::test]
    async fn add_resource_twice_same_uri_is_idempotent() {
        let app = test_router().await;
        app.clone()
            .oneshot(post_resources("proj", "file:///tmp/proj/"))
            .await
            .expect("post 1");
        let response = app
            .clone()
            .oneshot(post_resources("proj", "file:///tmp/proj/"))
            .await
            .expect("post 2");
        assert_eq!(response.status(), StatusCode::OK);

        let list_request = Request::builder()
            .uri("/resources")
            .body(Body::empty())
            .expect("valid request");
        let json = body_json(app.oneshot(list_request).await.expect("get")).await;
        assert_eq!(json["total_count"], 1);
    }

    #[tokio::test]
    async fn add_resource_duplicate_name_different_uri_conflicts() {
        let app = test_router().await;
        app.clone()
            .oneshot(post_resources("proj", "file:///tmp/proj-a/"))
            .await
            .expect("post 1");
        let response = app
            .oneshot(post_resources("proj", "file:///tmp/proj-b/"))
            .await
            .expect("post 2");

        assert_eq!(response.status(), StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn remove_resource_marks_inactive_not_deleted() {
        let app = test_router().await;
        app.clone()
            .oneshot(post_resources("proj", "file:///tmp/proj/"))
            .await
            .expect("post");

        let delete_request = Request::builder()
            .method("DELETE")
            .uri("/resources/proj")
            .body(Body::empty())
            .expect("valid request");
        let response = app.clone().oneshot(delete_request).await.expect("delete");
        assert_eq!(response.status(), StatusCode::OK);

        let list_request = Request::builder()
            .uri("/resources")
            .body(Body::empty())
            .expect("valid request");
        let json = body_json(app.oneshot(list_request).await.expect("get")).await;
        assert_eq!(json["total_count"], 1);
        assert_eq!(json["resources"][0]["status"], "inactive");
    }

    #[tokio::test]
    async fn remove_missing_resource_returns_404() {
        let app = test_router().await;
        let delete_request = Request::builder()
            .method("DELETE")
            .uri("/resources/does-not-exist")
            .body(Body::empty())
            .expect("valid request");
        let response = app.oneshot(delete_request).await.expect("delete");

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
}
