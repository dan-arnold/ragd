//! HTTP API: health, resource management, and querying.

use axum::{Json, Router, routing::get};
use serde::Serialize;

/// Builds the axum [`Router`] for the daemon's HTTP API.
///
/// Route handlers for resource management and querying are added as those
/// components land; for now this only exposes `/health`.
pub fn router() -> Router {
    Router::new().route("/health", get(health))
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    #[tokio::test]
    async fn health_returns_ok() {
        let app = router();
        let request = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .expect("valid request");
        let response = app.oneshot(request).await.expect("request should succeed");

        assert_eq!(response.status(), StatusCode::OK);
    }
}
