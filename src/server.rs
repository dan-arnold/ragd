//! HTTP API: health, resource management, and querying.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use crate::chunker::ChunkingConfig;
use crate::db::{Database, IndexingStatus, ResourceRecord, ResourceStatus};
use crate::error::RagdError;
use crate::openai_client::OpenAiClient;
use crate::resource::{ChunkWriter, ResourceManager, uri_to_path};

const DEFAULT_TOP_K: usize = 5;
const MAX_TOP_K: usize = 20;
const WATCH_DEBOUNCE: Duration = Duration::from_secs(2);

/// Shared dependencies for the HTTP API. Constructed once at startup (see
/// `main.rs`) and cloned cheaply per-request via axum's `State` extractor.
#[derive(Clone)]
pub struct AppState {
    pub db: Arc<Database>,
    pub client: Arc<OpenAiClient>,
    pub writer: ChunkWriter,
    pub manager: ResourceManager,
}

/// Builds the axum [`Router`] for the daemon's HTTP API.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/resources", get(list_resources).post(add_resource))
        .route("/resources/{name}", axum::routing::delete(remove_resource))
        .route("/query", axum::routing::post(query))
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

    let root = uri_to_path(&request.uri)?;

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

    state
        .manager
        .start(
            request.name.clone(),
            root,
            Arc::clone(&state.client),
            state.writer.clone(),
            ChunkingConfig::default(),
            WATCH_DEBOUNCE,
        )
        .await?;

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
    state.manager.stop(&name).await;

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

#[derive(Debug, Deserialize)]
struct QueryRequest {
    resource: String,
    query: String,
    top_k: Option<usize>,
}

#[derive(Debug, Serialize)]
struct SourceChunk {
    path: String,
    content: String,
    /// Cosine similarity in `0.0..=1.0`; higher is more relevant.
    score: f32,
}

#[derive(Debug, Serialize)]
struct QueryResponse {
    answer: String,
    sources: Vec<SourceChunk>,
}

async fn query(
    State(state): State<AppState>,
    Json(request): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, ApiError> {
    let resource = state
        .db
        .get_resource_by_name(&request.resource)
        .await?
        .ok_or_else(|| RagdError::ResourceNotFound(request.resource.clone()))?;

    let top_k = request.top_k.unwrap_or(DEFAULT_TOP_K).clamp(1, MAX_TOP_K);
    let query_embedding = state.client.embed(&request.query).await?;
    let scored_chunks = state
        .db
        .query_similar_with_scores(&resource.name, &query_embedding, top_k)
        .await?;

    if scored_chunks.is_empty() {
        return Ok(Json(QueryResponse {
            answer: "No indexed content matched this query.".to_string(),
            sources: Vec::new(),
        }));
    }

    let context = scored_chunks
        .iter()
        .map(|(chunk, _)| format!("Source: {}\n{}", chunk.file_path, chunk.content))
        .collect::<Vec<_>>()
        .join("\n\n---\n\n");
    let system_prompt = format!(
        "Answer the user's question using only the following retrieved context. If the context doesn't contain the answer, say so plainly rather than guessing.\n\n{context}"
    );
    let answer = state.client.chat(&system_prompt, &request.query).await?;

    let sources = scored_chunks
        .into_iter()
        .map(|(chunk, score)| SourceChunk {
            path: chunk.file_path,
            content: chunk.content,
            score,
        })
        .collect();

    Ok(Json(QueryResponse { answer, sources }))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::db::ChunkRecord;
    use axum::body::Body;
    use axum::http::Request;
    use axum::routing::post;
    use tower::ServiceExt;

    fn test_app_state(db: Arc<Database>, client: Arc<OpenAiClient>) -> AppState {
        AppState {
            writer: ChunkWriter::spawn(Arc::clone(&db)),
            db,
            client,
            manager: ResourceManager::new(),
        }
    }

    async fn test_router_with_db() -> (Router, Arc<Database>) {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = Arc::new(
            Database::connect(dir.keep().as_path(), 4)
                .await
                .expect("connect"),
        );
        // Resource-CRUD tests never call the model endpoints; point at an
        // address nothing listens on, since it's unreachable-but-unused.
        let client = Arc::new(
            OpenAiClient::new(
                "http://127.0.0.1:1",
                "",
                "test-embed",
                "http://127.0.0.1:1",
                "",
                "test-llm",
            )
            .expect("client"),
        );
        (router(test_app_state(Arc::clone(&db), client)), db)
    }

    async fn test_router() -> Router {
        test_router_with_db().await.0
    }

    async fn mock_model_server(embedding: Vec<f32>, chat_answer: &str) -> String {
        let chat_answer = chat_answer.to_string();
        let app = Router::new()
            .route("/embeddings", post(move || { let embedding = embedding.clone(); async move { Json(serde_json::json!({ "data": [{ "embedding": embedding }] })) } }))
            .route("/chat/completions", post(move || { let chat_answer = chat_answer.clone(); async move { Json(serde_json::json!({ "choices": [{ "message": { "content": chat_answer } }] })) } }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        tokio::spawn(async move {
            axum::serve(listener, app).await.expect("serve");
        });
        format!("http://{addr}")
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

    /// `add_resource` starts a real watcher, which needs a real directory to
    /// watch -- returns the `file://` URI alongside the `TempDir` guard,
    /// which the caller must keep alive for as long as the resource stays
    /// registered in the test.
    fn temp_resource_uri() -> (tempfile::TempDir, String) {
        let dir = tempfile::tempdir().expect("tempdir");
        let uri = format!("file://{}", dir.path().display());
        (dir, uri)
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
        let (_dir, uri) = temp_resource_uri();
        let response = app
            .clone()
            .oneshot(post_resources("proj", &uri))
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
        let (_dir, uri) = temp_resource_uri();
        app.clone()
            .oneshot(post_resources("proj", &uri))
            .await
            .expect("post 1");
        let response = app
            .clone()
            .oneshot(post_resources("proj", &uri))
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
        let (_dir_a, uri_a) = temp_resource_uri();
        let (_dir_b, uri_b) = temp_resource_uri();
        app.clone()
            .oneshot(post_resources("proj", &uri_a))
            .await
            .expect("post 1");
        let response = app
            .oneshot(post_resources("proj", &uri_b))
            .await
            .expect("post 2");

        assert_eq!(response.status(), StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn remove_resource_marks_inactive_not_deleted() {
        let app = test_router().await;
        let (_dir, uri) = temp_resource_uri();
        app.clone()
            .oneshot(post_resources("proj", &uri))
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

    fn post_query(resource: &str, query: &str) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri("/query")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::json!({ "resource": resource, "query": query }).to_string(),
            ))
            .expect("valid request")
    }

    #[tokio::test]
    async fn query_returns_synthesized_answer_and_scored_sources() {
        let model_base = mock_model_server(vec![1.0, 0.0, 0.0, 0.0], "synthesized answer").await;
        let dir = tempfile::tempdir().expect("tempdir");
        let db = Arc::new(
            Database::connect(dir.keep().as_path(), 4)
                .await
                .expect("connect"),
        );
        let client = Arc::new(
            OpenAiClient::new(&model_base, "", "test-embed", &model_base, "", "test-llm")
                .expect("client"),
        );
        let app = router(test_app_state(Arc::clone(&db), client));

        let (_dir, uri) = temp_resource_uri();
        app.clone()
            .oneshot(post_resources("proj", &uri))
            .await
            .expect("add resource");
        let file_path = format!("{}/a.rs", uri.strip_prefix("file://").expect("file uri"));
        db.replace_file_chunks(
            "proj",
            &file_path,
            &[ChunkRecord {
                resource_name: "proj".to_string(),
                file_path: file_path.clone(),
                chunk_index: 0,
                content: "fn a() {}".to_string(),
                content_hash: "hash".to_string(),
                start_line: 0,
                end_line: 0,
                embedding: vec![1.0, 0.0, 0.0, 0.0],
                updated_at: "2026-09-24T00:00:00Z".to_string(),
            }],
        )
        .await
        .expect("seed chunk");

        let response = app
            .oneshot(post_query("proj", "what does a do?"))
            .await
            .expect("query");
        assert_eq!(response.status(), StatusCode::OK);

        let json = body_json(response).await;
        assert_eq!(json["answer"], "synthesized answer");
        assert_eq!(json["sources"].as_array().expect("array").len(), 1);
        assert_eq!(json["sources"][0]["path"], file_path);
        let score = json["sources"][0]["score"].as_f64().expect("score");
        assert!(
            (score - 1.0).abs() < 1e-3,
            "expected score ~1.0, got {score}"
        );
    }

    #[tokio::test]
    async fn query_returns_404_for_unknown_resource() {
        let model_base = mock_model_server(vec![1.0, 0.0, 0.0, 0.0], "unused").await;
        let (app, _db) = {
            let dir = tempfile::tempdir().expect("tempdir");
            let db = Arc::new(
                Database::connect(dir.keep().as_path(), 4)
                    .await
                    .expect("connect"),
            );
            let client = Arc::new(
                OpenAiClient::new(&model_base, "", "test-embed", &model_base, "", "test-llm")
                    .expect("client"),
            );
            (router(test_app_state(Arc::clone(&db), client)), db)
        };

        let response = app
            .oneshot(post_query("does-not-exist", "anything"))
            .await
            .expect("query");

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn query_returns_empty_sources_when_resource_has_no_chunks() {
        let model_base = mock_model_server(vec![1.0, 0.0, 0.0, 0.0], "unused").await;
        let dir = tempfile::tempdir().expect("tempdir");
        let db = Arc::new(
            Database::connect(dir.keep().as_path(), 4)
                .await
                .expect("connect"),
        );
        let client = Arc::new(
            OpenAiClient::new(&model_base, "", "test-embed", &model_base, "", "test-llm")
                .expect("client"),
        );
        let app = router(test_app_state(Arc::clone(&db), client));

        let (_dir, uri) = temp_resource_uri();
        app.clone()
            .oneshot(post_resources("proj", &uri))
            .await
            .expect("add resource");

        let response = app
            .oneshot(post_query("proj", "anything"))
            .await
            .expect("query");
        assert_eq!(response.status(), StatusCode::OK);

        let json = body_json(response).await;
        assert_eq!(json["sources"].as_array().expect("array").len(), 0);
        assert_eq!(json["answer"], "No indexed content matched this query.");
    }
}
