//! Integration tests against `ragd`'s public API: the full HTTP
//! add→index→query→remove lifecycle, and the idempotency-across-a-restart
//! regression test that this whole rewrite exists to fix (see the project
//! plan for the empirical bug this targets in the Python original).

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use ragd::chunker::ChunkingConfig;
use ragd::db::Database;
use ragd::openai_client::OpenAiClient;
use ragd::resource::{ChunkWriter, ResourceManager, index_resource};
use ragd::server::{self, AppState};
use tokio_util::sync::CancellationToken;
use tower::ServiceExt;

const EMBED_DIM: usize = 4;

async fn mock_model_server() -> String {
    let app = axum::Router::new()
        .route(
            "/embeddings",
            axum::routing::post(|| async {
                let embedding = vec![1.0_f32; EMBED_DIM];
                axum::Json(serde_json::json!({ "data": [{ "embedding": embedding }] }))
            }),
        )
        .route(
            "/chat/completions",
            axum::routing::post(|| async {
                axum::Json(
                    serde_json::json!({ "choices": [{ "message": { "content": "the answer" } }] }),
                )
            }),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve");
    });
    format!("http://{addr}")
}

async fn poll_until<F, Fut>(timeout: std::time::Duration, mut check: F) -> bool
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    let start = tokio::time::Instant::now();
    loop {
        if check().await {
            return true;
        }
        if start.elapsed() > timeout {
            return false;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
}

async fn body_json(response: axum::response::Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    serde_json::from_slice(&bytes).expect("valid json")
}

#[tokio::test]
async fn full_lifecycle_add_index_query_remove() {
    let project = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        project.path().join("lib.rs"),
        "fn add(a: i32, b: i32) -> i32 { a + b }\n",
    )
    .expect("write lib.rs");

    let data_dir = tempfile::tempdir().expect("tempdir");
    let model_base = mock_model_server().await;

    let db = Arc::new(
        Database::connect(data_dir.path(), EMBED_DIM)
            .await
            .expect("connect"),
    );
    let client = Arc::new(
        OpenAiClient::new(&model_base, "", "test-embed", &model_base, "", "test-llm")
            .expect("client"),
    );
    let writer = ChunkWriter::spawn(Arc::clone(&db));
    let manager = ResourceManager::new();
    let app = server::router(AppState {
        db: Arc::clone(&db),
        client,
        writer,
        manager,
    });

    // 1. Add the resource -- this should start an initial scan.
    let add_request = Request::builder()
        .method("POST")
        .uri("/resources")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::json!({ "name": "proj", "uri": format!("file://{}", project.path().display()) }).to_string()))
        .expect("valid request");
    let response = app
        .clone()
        .oneshot(add_request)
        .await
        .expect("add resource");
    assert_eq!(response.status(), StatusCode::OK);

    // 2. Wait for the background scan to index the file.
    let indexed = poll_until(std::time::Duration::from_secs(5), || async {
        db.query_similar("proj", &[1.0; EMBED_DIM], 10)
            .await
            .map(|r| !r.is_empty())
            .unwrap_or(false)
    })
    .await;
    assert!(indexed, "resource should have been indexed after add");

    // 3. Query it and confirm a synthesized answer with sources comes back.
    let query_request = Request::builder()
        .method("POST")
        .uri("/query")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::json!({ "resource": "proj", "query": "what does add do?" }).to_string(),
        ))
        .expect("valid request");
    let response = app.clone().oneshot(query_request).await.expect("query");
    assert_eq!(response.status(), StatusCode::OK);
    let json = body_json(response).await;
    assert_eq!(json["answer"], "the answer");
    assert!(!json["sources"].as_array().expect("array").is_empty());

    // 4. List resources and confirm it shows up as active.
    let list_request = Request::builder()
        .uri("/resources")
        .body(Body::empty())
        .expect("valid request");
    let json = body_json(app.clone().oneshot(list_request).await.expect("list")).await;
    assert_eq!(json["resources"][0]["status"], "active");

    // 5. Remove it -- status flips to inactive and its watcher stops.
    let remove_request = Request::builder()
        .method("DELETE")
        .uri("/resources/proj")
        .body(Body::empty())
        .expect("valid request");
    let response = app.clone().oneshot(remove_request).await.expect("remove");
    assert_eq!(response.status(), StatusCode::OK);

    let list_request = Request::builder()
        .uri("/resources")
        .body(Body::empty())
        .expect("valid request");
    let json = body_json(app.oneshot(list_request).await.expect("list")).await;
    assert_eq!(json["resources"][0]["status"], "inactive");
}

#[tokio::test]
async fn restart_resync_does_not_duplicate_or_orphan_chunks() {
    let project = tempfile::tempdir().expect("tempdir");
    std::fs::write(project.path().join("a.rs"), "fn a() {}\nfn b() {}\n").expect("write a.rs");
    std::fs::write(project.path().join("c.md"), "# doc\n").expect("write c.md");

    let data_dir = tempfile::tempdir().expect("tempdir");
    let model_base = mock_model_server().await;
    let client = Arc::new(
        OpenAiClient::new(&model_base, "", "test-embed", &model_base, "", "test-llm")
            .expect("client"),
    );

    // "First run": index everything, then simulate the process exiting by
    // dropping this Database/ChunkWriter pair entirely.
    let first_run_count = {
        let db = Arc::new(
            Database::connect(data_dir.path(), EMBED_DIM)
                .await
                .expect("connect"),
        );
        let writer = ChunkWriter::spawn(Arc::clone(&db));
        index_resource(
            "proj",
            project.path(),
            Arc::clone(&client),
            writer,
            ChunkingConfig::default(),
            CancellationToken::new(),
            4,
        )
        .await;
        db.query_similar("proj", &[1.0; EMBED_DIM], 100)
            .await
            .expect("query")
            .len()
    };
    assert!(
        first_run_count > 0,
        "first run should have indexed something"
    );

    // "Restart": a fresh Database/ChunkWriter pointed at the same data_dir,
    // re-scanning the exact same, unchanged files. This is exactly what
    // happens every time avante (or any client) starts ragd and re-adds a
    // resource it already indexed -- the case that produced real duplicate
    // vectors in the Python original.
    let db2 = Arc::new(
        Database::connect(data_dir.path(), EMBED_DIM)
            .await
            .expect("connect"),
    );
    let writer2 = ChunkWriter::spawn(Arc::clone(&db2));
    index_resource(
        "proj",
        project.path(),
        Arc::clone(&client),
        writer2.clone(),
        ChunkingConfig::default(),
        CancellationToken::new(),
        4,
    )
    .await;
    let restarted_count = db2
        .query_similar("proj", &[1.0; EMBED_DIM], 100)
        .await
        .expect("query")
        .len();

    assert_eq!(
        first_run_count, restarted_count,
        "re-scanning unchanged files after a restart must not duplicate chunks"
    );

    // Now actually change a file (shifting its chunk boundaries) and rescan
    // again: the old chunks for that file must be pruned, not accumulated.
    std::fs::write(project.path().join("a.rs"), "fn a() { /* edited */ }\n").expect("edit a.rs");
    index_resource(
        "proj",
        project.path(),
        client,
        writer2,
        ChunkingConfig::default(),
        CancellationToken::new(),
        4,
    )
    .await;
    let results = db2
        .query_similar("proj", &[1.0; EMBED_DIM], 100)
        .await
        .expect("query");

    let a_rs_chunks: Vec<_> = results
        .iter()
        .filter(|chunk| chunk.file_path.ends_with("a.rs"))
        .collect();
    assert_eq!(
        a_rs_chunks.len(),
        1,
        "editing a.rs should leave exactly one current chunk for it, not accumulate old versions: {a_rs_chunks:?}"
    );
    assert_eq!(a_rs_chunks[0].content, "fn a() { /* edited */ }");
}
