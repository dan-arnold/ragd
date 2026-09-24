//! Per-resource indexing: walking a directory, chunking and embedding its
//! files, and writing the results through a single-writer actor so that
//! concurrent per-file work never races on the storage layer's
//! `merge_insert` calls (LanceDB's optimistic concurrency control rejects
//! heavily concurrent writers to one table — see the project plan).

use std::path::Path;
use std::sync::Arc;

use tokio::sync::{Semaphore, mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use crate::chunker::{ChunkingConfig, chunk_file, is_binary_extension};
use crate::db::{ChunkRecord, Database, ReplaceChunksOutcome};
use crate::error::{RagdError, Result};
use crate::openai_client::OpenAiClient;

struct WriteJob {
    resource_name: String,
    file_path: String,
    chunks: Vec<ChunkRecord>,
    respond_to: oneshot::Sender<Result<ReplaceChunksOutcome>>,
}

/// Serializes all [`Database::replace_file_chunks`] calls through a single
/// background task, regardless of how many files are being processed
/// concurrently.
#[derive(Clone)]
pub struct ChunkWriter {
    tx: mpsc::Sender<WriteJob>,
}

impl ChunkWriter {
    pub fn spawn(db: Arc<Database>) -> Self {
        let (tx, mut rx) = mpsc::channel::<WriteJob>(256);
        tokio::spawn(async move {
            while let Some(job) = rx.recv().await {
                let result = db
                    .replace_file_chunks(&job.resource_name, &job.file_path, &job.chunks)
                    .await;
                // The caller may have stopped waiting (e.g. cancelled); nothing to do if so.
                let _ = job.respond_to.send(result);
            }
        });
        Self { tx }
    }

    pub async fn replace_file_chunks(
        &self,
        resource_name: &str,
        file_path: &str,
        chunks: Vec<ChunkRecord>,
    ) -> Result<ReplaceChunksOutcome> {
        let (respond_to, response) = oneshot::channel();
        let job = WriteJob {
            resource_name: resource_name.to_string(),
            file_path: file_path.to_string(),
            chunks,
            respond_to,
        };
        self.tx.send(job).await.map_err(|_| {
            RagdError::Storage("chunk writer task is no longer running".to_string())
        })?;
        response
            .await
            .map_err(|_| RagdError::Storage("chunk writer task dropped the response".to_string()))?
    }
}

/// Summary of one full-directory indexing pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IndexOutcome {
    pub files_indexed: usize,
    pub files_failed: usize,
}

/// Walks `root` (gitignore-aware, including nested `.gitignore` files),
/// chunks and embeds every non-binary file, and writes each file's chunks
/// through `writer`. Files are processed concurrently, bounded by
/// `max_concurrency`; `cancellation` is checked between files and around
/// each embedding call so a resource removal can stop an in-flight scan
/// promptly rather than waiting for it to finish naturally.
pub async fn index_resource(
    resource_name: &str,
    root: &Path,
    client: Arc<OpenAiClient>,
    writer: ChunkWriter,
    config: ChunkingConfig,
    cancellation: CancellationToken,
    max_concurrency: usize,
) -> IndexOutcome {
    let semaphore = Arc::new(Semaphore::new(max_concurrency.max(1)));
    let mut tasks = Vec::new();

    // `require_git(false)`: honor `.gitignore` files even when `root` isn't
    // inside an actual git repository, which is the `ignore` crate's
    // default-off behavior otherwise -- surprising, since a plain folder of
    // docs with a `.gitignore` should still be respected.
    for entry in ignore::WalkBuilder::new(root).require_git(false).build() {
        if cancellation.is_cancelled() {
            break;
        }
        let Ok(entry) = entry else { continue };
        if !entry
            .file_type()
            .is_some_and(|file_type| file_type.is_file())
        {
            continue;
        }
        let path = entry.into_path();
        let Some(extension) = indexable_extension(&path) else {
            continue;
        };

        let semaphore = Arc::clone(&semaphore);
        let client = Arc::clone(&client);
        let writer = writer.clone();
        let cancellation = cancellation.clone();
        let resource_name = resource_name.to_string();

        tasks.push(tokio::spawn(async move {
            let Ok(_permit) = semaphore.acquire_owned().await else {
                return false;
            };
            index_one_file(
                &resource_name,
                &path,
                &extension,
                &client,
                &writer,
                &config,
                &cancellation,
            )
            .await
        }));
    }

    let mut outcome = IndexOutcome::default();
    for task in tasks {
        match task.await {
            Ok(true) => outcome.files_indexed += 1,
            Ok(false) | Err(_) => outcome.files_failed += 1,
        }
    }
    outcome
}

/// Extension for `path` if it's a file this daemon indexes, `None` if it
/// has no extension or is a recognized binary format. Shared by the
/// initial full scan and the live watcher so they can never silently
/// diverge on what counts as indexable (unlike the Python original, whose
/// bulk indexer and file watcher drifted apart this way).
pub(crate) fn indexable_extension(path: &Path) -> Option<String> {
    let extension = path.extension()?.to_str()?.to_string();
    if is_binary_extension(&extension) {
        None
    } else {
        Some(extension)
    }
}

pub(crate) async fn index_one_file(
    resource_name: &str,
    path: &Path,
    extension: &str,
    client: &OpenAiClient,
    writer: &ChunkWriter,
    config: &ChunkingConfig,
    cancellation: &CancellationToken,
) -> bool {
    if cancellation.is_cancelled() {
        return false;
    }

    let Ok(content) = tokio::fs::read_to_string(path).await else {
        return false;
    };

    let mut chunk_records = Vec::new();
    for (index, chunk) in chunk_file(extension, &content, config)
        .into_iter()
        .enumerate()
    {
        let embedding = tokio::select! {
            biased;
            () = cancellation.cancelled() => return false,
            result = client.embed(&chunk.content) => match result {
                Ok(embedding) => embedding,
                Err(_) => return false,
            },
        };
        chunk_records.push(ChunkRecord {
            resource_name: resource_name.to_string(),
            file_path: path.to_string_lossy().into_owned(),
            chunk_index: i32::try_from(index).unwrap_or(i32::MAX),
            content_hash: content_hash(&chunk.content),
            content: chunk.content,
            start_line: chunk.start_line as i32,
            end_line: chunk.end_line as i32,
            embedding,
            updated_at: chrono::Utc::now().to_rfc3339(),
        });
    }

    if cancellation.is_cancelled() {
        return false;
    }

    writer
        .replace_file_chunks(resource_name, &path.to_string_lossy(), chunk_records)
        .await
        .is_ok()
}

/// A cheap, non-cryptographic fingerprint used only for change detection,
/// not for identity (that's `chunk_key`, in `db.rs`).
fn content_hash(content: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    content.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use axum::routing::post;
    use axum::{Json, Router};

    const EMBED_DIM: usize = 4;

    async fn mock_embed_server() -> String {
        let app = Router::new().route(
            "/embeddings",
            post(|| async {
                let embedding = vec![1.0_f32; EMBED_DIM];
                Json(serde_json::json!({ "data": [{ "embedding": embedding }] }))
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

    async fn test_db() -> Arc<Database> {
        let dir = tempfile::tempdir().expect("tempdir");
        Arc::new(
            Database::connect(dir.keep().as_path(), EMBED_DIM)
                .await
                .expect("connect"),
        )
    }

    #[tokio::test]
    async fn index_resource_indexes_matching_files_and_skips_binaries() {
        let project = tempfile::tempdir().expect("tempdir");
        std::fs::write(project.path().join("a.rs"), "fn a() {}\n").expect("write a.rs");
        std::fs::write(project.path().join("b.md"), "# hello\n").expect("write b.md");
        std::fs::write(project.path().join("logo.png"), [0u8, 1, 2, 3]).expect("write logo.png");

        let embed_base = mock_embed_server().await;
        let client = Arc::new(
            OpenAiClient::new(&embed_base, "", "test-embed", &embed_base, "", "test-llm")
                .expect("client"),
        );
        let db = test_db().await;
        let writer = ChunkWriter::spawn(Arc::clone(&db));

        let outcome = index_resource(
            "proj",
            project.path(),
            client,
            writer,
            ChunkingConfig::default(),
            CancellationToken::new(),
            4,
        )
        .await;

        assert_eq!(outcome.files_indexed, 2);
        assert_eq!(outcome.files_failed, 0);

        let results = db
            .query_similar("proj", &[1.0; EMBED_DIM], 10)
            .await
            .expect("query");
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn index_resource_respects_nested_gitignore() {
        let project = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(project.path().join("vendor")).expect("mkdir vendor");
        std::fs::write(project.path().join("vendor/.gitignore"), "ignored.rs\n")
            .expect("write nested gitignore");
        std::fs::write(
            project.path().join("vendor/ignored.rs"),
            "fn ignored() {}\n",
        )
        .expect("write ignored.rs");
        std::fs::write(project.path().join("kept.rs"), "fn kept() {}\n").expect("write kept.rs");

        let embed_base = mock_embed_server().await;
        let client = Arc::new(
            OpenAiClient::new(&embed_base, "", "test-embed", &embed_base, "", "test-llm")
                .expect("client"),
        );
        let db = test_db().await;
        let writer = ChunkWriter::spawn(Arc::clone(&db));

        let outcome = index_resource(
            "proj",
            project.path(),
            client,
            writer,
            ChunkingConfig::default(),
            CancellationToken::new(),
            4,
        )
        .await;

        assert_eq!(
            outcome.files_indexed, 1,
            "only kept.rs should have been indexed"
        );

        let results = db
            .query_similar("proj", &[1.0; EMBED_DIM], 10)
            .await
            .expect("query");
        assert_eq!(results.len(), 1);
        assert!(results[0].file_path.ends_with("kept.rs"));
    }

    #[tokio::test]
    async fn index_resource_rerun_on_unchanged_files_does_not_grow_chunk_count() {
        let project = tempfile::tempdir().expect("tempdir");
        std::fs::write(project.path().join("a.rs"), "fn a() {}\nfn b() {}\n").expect("write a.rs");

        let embed_base = mock_embed_server().await;
        let client = Arc::new(
            OpenAiClient::new(&embed_base, "", "test-embed", &embed_base, "", "test-llm")
                .expect("client"),
        );
        let db = test_db().await;
        let writer = ChunkWriter::spawn(Arc::clone(&db));

        index_resource(
            "proj",
            project.path(),
            Arc::clone(&client),
            writer.clone(),
            ChunkingConfig::default(),
            CancellationToken::new(),
            4,
        )
        .await;
        let first_count = db
            .query_similar("proj", &[1.0; EMBED_DIM], 100)
            .await
            .expect("query")
            .len();

        // Re-run against the exact same, unchanged files -- must be a no-op.
        index_resource(
            "proj",
            project.path(),
            client,
            writer,
            ChunkingConfig::default(),
            CancellationToken::new(),
            4,
        )
        .await;
        let second_count = db
            .query_similar("proj", &[1.0; EMBED_DIM], 100)
            .await
            .expect("query")
            .len();

        assert_eq!(first_count, second_count);
    }

    #[tokio::test]
    async fn index_resource_cancelled_before_start_indexes_nothing() {
        let project = tempfile::tempdir().expect("tempdir");
        std::fs::write(project.path().join("a.rs"), "fn a() {}\n").expect("write a.rs");

        let embed_base = mock_embed_server().await;
        let client = Arc::new(
            OpenAiClient::new(&embed_base, "", "test-embed", &embed_base, "", "test-llm")
                .expect("client"),
        );
        let db = test_db().await;
        let writer = ChunkWriter::spawn(Arc::clone(&db));

        let cancellation = CancellationToken::new();
        cancellation.cancel();

        index_resource(
            "proj",
            project.path(),
            client,
            writer,
            ChunkingConfig::default(),
            cancellation,
            4,
        )
        .await;

        let results = db
            .query_similar("proj", &[1.0; EMBED_DIM], 10)
            .await
            .expect("query");
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn chunk_writer_serializes_concurrent_writes_to_different_files() {
        let db = test_db().await;
        let writer = ChunkWriter::spawn(Arc::clone(&db));

        let mut handles = Vec::new();
        for i in 0..8 {
            let writer = writer.clone();
            handles.push(tokio::spawn(async move {
                let chunk = ChunkRecord {
                    resource_name: "proj".to_string(),
                    file_path: format!("/tmp/proj/file{i}.rs"),
                    chunk_index: 0,
                    content: format!("fn f{i}() {{}}"),
                    content_hash: "hash".to_string(),
                    start_line: 0,
                    end_line: 0,
                    embedding: vec![1.0; EMBED_DIM],
                    updated_at: "2026-09-24T00:00:00Z".to_string(),
                };
                writer
                    .replace_file_chunks("proj", &format!("/tmp/proj/file{i}.rs"), vec![chunk])
                    .await
            }));
        }

        for handle in handles {
            handle.await.expect("task").expect("write succeeded");
        }

        let results = db
            .query_similar("proj", &[1.0; EMBED_DIM], 100)
            .await
            .expect("query");
        assert_eq!(results.len(), 8);
    }
}
