// Copyright (C) 2026  Daniel Arnold
//
// This file is part of ragd.
//
// ragd is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// ragd is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with ragd.  If not, see <https://www.gnu.org/licenses/>.

//! Per-resource indexing and lifecycle management: walking a directory,
//! chunking and embedding its files, writing the results through a
//! single-writer actor so that concurrent per-file work never races on the
//! storage layer's `merge_insert` calls (LanceDB's optimistic concurrency
//! control rejects heavily concurrent writers to one table), and tracking
//! which resources are actively indexing/watching so removal can cancel
//! them promptly.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{Mutex, Semaphore, mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use crate::chunker::{ChunkingConfig, chunk_file, is_binary_extension, is_lockfile_name};
use crate::db::{ChunkFingerprint, ChunkRecord, Database, IndexingStatus, ReplaceChunksOutcome};
use crate::error::{RagdError, Result};
use crate::openai_client::OpenAiClient;
use crate::watcher::ResourceWatcher;

struct WriteJob {
    resource_name: String,
    file_path: String,
    chunks: Vec<ChunkRecord>,
    respond_to: oneshot::Sender<Result<ReplaceChunksOutcome>>,
}

/// Keys a [`ChunkWriter`]'s in-memory fingerprint cache: one file, scoped
/// to the resource that owns it (the same file path can't collide across
/// resources, since chunk identity is already `resource_name` + `file_path`
/// + `chunk_index` everywhere else in this module).
type FileKey = (String, String);

/// Serializes all [`Database::replace_file_chunks`] calls through a single
/// background task, regardless of how many files are being processed
/// concurrently. Also write-through caches each file's chunk fingerprints
/// in memory, so a file already touched this run doesn't need a DB round
/// trip just to check whether its content actually changed.
#[derive(Clone)]
pub struct ChunkWriter {
    tx: mpsc::Sender<WriteJob>,
    // Reads don't need to go through the single-writer actor below --
    // only `replace_file_chunks` needs serializing, since that's what
    // races on LanceDB's optimistic concurrency control.
    db: Arc<Database>,
    // Populated by the background task below on every successful write,
    // so it can never observe a chunk set the DB doesn't also have.
    cache: Arc<Mutex<HashMap<FileKey, HashMap<i32, ChunkFingerprint>>>>,
}

impl ChunkWriter {
    pub fn spawn(db: Arc<Database>) -> Self {
        let (tx, mut rx) = mpsc::channel::<WriteJob>(256);
        let writer_db = Arc::clone(&db);
        let cache: Arc<Mutex<HashMap<FileKey, HashMap<i32, ChunkFingerprint>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let writer_cache = Arc::clone(&cache);
        tokio::spawn(async move {
            while let Some(job) = rx.recv().await {
                let result = writer_db
                    .replace_file_chunks(&job.resource_name, &job.file_path, &job.chunks)
                    .await;
                if result.is_ok() {
                    let key = (job.resource_name.clone(), job.file_path.clone());
                    if job.chunks.is_empty() {
                        // An empty write is a prune (file deleted, renamed
                        // away, or newly ignored): drop the cache entry
                        // rather than caching "zero chunks", so a later
                        // recreation of the same path starts from a clean
                        // slate instead of an empty-but-present entry.
                        writer_cache.lock().await.remove(&key);
                    } else {
                        let fingerprints = job
                            .chunks
                            .iter()
                            .map(|chunk| {
                                (
                                    chunk.chunk_index,
                                    ChunkFingerprint {
                                        content_hash: chunk.content_hash.clone(),
                                        embedding: chunk.embedding.clone(),
                                    },
                                )
                            })
                            .collect();
                        writer_cache.lock().await.insert(key, fingerprints);
                    }
                }
                // The caller may have stopped waiting (e.g. cancelled); nothing to do if so.
                let _ = job.respond_to.send(result);
            }
        });
        Self { tx, db, cache }
    }

    /// Existing chunk fingerprints for one file, keyed by `chunk_index` --
    /// used to skip re-embedding content whose `content_hash` hasn't
    /// changed. Served from the in-memory cache when this file has already
    /// been written this run; falls back to (and populates the cache
    /// from) the database otherwise.
    pub async fn get_file_chunk_fingerprints(
        &self,
        resource_name: &str,
        file_path: &str,
    ) -> Result<HashMap<i32, ChunkFingerprint>> {
        let key = (resource_name.to_string(), file_path.to_string());
        if let Some(cached) = self.cache.lock().await.get(&key) {
            return Ok(cached.clone());
        }

        let fetched: HashMap<i32, ChunkFingerprint> = self
            .db
            .get_file_chunk_fingerprints(resource_name, file_path)
            .await?
            .into_iter()
            .collect();
        self.cache.lock().await.insert(key, fetched.clone());
        Ok(fetched)
    }

    /// Deletes every chunk belonging to `resource_name` and evicts any
    /// cached fingerprints for it. Without the eviction, a resource
    /// removed and later re-added under the same name could see stale
    /// cached embeddings survive from content that no longer exists.
    pub async fn delete_resource_chunks(&self, resource_name: &str) -> Result<()> {
        self.db.delete_resource_chunks(resource_name).await?;
        self.cache
            .lock()
            .await
            .retain(|(cached_resource, _), _| cached_resource != resource_name);
        Ok(())
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
/// has no extension, is a recognized binary format, or is a
/// dependency-manager lockfile. Shared by the initial full scan and the
/// live watcher so they can never silently diverge on what counts as
/// indexable.
pub(crate) fn indexable_extension(path: &Path) -> Option<String> {
    let extension = path.extension()?.to_str()?.to_string();
    if is_binary_extension(&extension) {
        return None;
    }
    if path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(is_lockfile_name)
    {
        return None;
    }
    Some(extension)
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

    let file_path = path.to_string_lossy().into_owned();
    // Keyed by chunk_index so an unchanged chunk can reuse its existing
    // embedding instead of paying for a re-embed of identical content --
    // this is the check a watcher false-positive (or any other spurious
    // reconcile) should hit before it ever reaches the embedding API.
    let existing = writer
        .get_file_chunk_fingerprints(resource_name, &file_path)
        .await
        .unwrap_or_default();

    let mut chunk_records = Vec::new();
    for (index, chunk) in chunk_file(extension, &content, config)
        .into_iter()
        .enumerate()
    {
        let chunk_index = i32::try_from(index).unwrap_or(i32::MAX);
        let hash = content_hash(&chunk.content);
        let previous = existing.get(&chunk_index);

        let embedding = if let Some(previous) = previous.filter(|p| p.content_hash == hash) {
            previous.embedding.clone()
        } else {
            tokio::select! {
                biased;
                () = cancellation.cancelled() => return false,
                result = client.embed(&chunk.content) => match result {
                    Ok(embedding) => embedding,
                    Err(_) => return false,
                },
            }
        };
        chunk_records.push(ChunkRecord {
            resource_name: resource_name.to_string(),
            file_path: file_path.clone(),
            chunk_index,
            content_hash: hash,
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

    // Nothing actually changed (same chunk count, same hash at every
    // index): skip the write too, not just the embedding, so a spurious
    // fs event -- or an editor touching a file without changing its
    // content -- doesn't even cost a LanceDB transaction.
    let unchanged = chunk_records.len() == existing.len()
        && chunk_records.iter().all(|record| {
            existing
                .get(&record.chunk_index)
                .is_some_and(|previous| previous.content_hash == record.content_hash)
        });
    if unchanged {
        return true;
    }

    writer
        .replace_file_chunks(resource_name, &file_path, chunk_records)
        .await
        .is_ok()
}

/// Converts a `file://` resource URI to a filesystem path. Remote (http/s)
/// resource URIs aren't supported -- an intentional scope decision, not an
/// oversight.
pub fn uri_to_path(uri: &str) -> Result<PathBuf> {
    uri.strip_prefix("file://")
        .map(PathBuf::from)
        .ok_or_else(|| {
            RagdError::Config(format!(
                "unsupported resource uri (only file:// is supported): {uri}"
            ))
        })
}

/// A cheap, non-cryptographic fingerprint used only for change detection,
/// not for identity (that's `chunk_key`, in `db.rs`).
fn content_hash(content: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    content.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

/// Tracks which resources are actively indexing/watching, so removing a
/// resource can cancel its in-flight indexing and stop its watcher rather
/// than leaving them to run against a resource that's gone.
///
/// Holds the shared dependencies needed to start a resource (client, db,
/// writer, chunking config, watch debounce) so `start` only needs the
/// per-call specifics (name, root) rather than threading all of them
/// through every call site.
#[derive(Clone)]
pub struct ResourceManager {
    active: Arc<Mutex<HashMap<String, ResourceWatcher>>>,
    client: Arc<OpenAiClient>,
    db: Arc<Database>,
    writer: ChunkWriter,
    config: ChunkingConfig,
    debounce: Duration,
}

impl ResourceManager {
    pub fn new(
        client: Arc<OpenAiClient>,
        db: Arc<Database>,
        writer: ChunkWriter,
        config: ChunkingConfig,
        debounce: Duration,
    ) -> Self {
        Self {
            active: Arc::new(Mutex::new(HashMap::new())),
            client,
            db,
            writer,
            config,
            debounce,
        }
    }

    /// Starts indexing and watching `resource_name` at `root`: an initial
    /// full scan runs in the background, followed by (and sharing a
    /// cancellation token with) an ongoing watch. If this resource was
    /// already active, its previous indexing/watching is stopped first.
    ///
    /// The `resources` row for `resource_name` is updated across the
    /// scan's lifecycle (pending -> indexing -> indexed), so
    /// `Database::get_resource_by_name` reflects real progress instead of
    /// staying "pending" forever once a scan actually finishes.
    pub async fn start(&self, resource_name: String, root: PathBuf) -> Result<()> {
        self.stop(&resource_name).await;

        if let Some(mut resource) = self.db.get_resource_by_name(&resource_name).await? {
            resource.indexing_status = IndexingStatus::Indexing;
            resource.indexing_status_message = None;
            resource.indexing_started_at = Some(chrono::Utc::now().to_rfc3339());
            self.db.upsert_resource(&resource).await?;
        }

        let cancellation = CancellationToken::new();

        let index_name = resource_name.clone();
        let index_root = root.clone();
        let index_client = Arc::clone(&self.client);
        let index_writer = self.writer.clone();
        let index_cancellation = cancellation.clone();
        let index_db = Arc::clone(&self.db);
        let config = self.config;
        tokio::spawn(async move {
            let outcome = index_resource(
                &index_name,
                &index_root,
                index_client,
                index_writer,
                config,
                index_cancellation,
                4,
            )
            .await;
            if let Ok(Some(mut resource)) = index_db.get_resource_by_name(&index_name).await {
                resource.indexing_status = IndexingStatus::Indexed;
                resource.indexing_status_message = if outcome.files_failed > 0 {
                    Some(format!(
                        "{} of {} files failed to index",
                        outcome.files_failed,
                        outcome.files_indexed + outcome.files_failed
                    ))
                } else {
                    None
                };
                resource.last_indexed_at = Some(chrono::Utc::now().to_rfc3339());
                let _ = index_db.upsert_resource(&resource).await;
            }
        });

        let watcher = ResourceWatcher::spawn(
            resource_name.clone(),
            root,
            Arc::clone(&self.client),
            self.writer.clone(),
            self.config,
            cancellation,
            self.debounce,
        )?;
        self.active.lock().await.insert(resource_name, watcher);
        Ok(())
    }

    /// Cancels indexing and stops watching `resource_name`, if it was
    /// active. A no-op if it wasn't.
    pub async fn stop(&self, resource_name: &str) {
        if let Some(watcher) = self.active.lock().await.remove(resource_name) {
            watcher.stop();
        }
    }
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

    /// Like [`mock_embed_server`], but also returns a shared counter
    /// incremented on every `/embeddings` request -- used to assert that
    /// unchanged content never reaches the embedding API at all, not just
    /// that the final chunk count looks right (which a wasted re-embed
    /// would also satisfy).
    async fn counting_mock_embed_server() -> (String, Arc<std::sync::atomic::AtomicUsize>) {
        let count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let route_count = Arc::clone(&count);
        let app = Router::new().route(
            "/embeddings",
            post(move || {
                let count = Arc::clone(&route_count);
                async move {
                    count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let embedding = vec![1.0_f32; EMBED_DIM];
                    Json(serde_json::json!({ "data": [{ "embedding": embedding }] }))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        tokio::spawn(async move {
            axum::serve(listener, app).await.expect("serve");
        });
        (format!("http://{addr}"), count)
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
    async fn index_resource_indexes_matching_files_and_skips_binaries_and_lockfiles() {
        let project = tempfile::tempdir().expect("tempdir");
        std::fs::write(project.path().join("a.rs"), "fn a() {}\n").expect("write a.rs");
        std::fs::write(project.path().join("b.md"), "# hello\n").expect("write b.md");
        std::fs::write(project.path().join("logo.png"), [0u8, 1, 2, 3]).expect("write logo.png");
        // A lockfile: valid (often huge) text with a non-binary extension,
        // but machine-generated and not worth indexing -- must be skipped
        // by exact filename, not just left to the extension check.
        std::fs::write(
            project.path().join("Cargo.lock"),
            "[[package]]\nname = \"a\"\n",
        )
        .expect("write Cargo.lock");

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
    async fn index_resource_rerun_on_unchanged_files_skips_reembedding_and_writing() {
        let project = tempfile::tempdir().expect("tempdir");
        std::fs::write(project.path().join("a.rs"), "fn a() {}\nfn b() {}\n").expect("write a.rs");

        let (embed_base, embed_calls) = counting_mock_embed_server().await;
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
        let calls_after_first_run = embed_calls.load(std::sync::atomic::Ordering::SeqCst);
        assert!(
            calls_after_first_run > 0,
            "the first run must embed something"
        );
        let version_after_first_run = db.chunks_table_version().await.expect("version");

        // Re-run against the exact same, unchanged file: a spurious
        // reconcile (or, here, a plain rerun) on content that hasn't
        // changed must neither call the embedding API again nor write a
        // new (even no-op) table version.
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

        assert_eq!(
            embed_calls.load(std::sync::atomic::Ordering::SeqCst),
            calls_after_first_run,
            "re-indexing unchanged content must not call the embedding API again"
        );
        assert_eq!(
            db.chunks_table_version().await.expect("version"),
            version_after_first_run,
            "re-indexing unchanged content must not write a new table version"
        );
    }

    #[tokio::test]
    async fn index_one_file_reembeds_only_the_chunk_that_changed() {
        let project = tempfile::tempdir().expect("tempdir");
        // Plain text (not a recognized code extension) chunks by a simple,
        // predictable line/char budget rather than tree-sitter's
        // function-boundary heuristics: chunk_lines=2, overlap=0 on 4
        // lines deterministically yields chunk 0 = lines [0,1], chunk 1 =
        // lines [2,3].
        let path = project.path().join("a.txt");
        std::fs::write(&path, "line0\nline1\nline2\nline3").expect("write a.txt");

        let (embed_base, embed_calls) = counting_mock_embed_server().await;
        let client = Arc::new(
            OpenAiClient::new(&embed_base, "", "test-embed", &embed_base, "", "test-llm")
                .expect("client"),
        );
        let db = test_db().await;
        let writer = ChunkWriter::spawn(Arc::clone(&db));
        let config = ChunkingConfig {
            chunk_lines: 2,
            overlap_lines: 0,
            max_chars: 1000,
        };
        let cancellation = CancellationToken::new();

        let extension = "txt";
        index_one_file(
            "proj",
            &path,
            extension,
            &client,
            &writer,
            &config,
            &cancellation,
        )
        .await;
        let fingerprints_before = db
            .get_file_chunk_fingerprints("proj", &path.to_string_lossy())
            .await
            .expect("fingerprints");
        assert_eq!(fingerprints_before.len(), 2, "expected two chunks");
        let calls_after_first_index = embed_calls.load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(calls_after_first_index, 2);

        // Change only the last line, which lives in chunk 1; chunk 0's
        // lines are byte-identical.
        std::fs::write(&path, "line0\nline1\nline2\nline3-changed").expect("rewrite a.txt");
        index_one_file(
            "proj",
            &path,
            extension,
            &client,
            &writer,
            &config,
            &cancellation,
        )
        .await;

        assert_eq!(
            embed_calls.load(std::sync::atomic::Ordering::SeqCst),
            calls_after_first_index + 1,
            "only the changed chunk should have been re-embedded"
        );

        let fingerprints_after = db
            .get_file_chunk_fingerprints("proj", &path.to_string_lossy())
            .await
            .expect("fingerprints");
        let before_chunk_0 = fingerprints_before
            .iter()
            .find(|(index, _)| *index == 0)
            .map(|(_, fp)| fp)
            .expect("chunk 0 present before");
        let after_chunk_0 = fingerprints_after
            .iter()
            .find(|(index, _)| *index == 0)
            .map(|(_, fp)| fp)
            .expect("chunk 0 present after");
        assert_eq!(
            before_chunk_0, after_chunk_0,
            "the unchanged first chunk's fingerprint (hash and embedding) must survive untouched"
        );
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

    fn sample_record(resource_name: &str, file_path: &str) -> ChunkRecord {
        ChunkRecord {
            resource_name: resource_name.to_string(),
            file_path: file_path.to_string(),
            chunk_index: 0,
            content: "fn a() {}".to_string(),
            content_hash: "hash".to_string(),
            start_line: 0,
            end_line: 0,
            embedding: vec![1.0; EMBED_DIM],
            updated_at: "2026-09-24T00:00:00Z".to_string(),
        }
    }

    #[tokio::test]
    async fn chunk_writer_populates_cache_on_write_and_evicts_on_prune() {
        let db = test_db().await;
        let writer = ChunkWriter::spawn(Arc::clone(&db));
        let key = ("proj".to_string(), "/tmp/proj/a.rs".to_string());

        writer
            .replace_file_chunks(
                "proj",
                "/tmp/proj/a.rs",
                vec![sample_record("proj", "/tmp/proj/a.rs")],
            )
            .await
            .expect("write");
        assert!(
            writer.cache.lock().await.contains_key(&key),
            "a successful write must populate the cache"
        );

        // An empty write is a prune (deleted / renamed away / newly
        // ignored): the cache entry must go with it, not linger as an
        // empty-but-present entry that could shadow a later recreation of
        // the same path.
        writer
            .replace_file_chunks("proj", "/tmp/proj/a.rs", Vec::new())
            .await
            .expect("prune");
        assert!(
            !writer.cache.lock().await.contains_key(&key),
            "a prune (empty write) must evict the cache entry"
        );
    }

    #[tokio::test]
    async fn chunk_writer_delete_resource_chunks_evicts_matching_cache_entries_only() {
        let db = test_db().await;
        let writer = ChunkWriter::spawn(Arc::clone(&db));

        writer
            .replace_file_chunks(
                "proj",
                "/tmp/proj/a.rs",
                vec![sample_record("proj", "/tmp/proj/a.rs")],
            )
            .await
            .expect("write proj");
        writer
            .replace_file_chunks(
                "other",
                "/tmp/proj/a.rs",
                vec![sample_record("other", "/tmp/proj/a.rs")],
            )
            .await
            .expect("write other");

        writer
            .delete_resource_chunks("proj")
            .await
            .expect("delete proj");

        let cache = writer.cache.lock().await;
        assert!(
            !cache.contains_key(&("proj".to_string(), "/tmp/proj/a.rs".to_string())),
            "deleting a resource must evict its cached fingerprints"
        );
        assert!(
            cache.contains_key(&("other".to_string(), "/tmp/proj/a.rs".to_string())),
            "deleting one resource must not evict another resource's cache entries, \
             even for the exact same file path"
        );
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

    #[tokio::test]
    async fn resource_manager_start_indexes_and_then_watches() {
        let project = tempfile::tempdir().expect("tempdir");
        std::fs::write(project.path().join("a.rs"), "fn a() {}\n").expect("write a.rs");

        let embed_base = mock_embed_server().await;
        let client = Arc::new(
            OpenAiClient::new(&embed_base, "", "test-embed", &embed_base, "", "test-llm")
                .expect("client"),
        );
        let db = test_db().await;
        let writer = ChunkWriter::spawn(Arc::clone(&db));
        let manager = ResourceManager::new(
            Arc::clone(&client),
            Arc::clone(&db),
            writer,
            ChunkingConfig::default(),
            std::time::Duration::from_millis(50),
        );

        manager
            .start("proj".to_string(), project.path().to_path_buf())
            .await
            .expect("start");

        let found_initial = poll_until(std::time::Duration::from_secs(5), || async {
            db.query_similar("proj", &[1.0; EMBED_DIM], 10)
                .await
                .map(|r| !r.is_empty())
                .unwrap_or(false)
        })
        .await;
        assert!(found_initial, "initial scan should have indexed a.rs");

        std::fs::write(project.path().join("b.rs"), "fn b() {}\n").expect("write b.rs");
        let found_live = poll_until(std::time::Duration::from_secs(5), || async {
            db.query_similar("proj", &[1.0; EMBED_DIM], 10)
                .await
                .map(|r| r.len() >= 2)
                .unwrap_or(false)
        })
        .await;
        assert!(
            found_live,
            "watcher should pick up b.rs after the initial scan"
        );
    }

    #[tokio::test]
    async fn resource_manager_start_updates_indexing_status_through_lifecycle() {
        use crate::db::{IndexingStatus, ResourceRecord, ResourceStatus};

        let project = tempfile::tempdir().expect("tempdir");
        std::fs::write(project.path().join("a.rs"), "fn a() {}\n").expect("write a.rs");

        let embed_base = mock_embed_server().await;
        let client = Arc::new(
            OpenAiClient::new(&embed_base, "", "test-embed", &embed_base, "", "test-llm")
                .expect("client"),
        );
        let db = test_db().await;
        let writer = ChunkWriter::spawn(Arc::clone(&db));
        let manager = ResourceManager::new(
            Arc::clone(&client),
            Arc::clone(&db),
            writer,
            ChunkingConfig::default(),
            std::time::Duration::from_millis(50),
        );

        db.upsert_resource(&ResourceRecord {
            uri: format!("file://{}", project.path().display()),
            name: "proj".to_string(),
            status: ResourceStatus::Active,
            indexing_status: IndexingStatus::Pending,
            indexing_status_message: None,
            created_at: "2026-09-24T00:00:00Z".to_string(),
            indexing_started_at: None,
            last_indexed_at: None,
            last_error: None,
        })
        .await
        .expect("seed resource");

        manager
            .start("proj".to_string(), project.path().to_path_buf())
            .await
            .expect("start");

        let indexed = poll_until(std::time::Duration::from_secs(5), || async {
            db.get_resource_by_name("proj")
                .await
                .map(|r| {
                    r.map(|r| r.indexing_status == IndexingStatus::Indexed)
                        .unwrap_or(false)
                })
                .unwrap_or(false)
        })
        .await;

        assert!(
            indexed,
            "indexing_status should reach Indexed once the scan completes, not stay Pending forever"
        );
        let resource = db
            .get_resource_by_name("proj")
            .await
            .expect("query")
            .expect("present");
        assert!(resource.last_indexed_at.is_some());
    }

    #[tokio::test]
    async fn resource_manager_stop_cancels_in_flight_scan_and_stops_watching() {
        let project = tempfile::tempdir().expect("tempdir");
        for i in 0..10 {
            std::fs::write(
                project.path().join(format!("f{i}.rs")),
                format!("fn f{i}() {{}}\n"),
            )
            .expect("write file");
        }

        // A slow embed endpoint gives cancellation a real window to interrupt
        // the scan before every file finishes.
        let app = axum::Router::new().route(
            "/embeddings",
            axum::routing::post(|| async {
                tokio::time::sleep(std::time::Duration::from_millis(150)).await;
                let embedding = vec![1.0_f32; EMBED_DIM];
                axum::Json(serde_json::json!({ "data": [{ "embedding": embedding }] }))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        tokio::spawn(async move {
            axum::serve(listener, app).await.expect("serve");
        });
        let embed_base = format!("http://{addr}");

        let client = Arc::new(
            OpenAiClient::new(&embed_base, "", "test-embed", &embed_base, "", "test-llm")
                .expect("client"),
        );
        let db = test_db().await;
        let writer = ChunkWriter::spawn(Arc::clone(&db));
        let manager = ResourceManager::new(
            Arc::clone(&client),
            Arc::clone(&db),
            writer,
            ChunkingConfig::default(),
            std::time::Duration::from_millis(50),
        );

        manager
            .start("proj".to_string(), project.path().to_path_buf())
            .await
            .expect("start");

        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        manager.stop("proj").await;

        // Give any (incorrectly) still-running work more than enough time to
        // have finished indexing everything, then confirm it didn't.
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        let results = db
            .query_similar("proj", &[1.0; EMBED_DIM], 20)
            .await
            .expect("query");
        assert!(
            results.len() < 10,
            "cancellation should have interrupted the scan, but all {} files were indexed",
            results.len()
        );

        // And the watcher must be stopped too: new files shouldn't appear.
        std::fs::write(project.path().join("after-stop.rs"), "fn late() {}\n")
            .expect("write after-stop.rs");
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        let results_after = db
            .query_similar("proj", &[1.0; EMBED_DIM], 20)
            .await
            .expect("query");
        assert!(
            results_after
                .iter()
                .all(|c| !c.file_path.ends_with("after-stop.rs")),
            "watcher should not still be running after stop()"
        );
    }
}
