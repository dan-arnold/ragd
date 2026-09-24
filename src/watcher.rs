//! Live filesystem watching for a resource: debounced change notifications
//! are reconciled against current disk state (does the path still exist,
//! is it still allowed by `.gitignore`) rather than trusting the
//! underlying watcher's event kind, which `notify-debouncer-mini`
//! deliberately doesn't distinguish beyond "something changed here" vs.
//! "continuous writes here" — there's no separate delete/rename event to
//! hook, which is exactly the gap that let the Python original's watcher
//! silently skip deletions. Reconciling against ground truth on every
//! event sidesteps needing that distinction at all: a path that no longer
//! exists gets its chunks pruned, one that exists and is still allowed
//! gets re-indexed.
//!
//! Ignore-checking is nested-`.gitignore`-aware, rebuilt from every
//! `.gitignore` under the resource root on each reconciliation. This is a
//! deliberate correctness-over-micro-optimization choice: contextd (the
//! Rust project surveyed while planning this daemon) has the same bug the
//! Python original did here, checking new files against only the
//! resource-root's own `.gitignore` at watch time even though its initial
//! scan handles nested ones correctly.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use ignore::WalkBuilder;
use ignore::gitignore::{Gitignore, GitignoreBuilder};
use notify_debouncer_mini::notify::RecursiveMode;
use notify_debouncer_mini::{DebounceEventResult, Debouncer, new_debouncer};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::chunker::ChunkingConfig;
use crate::error::{RagdError, Result};
use crate::openai_client::OpenAiClient;
use crate::resource::{ChunkWriter, index_one_file, indexable_extension};

/// A live watch on one resource's root directory. Dropping this (or
/// calling [`ResourceWatcher::stop`]) stops both the underlying OS watch
/// and the reconciliation task.
pub struct ResourceWatcher {
    _debouncer: Debouncer<notify_debouncer_mini::notify::RecommendedWatcher>,
    cancellation: CancellationToken,
}

impl ResourceWatcher {
    /// Starts watching `root` for changes, reconciling every `debounce`
    /// interval. Reconciliation reads and embeds changed files and prunes
    /// chunks for files that no longer exist or are no longer allowed.
    pub fn spawn(
        resource_name: String,
        root: PathBuf,
        client: Arc<OpenAiClient>,
        writer: ChunkWriter,
        config: ChunkingConfig,
        debounce: Duration,
    ) -> Result<Self> {
        let (tx, mut rx) = mpsc::unbounded_channel::<DebounceEventResult>();

        let mut debouncer = new_debouncer(debounce, move |result: DebounceEventResult| {
            // The debouncer runs this on its own thread; UnboundedSender::send
            // is synchronous (never blocks), so this is safe to call directly.
            let _ = tx.send(result);
        })
        .map_err(|err| RagdError::Storage(format!("failed to start watcher: {err}")))?;
        debouncer
            .watcher()
            .watch(&root, RecursiveMode::Recursive)
            .map_err(|err| {
                RagdError::Storage(format!("failed to watch {}: {err}", root.display()))
            })?;

        let cancellation = CancellationToken::new();
        let task_cancellation = cancellation.clone();

        tokio::spawn(async move {
            loop {
                tokio::select! {
                    biased;
                    () = task_cancellation.cancelled() => break,
                    message = rx.recv() => {
                        let Some(Ok(events)) = message else { continue };
                        let paths: HashSet<PathBuf> = events.into_iter().map(|event| event.path).collect();
                        reconcile(&resource_name, &root, paths, &client, &writer, &config, &task_cancellation).await;
                    }
                }
            }
        });

        Ok(Self {
            _debouncer: debouncer,
            cancellation,
        })
    }

    /// Stops watching and reconciling. Any in-flight reconciliation for a
    /// single file finishes; no new work is picked up after this returns.
    pub fn stop(self) {
        self.cancellation.cancel();
    }
}

async fn reconcile(
    resource_name: &str,
    root: &Path,
    paths: HashSet<PathBuf>,
    client: &Arc<OpenAiClient>,
    writer: &ChunkWriter,
    config: &ChunkingConfig,
    cancellation: &CancellationToken,
) {
    if paths.is_empty() || cancellation.is_cancelled() {
        return;
    }

    let ignore_matcher = build_ignore_matcher(root);

    for path in paths {
        if cancellation.is_cancelled() {
            return;
        }

        let is_dir = path.is_dir();
        let allowed = path.starts_with(root)
            && !ignore_matcher
                .matched_path_or_any_parents(&path, is_dir)
                .is_ignore();
        let extension = if allowed && !is_dir {
            indexable_extension(&path)
        } else {
            None
        };

        match extension {
            Some(extension) if path.is_file() => {
                index_one_file(
                    resource_name,
                    &path,
                    &extension,
                    client,
                    writer,
                    config,
                    cancellation,
                )
                .await;
            }
            _ => {
                // Deleted, renamed away, now ignored, or no longer a
                // recognized extension: prune whatever chunks it had.
                // Passing an empty chunk set deletes everything currently
                // recorded for this exact path in one transaction.
                let _ = writer
                    .replace_file_chunks(resource_name, &path.to_string_lossy(), Vec::new())
                    .await;
            }
        }
    }
}

/// Builds a single ignore matcher from every `.gitignore` under `root`,
/// correctly scoped per-directory (verified against the `ignore` crate's
/// source: each pattern remembers the `.gitignore` file it came from and
/// is matched relative to that file's directory, not `root`).
fn build_ignore_matcher(root: &Path) -> Gitignore {
    let mut builder = GitignoreBuilder::new(root);
    // `hidden(false)`: .gitignore files are themselves dotfiles, and
    // WalkBuilder's default hidden-file filter applies to what the
    // iterator *yields*, separately from (and in addition to) the
    // ignore-matching it does internally while descending -- so without
    // this, this walk would never see a .gitignore to add in the first
    // place.
    for entry in WalkBuilder::new(root)
        .require_git(false)
        .hidden(false)
        .build()
        .filter_map(std::result::Result::ok)
    {
        if entry.file_name() == ".gitignore" {
            let _ = builder.add(entry.path());
        }
    }
    builder.build().unwrap_or_else(|_| {
        GitignoreBuilder::new(root)
            .build()
            .unwrap_or_else(|_| Gitignore::empty())
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::db::Database;
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

    async fn poll_until<F, Fut>(timeout: Duration, mut check: F) -> bool
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
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    struct TestSetup {
        project: tempfile::TempDir,
        db: Arc<Database>,
        watcher: ResourceWatcher,
    }

    async fn setup() -> TestSetup {
        let project = tempfile::tempdir().expect("tempdir");
        let data_dir = tempfile::tempdir().expect("tempdir");
        let db = Arc::new(
            Database::connect(data_dir.keep().as_path(), EMBED_DIM)
                .await
                .expect("connect"),
        );
        let embed_base = mock_embed_server().await;
        let client = Arc::new(
            OpenAiClient::new(&embed_base, "", "test-embed", &embed_base, "", "test-llm")
                .expect("client"),
        );
        let writer = ChunkWriter::spawn(Arc::clone(&db));

        let watcher = ResourceWatcher::spawn(
            "proj".to_string(),
            project.path().to_path_buf(),
            Arc::clone(&client),
            writer.clone(),
            ChunkingConfig::default(),
            Duration::from_millis(50),
        )
        .expect("spawn watcher");

        TestSetup {
            project,
            db,
            watcher,
        }
    }

    #[tokio::test]
    async fn watcher_indexes_newly_created_file() {
        let setup = setup().await;
        std::fs::write(setup.project.path().join("a.rs"), "fn a() {}\n").expect("write a.rs");

        let found = poll_until(Duration::from_secs(5), || async {
            setup
                .db
                .query_similar("proj", &[1.0; EMBED_DIM], 10)
                .await
                .map(|r| !r.is_empty())
                .unwrap_or(false)
        })
        .await;

        assert!(found, "watcher should have indexed the newly created file");
        setup.watcher.stop();
    }

    #[tokio::test]
    async fn watcher_prunes_chunks_when_file_deleted() {
        let setup = setup().await;
        let file_path = setup.project.path().join("a.rs");
        std::fs::write(&file_path, "fn a() {}\n").expect("write a.rs");

        poll_until(Duration::from_secs(5), || async {
            setup
                .db
                .query_similar("proj", &[1.0; EMBED_DIM], 10)
                .await
                .map(|r| !r.is_empty())
                .unwrap_or(false)
        })
        .await;

        std::fs::remove_file(&file_path).expect("remove a.rs");

        let pruned = poll_until(Duration::from_secs(5), || async {
            setup
                .db
                .query_similar("proj", &[1.0; EMBED_DIM], 10)
                .await
                .map(|r| r.is_empty())
                .unwrap_or(false)
        })
        .await;

        assert!(
            pruned,
            "watcher should have pruned chunks for the deleted file"
        );
        setup.watcher.stop();
    }

    #[tokio::test]
    async fn watcher_ignores_new_files_matching_nested_gitignore() {
        let setup = setup().await;
        std::fs::create_dir_all(setup.project.path().join("vendor")).expect("mkdir vendor");
        std::fs::write(
            setup.project.path().join("vendor/.gitignore"),
            "ignored.rs\n",
        )
        .expect("write nested gitignore");

        // Let the watcher observe the .gitignore itself first.
        tokio::time::sleep(Duration::from_millis(200)).await;

        std::fs::write(
            setup.project.path().join("vendor/ignored.rs"),
            "fn ignored() {}\n",
        )
        .expect("write ignored.rs");
        std::fs::write(setup.project.path().join("kept.rs"), "fn kept() {}\n")
            .expect("write kept.rs");

        let found_kept = poll_until(Duration::from_secs(5), || async {
            setup
                .db
                .query_similar("proj", &[1.0; EMBED_DIM], 10)
                .await
                .map(|r| !r.is_empty())
                .unwrap_or(false)
        })
        .await;
        assert!(found_kept, "kept.rs should have been indexed");

        // Give the (correctly ignored) file a chance to have been picked up if the bug were present.
        tokio::time::sleep(Duration::from_millis(300)).await;
        let results = setup
            .db
            .query_similar("proj", &[1.0; EMBED_DIM], 10)
            .await
            .expect("query");
        assert!(
            results
                .iter()
                .all(|chunk| !chunk.file_path.ends_with("ignored.rs")),
            "ignored.rs must not be indexed: {results:?}"
        );

        setup.watcher.stop();
    }

    #[tokio::test]
    async fn watcher_stop_halts_further_reconciliation() {
        let setup = setup().await;
        setup.watcher.stop();

        std::fs::write(setup.project.path().join("a.rs"), "fn a() {}\n").expect("write a.rs");
        tokio::time::sleep(Duration::from_millis(300)).await;

        let results = setup
            .db
            .query_similar("proj", &[1.0; EMBED_DIM], 10)
            .await
            .expect("query");
        assert!(
            results.is_empty(),
            "no reconciliation should happen after stop()"
        );
    }
}
