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

//! Storage layer: LanceDB-backed resource and chunk tables.
//!
//! The core correctness property this module provides is atomic
//! upsert-and-prune: [`Database::replace_file_chunks`] replaces *exactly*
//! one file's chunk set in a single `merge_insert` transaction, so
//! re-indexing unchanged content is a no-op and a file whose chunk
//! boundaries shift never leaves orphaned rows behind.

use std::path::Path;
use std::sync::Arc;

use arrow_array::builder::{Int32Builder, StringBuilder};
use arrow_array::types::Float32Type;
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Int32Array, RecordBatch, RecordBatchIterator, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use serde::{Deserialize, Serialize};

use crate::error::{RagdError, Result};

const RESOURCES_TABLE: &str = "resources";
const CHUNKS_TABLE: &str = "chunks";

/// Whether a resource is actively watched/indexed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceStatus {
    Active,
    Inactive,
}

impl ResourceStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Inactive => "inactive",
        }
    }
}

impl std::str::FromStr for ResourceStatus {
    type Err = RagdError;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "active" => Ok(Self::Active),
            "inactive" => Ok(Self::Inactive),
            other => Err(RagdError::Storage(format!(
                "unknown resource status: {other}"
            ))),
        }
    }
}

impl Serialize for ResourceStatus {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for ResourceStatus {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

/// Where a resource is in its indexing lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexingStatus {
    Pending,
    Indexing,
    Indexed,
    Failed,
}

impl IndexingStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Indexing => "indexing",
            Self::Indexed => "indexed",
            Self::Failed => "failed",
        }
    }
}

impl std::str::FromStr for IndexingStatus {
    type Err = RagdError;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "pending" => Ok(Self::Pending),
            "indexing" => Ok(Self::Indexing),
            "indexed" => Ok(Self::Indexed),
            "failed" => Ok(Self::Failed),
            other => Err(RagdError::Storage(format!(
                "unknown indexing status: {other}"
            ))),
        }
    }
}

impl Serialize for IndexingStatus {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for IndexingStatus {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

/// A watched/indexed resource (a local directory, today; remote URIs may
/// follow later).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceRecord {
    pub uri: String,
    pub name: String,
    pub status: ResourceStatus,
    pub indexing_status: IndexingStatus,
    pub indexing_status_message: Option<String>,
    pub created_at: String,
    pub indexing_started_at: Option<String>,
    pub last_indexed_at: Option<String>,
    pub last_error: Option<String>,
}

/// A single embedded chunk of a file, scoped to the resource that owns it.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkRecord {
    pub resource_name: String,
    pub file_path: String,
    pub chunk_index: i32,
    pub content: String,
    pub content_hash: String,
    pub start_line: i32,
    pub end_line: i32,
    pub embedding: Vec<f32>,
    pub updated_at: String,
}

impl ChunkRecord {
    /// The stable, globally unique key this chunk is upserted on.
    fn chunk_key(&self) -> String {
        chunk_key(&self.resource_name, &self.file_path, self.chunk_index)
    }
}

/// Just enough of a [`ChunkRecord`] to decide whether a freshly chunked
/// piece of content is identical to what's already stored, and to reuse
/// its embedding if so: `content`, `start_line`, `end_line`, and
/// `updated_at` get recomputed from the current chunking pass regardless,
/// and `resource_name`/`file_path` are the caller's lookup key already --
/// none of that is worth carrying across the wire or holding in a cache
/// just to answer "did this change".
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkFingerprint {
    pub content_hash: String,
    pub embedding: Vec<f32>,
}

/// NUL-byte separated so that resource names or file paths containing `:`
/// (or any other printable delimiter) can't collide with a different
/// resource/file/index tuple.
fn chunk_key(resource_name: &str, file_path: &str, chunk_index: i32) -> String {
    format!("{resource_name}\u{0}{file_path}\u{0}{chunk_index}")
}

/// Escapes a value for embedding in a LanceDB/DataFusion SQL filter string.
fn sql_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

impl From<lancedb::Error> for RagdError {
    fn from(err: lancedb::Error) -> Self {
        RagdError::Storage(err.to_string())
    }
}

impl From<arrow_schema::ArrowError> for RagdError {
    fn from(err: arrow_schema::ArrowError) -> Self {
        RagdError::Storage(err.to_string())
    }
}

/// The result of an upsert-and-prune call: how many chunks were written and
/// how many stale ones were removed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ReplaceChunksOutcome {
    pub written: u64,
    pub deleted: u64,
}

/// Handle to the daemon's LanceDB-backed storage.
///
/// Each operation re-resolves its table via [`Self::ensure_resources_table`]
/// / [`Self::ensure_chunks_table`] rather than holding a long-lived
/// `lancedb::Table` handle. A `Table` carries a `DatasetConsistencyWrapper`
/// with an `Arc<Mutex<DatasetState>>` shared across every clone; LanceDB's
/// own test suite (`dataset.rs`: `test_get_returns_error_on_poisoned_lock`
/// et al.) treats a poisoned lock on that shared state as an expected
/// condition a long-lived handle can reach. A fresh handle per call means
/// one bad operation can't leave every later request on this connection
/// permanently wedged.
pub struct Database {
    connection: lancedb::Connection,
    embed_dim: usize,
}

impl Database {
    /// Opens (creating if necessary) the LanceDB database rooted at
    /// `data_dir`. `embed_dim` is the fixed dimensionality of the
    /// configured embedding model and determines the `chunks` table's
    /// vector column width.
    pub async fn connect(data_dir: &Path, embed_dim: usize) -> Result<Self> {
        tokio::fs::create_dir_all(data_dir).await?;
        let uri = data_dir.to_string_lossy().into_owned();
        let connection = lancedb::connect(&uri).execute().await?;
        let db = Self {
            connection,
            embed_dim,
        };
        db.ensure_resources_table().await?;
        db.ensure_chunks_table().await?;
        Ok(db)
    }

    fn resources_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("uri", DataType::Utf8, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("status", DataType::Utf8, false),
            Field::new("indexing_status", DataType::Utf8, false),
            Field::new("indexing_status_message", DataType::Utf8, true),
            Field::new("created_at", DataType::Utf8, false),
            Field::new("indexing_started_at", DataType::Utf8, true),
            Field::new("last_indexed_at", DataType::Utf8, true),
            Field::new("last_error", DataType::Utf8, true),
        ]))
    }

    fn chunks_schema(embed_dim: usize) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("chunk_key", DataType::Utf8, false),
            Field::new("resource_name", DataType::Utf8, false),
            Field::new("file_path", DataType::Utf8, false),
            Field::new("chunk_index", DataType::Int32, false),
            Field::new("content", DataType::Utf8, false),
            Field::new("content_hash", DataType::Utf8, false),
            Field::new("start_line", DataType::Int32, false),
            Field::new("end_line", DataType::Int32, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    embed_dim as i32,
                ),
                false,
            ),
            Field::new("updated_at", DataType::Utf8, false),
        ]))
    }

    /// Opens `table_name` if it already exists, creating it (empty, with
    /// `schema`) otherwise.
    async fn open_or_create_table(
        connection: &lancedb::Connection,
        table_name: &str,
        schema: Arc<Schema>,
    ) -> Result<lancedb::Table> {
        let names = connection.table_names().execute().await?;
        if names.iter().any(|n| n == table_name) {
            return Ok(connection.open_table(table_name).execute().await?);
        }
        Ok(connection
            .create_empty_table(table_name, schema)
            .execute()
            .await?)
    }

    async fn ensure_resources_table(&self) -> Result<lancedb::Table> {
        Self::open_or_create_table(&self.connection, RESOURCES_TABLE, Self::resources_schema())
            .await
    }

    async fn ensure_chunks_table(&self) -> Result<lancedb::Table> {
        Self::open_or_create_table(
            &self.connection,
            CHUNKS_TABLE,
            Self::chunks_schema(self.embed_dim),
        )
        .await
    }

    /// Inserts or updates a resource, keyed on its URI.
    pub async fn upsert_resource(&self, resource: &ResourceRecord) -> Result<()> {
        let table = self.ensure_resources_table().await?;
        let schema = Self::resources_schema();
        let batch = resource_records_to_batch(&schema, std::slice::from_ref(resource))?;
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);

        let mut merge = table.merge_insert(&["uri"]);
        merge
            .when_matched_update_all(None)
            .when_not_matched_insert_all();
        merge.execute(Box::new(reader)).await?;
        Ok(())
    }

    /// Looks up a resource by URI.
    pub async fn get_resource(&self, uri: &str) -> Result<Option<ResourceRecord>> {
        let mut resources = self
            .find_resources(&format!("uri = {}", sql_quote(uri)))
            .await?;
        Ok(resources.pop())
    }

    /// Looks up a resource by its user-facing name.
    pub async fn get_resource_by_name(&self, name: &str) -> Result<Option<ResourceRecord>> {
        let mut resources = self
            .find_resources(&format!("name = {}", sql_quote(name)))
            .await?;
        Ok(resources.pop())
    }

    /// Lists every resource, regardless of status.
    pub async fn list_resources(&self) -> Result<Vec<ResourceRecord>> {
        self.find_resources("true").await
    }

    async fn find_resources(&self, filter: &str) -> Result<Vec<ResourceRecord>> {
        let table = self.ensure_resources_table().await?;
        let batches: Vec<RecordBatch> = table
            .query()
            .only_if(filter)
            .execute()
            .await?
            .try_collect()
            .await?;

        batches
            .iter()
            .filter(|b| b.num_rows() > 0)
            .flat_map(resource_records_from_batch)
            .collect()
    }

    /// Atomically replaces the full chunk set for one file within one
    /// resource: matching chunk keys are updated, new ones are inserted,
    /// and any chunk previously recorded for this file but absent from
    /// `chunks` is deleted — all in a single `merge_insert` transaction.
    ///
    /// Calling this twice with the same `chunks` is a no-op the second
    /// time; this is the property that makes re-indexing on daemon
    /// restart safe.
    ///
    /// # Panics
    ///
    /// Panics if any chunk's embedding length doesn't match the
    /// configured embedding dimension, or if `chunks` contains entries for
    /// a resource/file other than the ones passed in — both indicate a
    /// caller bug, not a recoverable runtime condition.
    pub async fn replace_file_chunks(
        &self,
        resource_name: &str,
        file_path: &str,
        chunks: &[ChunkRecord],
    ) -> Result<ReplaceChunksOutcome> {
        for chunk in chunks {
            assert_eq!(
                chunk.embedding.len(),
                self.embed_dim,
                "chunk embedding dimension mismatch: expected {}, got {}",
                self.embed_dim,
                chunk.embedding.len()
            );
            assert_eq!(
                chunk.resource_name, resource_name,
                "chunk resource_name does not match target resource"
            );
            assert_eq!(
                chunk.file_path, file_path,
                "chunk file_path does not match target file"
            );
        }

        let table = self.ensure_chunks_table().await?;
        let batch = self.chunks_to_record_batch(chunks)?;
        let schema = batch.schema();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);

        let scope_filter = format!(
            "resource_name = {} AND file_path = {}",
            sql_quote(resource_name),
            sql_quote(file_path)
        );

        let mut merge = table.merge_insert(&["chunk_key"]);
        merge
            .when_matched_update_all(None)
            .when_not_matched_insert_all()
            .when_not_matched_by_source_delete(Some(scope_filter));
        let result = merge.execute(Box::new(reader)).await?;

        Ok(ReplaceChunksOutcome {
            written: result.num_inserted_rows + result.num_updated_rows,
            deleted: result.num_deleted_rows,
        })
    }

    /// Returns `(chunk_index, fingerprint)` for every chunk currently
    /// stored for `file_path` within `resource_name` -- used to check
    /// `content_hash` against freshly chunked content before paying for a
    /// re-embed of content that hasn't actually changed. Projects down to
    /// just `chunk_index`, `content_hash`, and `embedding`: `content` in
    /// particular can be as large as (or larger than) the embedding
    /// vector, and there's no reason to pull it off disk for a hash
    /// comparison the caller's about to throw it away after.
    ///
    /// `only_if` with no index is a scan of every chunk this daemon has
    /// stored, across every resource, not just this one -- LanceDB has no
    /// notion of a per-table "cheap filter" without a scalar index, and a
    /// scalar index here would need periodic `optimize()` calls to cover
    /// new writes (indices aren't updated automatically; unindexed rows
    /// still get a flat scan) for a benefit LanceDB's own docs say doesn't
    /// start to matter until ~100K+ rows. Accepting the full scan is the
    /// right call at the scale this daemon actually runs at (one process
    /// tracking a handful of project-sized resources), not because the
    /// scan is somehow cheaper than it looks.
    pub async fn get_file_chunk_fingerprints(
        &self,
        resource_name: &str,
        file_path: &str,
    ) -> Result<Vec<(i32, ChunkFingerprint)>> {
        let table = self.ensure_chunks_table().await?;
        let filter = format!(
            "resource_name = {} AND file_path = {}",
            sql_quote(resource_name),
            sql_quote(file_path)
        );
        let batches: Vec<RecordBatch> = table
            .query()
            .only_if(filter)
            .select(Select::columns(&[
                "chunk_index",
                "content_hash",
                "embedding",
            ]))
            .execute()
            .await?
            .try_collect()
            .await?;
        Ok(batches
            .iter()
            .flat_map(chunk_fingerprints_from_batch)
            .collect())
    }

    /// Removes every chunk belonging to a resource (used when a resource is
    /// removed entirely, not just when one file within it changes).
    pub async fn delete_resource_chunks(&self, resource_name: &str) -> Result<()> {
        let table = self.ensure_chunks_table().await?;
        table
            .delete(&format!("resource_name = {}", sql_quote(resource_name)))
            .await?;
        Ok(())
    }

    /// Current version number of the chunks table. Test-only: lets tests
    /// assert that a write was actually skipped (e.g. unchanged content,
    /// or a reconciled path that was never indexable) rather than merely
    /// having no visible effect -- a no-op `merge_insert` still commits a
    /// new table version, which `query_similar`-style assertions can't see.
    #[cfg(test)]
    pub(crate) async fn chunks_table_version(&self) -> Result<u64> {
        Ok(self.ensure_chunks_table().await?.version().await?)
    }

    /// Returns the `top_k` chunks within `resource_name` nearest to
    /// `query_embedding`, ranked by vector distance.
    ///
    /// # Panics
    ///
    /// Panics if `query_embedding.len()` doesn't match the configured
    /// embedding dimension — a caller bug (the query must be embedded with
    /// the same model as the corpus).
    pub async fn query_similar(
        &self,
        resource_name: &str,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Result<Vec<ChunkRecord>> {
        let batches = self
            .query_similar_batches(resource_name, query_embedding, top_k)
            .await?;
        batches.iter().flat_map(chunk_records_from_batch).collect()
    }

    /// Like [`Self::query_similar`], but pairs each chunk with a
    /// `0.0..=1.0` similarity score (`1.0` = identical direction, under
    /// cosine similarity), highest first. Used for the `/query` endpoint,
    /// which needs to expose relevance to the caller; `query_similar`
    /// doesn't, so it's kept score-free rather than have every caller
    /// discard a value they don't need.
    ///
    /// # Panics
    ///
    /// Same precondition as [`Self::query_similar`].
    pub async fn query_similar_with_scores(
        &self,
        resource_name: &str,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Result<Vec<(ChunkRecord, f32)>> {
        let batches = self
            .query_similar_batches(resource_name, query_embedding, top_k)
            .await?;

        let mut results = Vec::new();
        for batch in &batches {
            let distances = downcast_f32(column(batch, "_distance"));
            for (row, chunk_result) in chunk_records_from_batch(batch).into_iter().enumerate() {
                // LanceDB's cosine "_distance" is 1 - cosine_similarity (see
                // lance_linalg::distance::cosine, verified against the exact
                // version we depend on); invert it back to a similarity score
                // so callers see "higher is more relevant".
                results.push((chunk_result?, 1.0 - distances.value(row)));
            }
        }
        Ok(results)
    }

    async fn query_similar_batches(
        &self,
        resource_name: &str,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Result<Vec<RecordBatch>> {
        assert_eq!(
            query_embedding.len(),
            self.embed_dim,
            "query embedding dimension mismatch: expected {}, got {}",
            self.embed_dim,
            query_embedding.len()
        );

        let table = self.ensure_chunks_table().await?;
        Ok(table
            .query()
            .only_if(format!("resource_name = {}", sql_quote(resource_name)))
            .nearest_to(query_embedding)?
            .distance_type(lancedb::DistanceType::Cosine)
            .limit(top_k)
            .execute()
            .await?
            .try_collect()
            .await?)
    }

    fn chunks_to_record_batch(&self, chunks: &[ChunkRecord]) -> Result<RecordBatch> {
        let mut chunk_key_b = StringBuilder::new();
        let mut resource_name_b = StringBuilder::new();
        let mut file_path_b = StringBuilder::new();
        let mut chunk_index_b = Int32Builder::new();
        let mut content_b = StringBuilder::new();
        let mut content_hash_b = StringBuilder::new();
        let mut start_line_b = Int32Builder::new();
        let mut end_line_b = Int32Builder::new();
        let mut updated_at_b = StringBuilder::new();

        for chunk in chunks {
            chunk_key_b.append_value(chunk.chunk_key());
            resource_name_b.append_value(&chunk.resource_name);
            file_path_b.append_value(&chunk.file_path);
            chunk_index_b.append_value(chunk.chunk_index);
            content_b.append_value(&chunk.content);
            content_hash_b.append_value(&chunk.content_hash);
            start_line_b.append_value(chunk.start_line);
            end_line_b.append_value(chunk.end_line);
            updated_at_b.append_value(&chunk.updated_at);
        }

        let embedding_array: ArrayRef =
            Arc::new(
                FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                    chunks.iter().map(|chunk| {
                        Some(
                            chunk
                                .embedding
                                .iter()
                                .copied()
                                .map(Some)
                                .collect::<Vec<_>>(),
                        )
                    }),
                    self.embed_dim as i32,
                ),
            );

        let columns: Vec<ArrayRef> = vec![
            Arc::new(chunk_key_b.finish()),
            Arc::new(resource_name_b.finish()),
            Arc::new(file_path_b.finish()),
            Arc::new(chunk_index_b.finish()),
            Arc::new(content_b.finish()),
            Arc::new(content_hash_b.finish()),
            Arc::new(start_line_b.finish()),
            Arc::new(end_line_b.finish()),
            embedding_array,
            Arc::new(updated_at_b.finish()),
        ];

        Ok(RecordBatch::try_new(
            Self::chunks_schema(self.embed_dim),
            columns,
        )?)
    }
}

fn column<'a>(batch: &'a RecordBatch, name: &str) -> &'a ArrayRef {
    let Some(col) = batch.column_by_name(name) else {
        panic!("chunks table missing expected column `{name}` (schema/query bug)");
    };
    col
}

fn resource_records_to_batch(
    schema: &Arc<Schema>,
    resources: &[ResourceRecord],
) -> Result<RecordBatch> {
    let mut uri_b = StringBuilder::new();
    let mut name_b = StringBuilder::new();
    let mut status_b = StringBuilder::new();
    let mut indexing_status_b = StringBuilder::new();
    let mut indexing_status_message_b = StringBuilder::new();
    let mut created_at_b = StringBuilder::new();
    let mut indexing_started_at_b = StringBuilder::new();
    let mut last_indexed_at_b = StringBuilder::new();
    let mut last_error_b = StringBuilder::new();

    for resource in resources {
        uri_b.append_value(&resource.uri);
        name_b.append_value(&resource.name);
        status_b.append_value(resource.status.as_str());
        indexing_status_b.append_value(resource.indexing_status.as_str());
        indexing_status_message_b.append_option(resource.indexing_status_message.as_deref());
        created_at_b.append_value(&resource.created_at);
        indexing_started_at_b.append_option(resource.indexing_started_at.as_deref());
        last_indexed_at_b.append_option(resource.last_indexed_at.as_deref());
        last_error_b.append_option(resource.last_error.as_deref());
    }

    let columns: Vec<ArrayRef> = vec![
        Arc::new(uri_b.finish()),
        Arc::new(name_b.finish()),
        Arc::new(status_b.finish()),
        Arc::new(indexing_status_b.finish()),
        Arc::new(indexing_status_message_b.finish()),
        Arc::new(created_at_b.finish()),
        Arc::new(indexing_started_at_b.finish()),
        Arc::new(last_indexed_at_b.finish()),
        Arc::new(last_error_b.finish()),
    ];

    Ok(RecordBatch::try_new(schema.clone(), columns)?)
}

fn resource_records_from_batch(batch: &RecordBatch) -> Vec<Result<ResourceRecord>> {
    let uri = downcast_string(column(batch, "uri"));
    let name = downcast_string(column(batch, "name"));
    let status = downcast_string(column(batch, "status"));
    let indexing_status = downcast_string(column(batch, "indexing_status"));
    let indexing_status_message = downcast_string(column(batch, "indexing_status_message"));
    let created_at = downcast_string(column(batch, "created_at"));
    let indexing_started_at = downcast_string(column(batch, "indexing_started_at"));
    let last_indexed_at = downcast_string(column(batch, "last_indexed_at"));
    let last_error = downcast_string(column(batch, "last_error"));

    (0..batch.num_rows())
        .map(|row| {
            Ok(ResourceRecord {
                uri: uri.value(row).to_string(),
                name: name.value(row).to_string(),
                status: status.value(row).parse()?,
                indexing_status: indexing_status.value(row).parse()?,
                indexing_status_message: opt_string(indexing_status_message, row),
                created_at: created_at.value(row).to_string(),
                indexing_started_at: opt_string(indexing_started_at, row),
                last_indexed_at: opt_string(last_indexed_at, row),
                last_error: opt_string(last_error, row),
            })
        })
        .collect()
}

fn opt_string(array: &StringArray, row: usize) -> Option<String> {
    if array.is_null(row) {
        None
    } else {
        Some(array.value(row).to_string())
    }
}

fn chunk_records_from_batch(batch: &RecordBatch) -> Vec<Result<ChunkRecord>> {
    let resource_name = downcast_string(column(batch, "resource_name"));
    let file_path = downcast_string(column(batch, "file_path"));
    let chunk_index = downcast_i32(column(batch, "chunk_index"));
    let content = downcast_string(column(batch, "content"));
    let content_hash = downcast_string(column(batch, "content_hash"));
    let start_line = downcast_i32(column(batch, "start_line"));
    let end_line = downcast_i32(column(batch, "end_line"));
    let updated_at = downcast_string(column(batch, "updated_at"));
    let embedding_col = column(batch, "embedding");
    let Some(embedding_col) = embedding_col
        .as_any()
        .downcast_ref::<arrow_array::FixedSizeListArray>()
    else {
        panic!("`embedding` column is not a FixedSizeListArray (schema bug)");
    };

    (0..batch.num_rows())
        .map(|row| {
            let Some(values) = embedding_col
                .value(row)
                .as_any()
                .downcast_ref::<arrow_array::Float32Array>()
                .map(|a| a.values().to_vec())
            else {
                panic!("embedding list element is not a Float32Array (schema bug)");
            };
            Ok(ChunkRecord {
                resource_name: resource_name.value(row).to_string(),
                file_path: file_path.value(row).to_string(),
                chunk_index: chunk_index.value(row),
                content: content.value(row).to_string(),
                content_hash: content_hash.value(row).to_string(),
                start_line: start_line.value(row),
                end_line: end_line.value(row),
                embedding: values,
                updated_at: updated_at.value(row).to_string(),
            })
        })
        .collect()
}

/// Decodes a batch produced by the `chunk_index, content_hash, embedding`
/// projection in [`Database::get_file_chunk_fingerprints`]. Panics on a
/// malformed `embedding` column for the same reason
/// [`chunk_records_from_batch`] does: it would mean the schema and the
/// query projection have drifted apart, which is a bug, not bad input.
fn chunk_fingerprints_from_batch(batch: &RecordBatch) -> Vec<(i32, ChunkFingerprint)> {
    let chunk_index = downcast_i32(column(batch, "chunk_index"));
    let content_hash = downcast_string(column(batch, "content_hash"));
    let embedding_col = column(batch, "embedding");
    let Some(embedding_col) = embedding_col
        .as_any()
        .downcast_ref::<arrow_array::FixedSizeListArray>()
    else {
        panic!("`embedding` column is not a FixedSizeListArray (schema bug)");
    };

    (0..batch.num_rows())
        .map(|row| {
            let Some(values) = embedding_col
                .value(row)
                .as_any()
                .downcast_ref::<arrow_array::Float32Array>()
                .map(|a| a.values().to_vec())
            else {
                panic!("embedding list element is not a Float32Array (schema bug)");
            };
            (
                chunk_index.value(row),
                ChunkFingerprint {
                    content_hash: content_hash.value(row).to_string(),
                    embedding: values,
                },
            )
        })
        .collect()
}

fn downcast_string(array: &ArrayRef) -> &StringArray {
    let Some(array) = array.as_any().downcast_ref::<StringArray>() else {
        panic!("expected a Utf8 column (schema bug)");
    };
    array
}

fn downcast_i32(array: &ArrayRef) -> &Int32Array {
    let Some(array) = array.as_any().downcast_ref::<Int32Array>() else {
        panic!("expected an Int32 column (schema bug)");
    };
    array
}

fn downcast_f32(array: &ArrayRef) -> &arrow_array::Float32Array {
    let Some(array) = array.as_any().downcast_ref::<arrow_array::Float32Array>() else {
        panic!("expected a Float32 column (schema bug)");
    };
    array
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    const EMBED_DIM: usize = 4;

    async fn test_db() -> Database {
        let dir = tempfile::tempdir().expect("tempdir");
        // Leak the tempdir so it outlives the returned Database in tests;
        // the OS cleans it up eventually and these are short-lived test runs.
        let path = dir.keep();
        Database::connect(&path, EMBED_DIM).await.expect("connect")
    }

    fn sample_resource(uri: &str, name: &str) -> ResourceRecord {
        ResourceRecord {
            uri: uri.to_string(),
            name: name.to_string(),
            status: ResourceStatus::Active,
            indexing_status: IndexingStatus::Pending,
            indexing_status_message: None,
            created_at: "2026-09-24T00:00:00Z".to_string(),
            indexing_started_at: None,
            last_indexed_at: None,
            last_error: None,
        }
    }

    fn sample_chunk(
        resource_name: &str,
        file_path: &str,
        chunk_index: i32,
        content: &str,
    ) -> ChunkRecord {
        ChunkRecord {
            resource_name: resource_name.to_string(),
            file_path: file_path.to_string(),
            chunk_index,
            content: content.to_string(),
            content_hash: format!("hash-{content}"),
            start_line: chunk_index * 10,
            end_line: chunk_index * 10 + 9,
            // +1 so chunk_index 0 never produces a degenerate all-zero
            // vector, which cosine similarity (used by query_similar) can't
            // meaningfully compare against anything.
            embedding: vec![(chunk_index + 1) as f32; EMBED_DIM],
            updated_at: "2026-09-24T00:00:00Z".to_string(),
        }
    }

    #[tokio::test]
    async fn upsert_and_get_resource_round_trips() {
        let db = test_db().await;
        let resource = sample_resource("file:///tmp/proj/", "proj");

        db.upsert_resource(&resource).await.expect("upsert");
        let fetched = db
            .get_resource("file:///tmp/proj/")
            .await
            .expect("get")
            .expect("present");

        assert_eq!(fetched, resource);
    }

    #[tokio::test]
    async fn upsert_resource_twice_does_not_duplicate() {
        let db = test_db().await;
        let mut resource = sample_resource("file:///tmp/proj/", "proj");

        db.upsert_resource(&resource).await.expect("upsert 1");
        resource.indexing_status = IndexingStatus::Indexed;
        db.upsert_resource(&resource).await.expect("upsert 2");

        let all = db.list_resources().await.expect("list");
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].indexing_status, IndexingStatus::Indexed);
    }

    #[tokio::test]
    async fn replace_file_chunks_is_idempotent() {
        let db = test_db().await;
        let chunks = vec![
            sample_chunk("proj", "/tmp/proj/a.rs", 0, "fn a() {}"),
            sample_chunk("proj", "/tmp/proj/a.rs", 1, "fn b() {}"),
        ];

        let first = db
            .replace_file_chunks("proj", "/tmp/proj/a.rs", &chunks)
            .await
            .expect("first replace");
        assert_eq!(first.written, 2);
        assert_eq!(first.deleted, 0);

        // Re-applying the exact same chunk set must not grow the table.
        let second = db
            .replace_file_chunks("proj", "/tmp/proj/a.rs", &chunks)
            .await
            .expect("second replace");
        assert_eq!(second.deleted, 0);

        let results = db
            .query_similar("proj", &[1.0; EMBED_DIM], 10)
            .await
            .expect("query");
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn replace_file_chunks_prunes_stale_chunks_when_boundaries_shift() {
        let db = test_db().await;
        let original = vec![
            sample_chunk("proj", "/tmp/proj/a.rs", 0, "fn a() {}"),
            sample_chunk("proj", "/tmp/proj/a.rs", 1, "fn b() {}"),
            sample_chunk("proj", "/tmp/proj/a.rs", 2, "fn c() {}"),
        ];
        db.replace_file_chunks("proj", "/tmp/proj/a.rs", &original)
            .await
            .expect("initial replace");

        // Simulate an edit that reflows the file into fewer, different chunks.
        let reflowed = vec![sample_chunk(
            "proj",
            "/tmp/proj/a.rs",
            0,
            "fn a() { /* edited */ }",
        )];
        let outcome = db
            .replace_file_chunks("proj", "/tmp/proj/a.rs", &reflowed)
            .await
            .expect("reflow replace");
        assert_eq!(outcome.deleted, 2);

        let results = db
            .query_similar("proj", &[1.0; EMBED_DIM], 10)
            .await
            .expect("query");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "fn a() { /* edited */ }");
    }

    #[tokio::test]
    async fn replace_file_chunks_does_not_touch_other_files() {
        let db = test_db().await;
        db.replace_file_chunks(
            "proj",
            "/tmp/proj/a.rs",
            &[sample_chunk("proj", "/tmp/proj/a.rs", 0, "a")],
        )
        .await
        .expect("a replace");
        db.replace_file_chunks(
            "proj",
            "/tmp/proj/b.rs",
            &[sample_chunk("proj", "/tmp/proj/b.rs", 0, "b")],
        )
        .await
        .expect("b replace");

        // Re-index a.rs with zero chunks (e.g. the file became empty) and
        // confirm b.rs's chunk survives untouched.
        db.replace_file_chunks("proj", "/tmp/proj/a.rs", &[])
            .await
            .expect("empty replace");

        let results = db
            .query_similar("proj", &[1.0; EMBED_DIM], 10)
            .await
            .expect("query");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].file_path, "/tmp/proj/b.rs");
    }

    #[tokio::test]
    async fn get_file_chunk_fingerprints_returns_hash_and_embedding_scoped_to_one_file() {
        let db = test_db().await;
        let a0 = sample_chunk("proj", "/tmp/proj/a.rs", 0, "fn a() {}");
        let a1 = sample_chunk("proj", "/tmp/proj/a.rs", 1, "fn b() {}");
        db.replace_file_chunks("proj", "/tmp/proj/a.rs", &[a0.clone(), a1.clone()])
            .await
            .expect("replace a");
        // A different file, and a different resource sharing the exact
        // same file path, must not leak into the lookup.
        db.replace_file_chunks(
            "proj",
            "/tmp/proj/b.rs",
            &[sample_chunk("proj", "/tmp/proj/b.rs", 0, "fn c() {}")],
        )
        .await
        .expect("replace b");
        db.replace_file_chunks(
            "other",
            "/tmp/proj/a.rs",
            &[sample_chunk("other", "/tmp/proj/a.rs", 0, "fn z() {}")],
        )
        .await
        .expect("replace other resource");

        let mut fingerprints = db
            .get_file_chunk_fingerprints("proj", "/tmp/proj/a.rs")
            .await
            .expect("fingerprints");
        fingerprints.sort_by_key(|(index, _)| *index);

        assert_eq!(fingerprints.len(), 2);
        assert_eq!(fingerprints[0].0, 0);
        assert_eq!(fingerprints[0].1.content_hash, a0.content_hash);
        assert_eq!(fingerprints[0].1.embedding, a0.embedding);
        assert_eq!(fingerprints[1].0, 1);
        assert_eq!(fingerprints[1].1.content_hash, a1.content_hash);
        assert_eq!(fingerprints[1].1.embedding, a1.embedding);
    }

    #[tokio::test]
    async fn query_similar_with_scores_ranks_by_cosine_similarity() {
        let db = test_db().await;
        let mut aligned = sample_chunk("proj", "/tmp/proj/a.rs", 0, "aligned");
        aligned.embedding = vec![1.0, 0.0, 0.0, 0.0];
        let mut orthogonal = sample_chunk("proj", "/tmp/proj/b.rs", 0, "orthogonal");
        orthogonal.embedding = vec![0.0, 1.0, 0.0, 0.0];

        db.replace_file_chunks("proj", "/tmp/proj/a.rs", &[aligned])
            .await
            .expect("replace a");
        db.replace_file_chunks("proj", "/tmp/proj/b.rs", &[orthogonal])
            .await
            .expect("replace b");

        let results = db
            .query_similar_with_scores("proj", &[1.0, 0.0, 0.0, 0.0], 10)
            .await
            .expect("query");

        assert_eq!(results.len(), 2);
        let (nearest, nearest_score) = &results[0];
        assert_eq!(nearest.content, "aligned");
        assert!(
            (*nearest_score - 1.0).abs() < 1e-4,
            "identical vectors should score ~1.0, got {nearest_score}"
        );

        let (_, farthest_score) = &results[1];
        assert!(
            farthest_score.abs() < 1e-4,
            "orthogonal vectors should score ~0.0, got {farthest_score}"
        );
    }

    #[tokio::test]
    #[should_panic(expected = "chunk embedding dimension mismatch")]
    async fn replace_file_chunks_panics_on_dimension_mismatch() {
        let db = test_db().await;
        let mut bad_chunk = sample_chunk("proj", "/tmp/proj/a.rs", 0, "a");
        bad_chunk.embedding = vec![0.0; EMBED_DIM + 1];

        let _ = db
            .replace_file_chunks("proj", "/tmp/proj/a.rs", &[bad_chunk])
            .await;
    }

    proptest::proptest! {
        #[test]
        fn replace_file_chunks_reapplication_is_a_no_op(
            contents in proptest::collection::vec("[a-z]{1,20}", 1..8)
        ) {
            let rt = tokio::runtime::Runtime::new().expect("runtime");
            rt.block_on(async {
                let db = test_db().await;
                let chunks: Vec<ChunkRecord> = contents
                    .iter()
                    .enumerate()
                    .map(|(i, c)| sample_chunk("proj", "/tmp/proj/a.rs", i as i32, c))
                    .collect();

                db.replace_file_chunks("proj", "/tmp/proj/a.rs", &chunks).await.expect("first");
                let second = db.replace_file_chunks("proj", "/tmp/proj/a.rs", &chunks).await.expect("second");

                prop_assert_eq!(second.deleted, 0);

                let results = db.query_similar("proj", &[1.0; EMBED_DIM], contents.len() + 1).await.expect("query");
                prop_assert_eq!(results.len(), chunks.len());
                Ok(())
            })?;
        }
    }
}
