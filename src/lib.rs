//! `ragd`: a generic RAG indexing/retrieval daemon.
//!
//! Watches configured directories, chunks and embeds their contents, and
//! serves semantic-search-backed answers over HTTP. Not tied to any
//! particular editor or client.

#![warn(clippy::all)]
#![warn(clippy::unwrap_used, clippy::expect_used)]

pub mod chunker;
pub mod config;
pub mod db;
pub mod error;
pub mod openai_client;
pub mod resource;
pub mod server;
pub mod watcher;

pub use error::{RagdError, Result};
