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
