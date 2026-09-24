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

//! Crate-wide error type.

/// Convenience alias for [`Result`](std::result::Result) with [`RagdError`].
pub type Result<T> = std::result::Result<T, RagdError>;

/// All recoverable failure modes exposed by this crate.
///
/// Programmer errors (broken invariants, violated preconditions) are not
/// represented here — those `panic!` instead, per the project's error
/// handling conventions (see `AGENTS.md`).
#[derive(Debug, thiserror::Error)]
pub enum RagdError {
    /// An I/O operation failed (reading a file, binding a socket, etc.).
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// The daemon's configuration was invalid or incomplete.
    #[error("invalid configuration: {0}")]
    Config(String),

    /// A call to the storage layer failed.
    #[error("storage error: {0}")]
    Storage(String),

    /// A call to the configured embedding or chat endpoint failed.
    #[error("model endpoint error: {0}")]
    ModelEndpoint(String),

    /// A named resource was not found.
    #[error("resource not found: {0}")]
    ResourceNotFound(String),

    /// A resource with the requested name already exists.
    #[error("resource already exists: {0}")]
    ResourceAlreadyExists(String),
}
