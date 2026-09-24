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

//! CLI/environment configuration.

use std::path::PathBuf;

use clap::Parser;

/// `ragd` runtime configuration, parsed from CLI flags or environment
/// variables.
#[derive(Debug, Clone, Parser)]
#[command(name = "ragd", about = "Generic RAG indexing/retrieval daemon")]
pub struct Config {
    /// Directory for persistent storage (the LanceDB database lives here).
    #[arg(long, env = "DATA_DIR")]
    pub data_dir: PathBuf,

    /// Port to listen on for the HTTP API.
    #[arg(long, env = "PORT", default_value_t = 20250)]
    pub port: u16,

    /// OpenAI-compatible embeddings endpoint, e.g. `http://localhost:8080/v1`.
    #[arg(long, env = "EMBED_ENDPOINT")]
    pub embed_endpoint: String,

    /// API key for the embeddings endpoint (may be empty for local servers).
    #[arg(long, env = "EMBED_API_KEY", default_value = "")]
    pub embed_api_key: String,

    /// Embedding model name.
    #[arg(long, env = "EMBED_MODEL")]
    pub embed_model: String,

    /// Dimensionality of the embedding model's output vectors. Fixed at
    /// table-creation time, so this must match whatever `embed_model`
    /// actually produces.
    #[arg(long, env = "EMBED_DIM")]
    pub embed_dim: usize,

    /// OpenAI-compatible chat-completions endpoint.
    #[arg(long, env = "LLM_ENDPOINT")]
    pub llm_endpoint: String,

    /// API key for the chat-completions endpoint (may be empty for local servers).
    #[arg(long, env = "LLM_API_KEY", default_value = "")]
    pub llm_api_key: String,

    /// Chat/LLM model name.
    #[arg(long, env = "LLM_MODEL")]
    pub llm_model: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_required_flags() {
        let config = Config::parse_from([
            "ragd",
            "--data-dir",
            "/tmp/ragd-data",
            "--embed-endpoint",
            "http://localhost:8080/v1",
            "--embed-model",
            "test-embed",
            "--embed-dim",
            "4",
            "--llm-endpoint",
            "http://localhost:8080/v1",
            "--llm-model",
            "test-llm",
        ]);

        assert_eq!(config.data_dir, PathBuf::from("/tmp/ragd-data"));
        assert_eq!(config.port, 20250);
        assert_eq!(config.embed_api_key, "");
        assert_eq!(config.llm_model, "test-llm");
    }
}
