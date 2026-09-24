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

//! Binary entry point: parses configuration and serves the HTTP API.

use std::sync::Arc;
use std::time::Duration;

use clap::Parser;
use ragd::chunker::ChunkingConfig;
use ragd::config::Config;
use ragd::db::{Database, ResourceStatus};
use ragd::openai_client::OpenAiClient;
use ragd::resource::{ChunkWriter, ResourceManager, uri_to_path};
use ragd::server::{self, AppState};

const WATCH_DEBOUNCE: Duration = Duration::from_secs(2);

#[tokio::main]
async fn main() -> ragd::Result<()> {
    println!(
        "ragd {}  Copyright (C) 2026  Daniel Arnold\n\
         This program comes with ABSOLUTELY NO WARRANTY.\n\
         This is free software, and you are welcome to redistribute it\n\
         under certain conditions; see the GNU General Public License\n\
         version 3 or later <https://www.gnu.org/licenses/> for details.",
        env!("CARGO_PKG_VERSION")
    );

    let config = Config::parse();

    let db = Arc::new(Database::connect(&config.data_dir, config.embed_dim).await?);
    let client = Arc::new(OpenAiClient::new(
        &config.embed_endpoint,
        &config.embed_api_key,
        &config.embed_model,
        &config.llm_endpoint,
        &config.llm_api_key,
        &config.llm_model,
    )?);
    let writer = ChunkWriter::spawn(Arc::clone(&db));
    let manager = ResourceManager::new(
        Arc::clone(&client),
        Arc::clone(&db),
        writer,
        ChunkingConfig::default(),
        WATCH_DEBOUNCE,
    );

    // Resume indexing/watching for resources left active from a previous
    // run -- without this, a daemon restart would silently stop watching
    // everything until each resource was manually re-added.
    for resource in db
        .list_resources()
        .await?
        .into_iter()
        .filter(|resource| resource.status == ResourceStatus::Active)
    {
        if let Ok(root) = uri_to_path(&resource.uri) {
            manager.start(resource.name, root).await?;
        }
    }

    let listener = tokio::net::TcpListener::bind(("0.0.0.0", config.port)).await?;
    let app = server::router(AppState {
        db,
        client,
        manager,
    });

    axum::serve(listener, app).await?;

    Ok(())
}
