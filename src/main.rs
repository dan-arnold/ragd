//! Binary entry point: parses configuration and serves the HTTP API.

use std::sync::Arc;

use clap::Parser;
use ragd::config::Config;
use ragd::db::Database;
use ragd::server;

#[tokio::main]
async fn main() -> ragd::Result<()> {
    let config = Config::parse();

    let db = Database::connect(&config.data_dir, config.embed_dim).await?;
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", config.port)).await?;
    let app = server::router(Arc::new(db));

    axum::serve(listener, app).await?;

    Ok(())
}
