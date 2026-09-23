//! Binary entry point: parses configuration and serves the HTTP API.

use clap::Parser;
use ragd::config::Config;
use ragd::server;

#[tokio::main]
async fn main() -> ragd::Result<()> {
    let config = Config::parse();

    let listener = tokio::net::TcpListener::bind(("0.0.0.0", config.port)).await?;
    let app = server::router();

    axum::serve(listener, app).await?;

    Ok(())
}
