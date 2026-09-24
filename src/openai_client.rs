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

//! Thin client for OpenAI-compatible embeddings and chat-completions
//! endpoints. Works against OpenAI itself, a local llama-swap/llama.cpp
//! server, or anything else speaking the same wire format — there is
//! deliberately no provider-plugin abstraction.
//!
//! When no API key is configured (the common case for local servers), we
//! omit the `Authorization` header entirely rather than sending `Bearer `
//! with an empty value, which some HTTP clients reject client-side in a
//! way that's easy to mistake for a network problem.

use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::error::{RagdError, Result};

/// Client for a pair of OpenAI-compatible endpoints: one for embeddings,
/// one for chat completions. They're commonly the same server but don't
/// have to be.
pub struct OpenAiClient {
    http: Client,
    embed_endpoint: String,
    embed_api_key: String,
    embed_model: String,
    llm_endpoint: String,
    llm_api_key: String,
    llm_model: String,
}

impl OpenAiClient {
    pub fn new(
        embed_endpoint: impl Into<String>,
        embed_api_key: impl Into<String>,
        embed_model: impl Into<String>,
        llm_endpoint: impl Into<String>,
        llm_api_key: impl Into<String>,
        llm_model: impl Into<String>,
    ) -> Result<Self> {
        let http = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .map_err(|err| RagdError::Config(err.to_string()))?;
        Ok(Self {
            http,
            embed_endpoint: embed_endpoint.into(),
            embed_api_key: embed_api_key.into(),
            embed_model: embed_model.into(),
            llm_endpoint: llm_endpoint.into(),
            llm_api_key: llm_api_key.into(),
            llm_model: llm_model.into(),
        })
    }

    /// Embeds a single piece of text, returning the raw embedding vector.
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        #[derive(Serialize)]
        struct RequestBody<'a> {
            model: &'a str,
            input: &'a str,
        }
        #[derive(Deserialize)]
        struct ResponseBody {
            data: Vec<Datum>,
        }
        #[derive(Deserialize)]
        struct Datum {
            embedding: Vec<f32>,
        }

        let url = format!("{}/embeddings", self.embed_endpoint.trim_end_matches('/'));
        let body = RequestBody {
            model: &self.embed_model,
            input: text,
        };
        let mut parsed: ResponseBody = self.post(&url, &self.embed_api_key, &body).await?;

        let Some(datum) = parsed.data.pop() else {
            return Err(RagdError::ModelEndpoint(
                "embed response contained no data".to_string(),
            ));
        };
        Ok(datum.embedding)
    }

    /// Synthesizes a grounded answer to `user_message` given `system_prompt`
    /// as context (typically the retrieved chunks plus instructions).
    pub async fn chat(&self, system_prompt: &str, user_message: &str) -> Result<String> {
        #[derive(Serialize)]
        struct Message<'a> {
            role: &'a str,
            content: &'a str,
        }
        #[derive(Serialize)]
        struct RequestBody<'a> {
            model: &'a str,
            messages: Vec<Message<'a>>,
        }
        #[derive(Deserialize)]
        struct ResponseBody {
            choices: Vec<Choice>,
        }
        #[derive(Deserialize)]
        struct Choice {
            message: ResponseMessage,
        }
        #[derive(Deserialize)]
        struct ResponseMessage {
            content: String,
        }

        let url = format!(
            "{}/chat/completions",
            self.llm_endpoint.trim_end_matches('/')
        );
        let body = RequestBody {
            model: &self.llm_model,
            messages: vec![
                Message {
                    role: "system",
                    content: system_prompt,
                },
                Message {
                    role: "user",
                    content: user_message,
                },
            ],
        };
        let mut parsed: ResponseBody = self.post(&url, &self.llm_api_key, &body).await?;

        let Some(choice) = parsed.choices.pop() else {
            return Err(RagdError::ModelEndpoint(
                "chat response contained no choices".to_string(),
            ));
        };
        Ok(choice.message.content)
    }

    async fn post<B: Serialize, T: for<'de> Deserialize<'de>>(
        &self,
        url: &str,
        api_key: &str,
        body: &B,
    ) -> Result<T> {
        let mut request = self.http.post(url).json(body);
        if !api_key.is_empty() {
            request = request.bearer_auth(api_key);
        }

        let response = request
            .send()
            .await
            .map_err(|err| RagdError::ModelEndpoint(format!("request to {url} failed: {err}")))?;
        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(RagdError::ModelEndpoint(format!(
                "{url} returned {status}: {text}"
            )));
        }

        response
            .json()
            .await
            .map_err(|err| RagdError::ModelEndpoint(format!("invalid response from {url}: {err}")))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use axum::extract::State;
    use axum::http::{HeaderMap, StatusCode, header};
    use axum::routing::post;
    use axum::{Json, Router};
    use std::sync::Arc;
    use tokio::sync::Mutex;

    #[derive(Clone, Default)]
    struct Captured {
        auth_header: Arc<Mutex<Option<String>>>,
    }

    async fn start_mock(app: Router) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        tokio::spawn(async move {
            axum::serve(listener, app).await.expect("serve");
        });
        format!("http://{addr}")
    }

    async fn capture_auth_and_embed(
        State(captured): State<Captured>,
        headers: HeaderMap,
    ) -> Json<serde_json::Value> {
        let value = headers
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        *captured.auth_header.lock().await = value;
        Json(serde_json::json!({ "data": [{ "embedding": [1.0] }] }))
    }

    fn client(base: &str, embed_api_key: &str) -> OpenAiClient {
        OpenAiClient::new(base, embed_api_key, "test-embed", base, "", "test-llm").expect("client")
    }

    #[tokio::test]
    async fn embed_parses_openai_shaped_response() {
        let app = Router::new().route(
            "/embeddings",
            post(|| async {
                Json(serde_json::json!({ "data": [{ "embedding": [0.1, 0.2, 0.3] }] }))
            }),
        );
        let base = start_mock(app).await;

        let embedding = client(&base, "").embed("hello").await.expect("embed ok");

        assert_eq!(embedding, vec![0.1, 0.2, 0.3]);
    }

    #[tokio::test]
    async fn embed_sends_bearer_auth_header_when_api_key_set() {
        let captured = Captured::default();
        let app = Router::new()
            .route("/embeddings", post(capture_auth_and_embed))
            .with_state(captured.clone());
        let base = start_mock(app).await;

        client(&base, "secret-key")
            .embed("hi")
            .await
            .expect("embed ok");

        let header = captured.auth_header.lock().await.clone();
        assert_eq!(header.as_deref(), Some("Bearer secret-key"));
    }

    #[tokio::test]
    async fn embed_omits_auth_header_when_api_key_empty() {
        let captured = Captured::default();
        let app = Router::new()
            .route("/embeddings", post(capture_auth_and_embed))
            .with_state(captured.clone());
        let base = start_mock(app).await;

        client(&base, "").embed("hi").await.expect("embed ok");

        assert_eq!(*captured.auth_header.lock().await, None);
    }

    #[tokio::test]
    async fn embed_propagates_http_error_status_as_model_endpoint_error() {
        let app = Router::new().route(
            "/embeddings",
            post(|| async { (StatusCode::INTERNAL_SERVER_ERROR, "boom") }),
        );
        let base = start_mock(app).await;

        let err = client(&base, "")
            .embed("hi")
            .await
            .expect_err("should fail");

        assert!(
            matches!(err, RagdError::ModelEndpoint(msg) if msg.contains("500") && msg.contains("boom"))
        );
    }

    #[tokio::test]
    async fn embed_errors_on_empty_data_array() {
        let app = Router::new().route(
            "/embeddings",
            post(|| async { Json(serde_json::json!({ "data": [] })) }),
        );
        let base = start_mock(app).await;

        let err = client(&base, "")
            .embed("hi")
            .await
            .expect_err("should fail");

        assert!(matches!(err, RagdError::ModelEndpoint(_)));
    }

    #[tokio::test]
    async fn chat_returns_completion_text() {
        let app = Router::new().route(
            "/chat/completions",
            post(|| async {
                Json(
                    serde_json::json!({ "choices": [{ "message": { "content": "hello world" } }] }),
                )
            }),
        );
        let base = start_mock(app).await;

        let text = client(&base, "")
            .chat("system prompt", "question")
            .await
            .expect("chat ok");

        assert_eq!(text, "hello world");
    }
}
