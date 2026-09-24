# ragd

A local retrieval-augmented generation (RAG) daemon: point it at a
directory, and it indexes the text files in it, keeps that index up to
date as files change, and answers natural-language questions about the
content over a small HTTP API.

`ragd` doesn't run any models itself. It talks to whatever OpenAI-compatible
embeddings and chat-completions endpoints you give it — a hosted provider,
a local [llama.cpp](https://github.com/ggml-org/llama.cpp) server, or
anything else speaking the same wire format — and handles everything
around that: chunking, embedding, storage, live re-indexing, similarity
search, and answer synthesis.

## Why

Editors, CLI tools, and agents increasingly want to ground a model's
answers in a specific codebase or document set instead of relying on
whatever the model happened to be trained on. Doing that yourself means
building a chunker, an embedding pipeline, a vector store, a file watcher
to keep it current, and a retrieval + synthesis step — none of which is
hard in isolation, but all of which is boilerplate you don't want to
maintain per project or per tool.

`ragd` is that piece, factored out into a single daemon with a small HTTP
surface: register a directory, ask it questions, get back an answer plus
the source chunks it was grounded in.

## Features

- **Incremental indexing.** A directory ("resource") gets a full initial
  scan, then a live filesystem watcher keeps it current — edits, creates,
  deletes, and renames are all picked up automatically.
- **Change-aware.** Each chunk's content hash is checked before
  re-embedding, so touching a file (or a spurious filesystem event)
  without actually changing its content never re-hits the embedding API.
- **`.gitignore`-aware**, including nested `.gitignore` files, evaluated
  fresh on every reconciliation.
- **Boundary-aware chunking** for Rust, Python, JavaScript/JSX, TypeScript/
  TSX, and Go via [tree-sitter](https://tree-sitter.github.io/tree-sitter/)
  — chunks align to function/class/impl boundaries rather than splitting
  mid-construct. Everything else falls back to line/character-budget
  windowing.
- **Sensible defaults for what not to index**: binary formats (images,
  archives, compiled artifacts, fonts, ...) and dependency-manager
  lockfiles (`Cargo.lock`, `package-lock.json`, `yarn.lock`, and friends)
  are skipped — the latter are valid text but machine-generated noise with
  no value for retrieval.
- **Local, embedded storage.** Chunks and their embeddings live in
  [LanceDB](https://lancedb.github.io/lancedb/), an embedded vector
  database — no separate database server to run or operate.
- **Provider-agnostic.** Any OpenAI-compatible embeddings/chat endpoint
  works, local or hosted, and the embedding and chat models can be
  entirely different providers.
- **Multiple independent resources** can be tracked by one daemon at once,
  each isolated from the others.

## How it works

1. **Register a resource** — `POST /resources` with a `file://` URI and a
   name. `ragd` walks the directory (respecting `.gitignore`), chunks each
   text file, embeds every chunk, and writes the result to its local
   LanceDB store. A filesystem watcher then keeps this resource in sync
   for as long as `ragd` keeps running.
2. **Ask a question** — `POST /query` with a resource name and a natural-
   language query. `ragd` embeds the query, runs a similarity search
   against that resource's chunks, and feeds the closest matches to the
   configured chat model as context, returning a synthesized answer
   alongside the source chunks it used.

Nothing above requires the caller to know anything about chunking,
embeddings, or vector search — from the outside, it's "add a directory,
then ask it things."

## Installation

Build from source. You'll need:

- A recent stable [Rust toolchain](https://rustup.rs/)
- `protoc` (the Protocol Buffers compiler), used by LanceDB's storage
  layer — e.g. `apt install protobuf-compiler`, `brew install protobuf`

```sh
git clone https://github.com/dan-arnold/ragd.git
cd ragd
cargo build --release
```

The binary ends up at `target/release/ragd`.

## Usage

`ragd` is configured entirely by CLI flags (each also available as an
environment variable):

| Flag                | Env var           | Required | Description                                                                    |
| -------------------- | ------------------ | :------: | -------------------------------------------------------------------------------- |
| `--data-dir`          | `DATA_DIR`          |    ✅    | Directory for persistent storage (the LanceDB database lives here)               |
| `--port`              | `PORT`              |          | HTTP API port (default `20250`)                                                  |
| `--embed-endpoint`    | `EMBED_ENDPOINT`    |    ✅    | OpenAI-compatible embeddings endpoint, e.g. `http://localhost:8080/v1`           |
| `--embed-api-key`     | `EMBED_API_KEY`     |          | API key for the embeddings endpoint (omit or leave empty for local servers)      |
| `--embed-model`       | `EMBED_MODEL`       |    ✅    | Embedding model name                                                             |
| `--embed-dim`         | `EMBED_DIM`         |    ✅    | Dimensionality of the embedding model's output vectors                          |
| `--llm-endpoint`      | `LLM_ENDPOINT`      |    ✅    | OpenAI-compatible chat-completions endpoint                                      |
| `--llm-api-key`       | `LLM_API_KEY`       |          | API key for the chat-completions endpoint (omit or leave empty for local servers)|
| `--llm-model`         | `LLM_MODEL`         |    ✅    | Chat/LLM model name                                                              |

`--embed-dim` is fixed at table-creation time for a given `--data-dir`, so
it must match whatever `--embed-model` actually produces — changing
embedding models later generally means starting from a fresh data
directory.

```sh
ragd \
  --data-dir ~/.local/share/ragd \
  --embed-endpoint http://localhost:8080/v1 \
  --embed-model text-embedding-nomic-embed-text-v1.5 \
  --embed-dim 768 \
  --llm-endpoint http://localhost:8080/v1 \
  --llm-model qwen2.5-coder-32b
```

By default the daemon resumes indexing and watching for any resources
that were registered in a previous run and left active, so a restart
doesn't require re-adding anything.

## HTTP API

### `GET /health`

Liveness check.

```sh
curl http://localhost:20250/health
# {"status":"ok"}
```

### `POST /resources`

Register a directory for indexing. `uri` must be a `file://` URI to a
local directory; `name` is how you'll refer to it in queries and must be
unique. Registering a URI that's already active is a no-op.

```sh
curl -X POST http://localhost:20250/resources \
  -H 'Content-Type: application/json' \
  -d '{"uri": "file:///home/me/projects/myapp/", "name": "myapp"}'
# {"status":"ok","message":"resource `myapp` added"}
```

### `GET /resources`

List every registered resource and its indexing status
(`pending` → `indexing` → `indexed`, or `failed`).

```sh
curl http://localhost:20250/resources
```

```json
{
  "resources": [
    {
      "uri": "file:///home/me/projects/myapp/",
      "name": "myapp",
      "status": "active",
      "indexing_status": "indexed",
      "indexing_status_message": null,
      "created_at": "2026-09-24T18:42:51Z",
      "indexing_started_at": "2026-09-24T18:42:51Z",
      "last_indexed_at": "2026-09-24T18:47:03Z",
      "last_error": null
    }
  ],
  "total_count": 1,
  "status_summary": { "active": 1 }
}
```

### `DELETE /resources/{name}`

Stop watching and indexing a resource. This deactivates it rather than
deleting its indexed chunks outright.

```sh
curl -X DELETE http://localhost:20250/resources/myapp
```

### `POST /query`

Ask a question of a registered resource. `top_k` (default `5`, max `20`)
controls how many source chunks are retrieved and passed to the chat
model as context.

```sh
curl -X POST http://localhost:20250/query \
  -H 'Content-Type: application/json' \
  -d '{"resource": "myapp", "query": "how are database connections managed?", "top_k": 10}'
```

```json
{
  "answer": "Database connections are managed through a single, long-lived...",
  "sources": [
    {
      "path": "/home/me/projects/myapp/src/db.rs",
      "content": "pub struct Database { ... }",
      "score": 0.87
    }
  ]
}
```

If nothing in the index is relevant to the query, `answer` says so
explicitly rather than letting the model guess, and `sources` is empty.

## Contributing

Engineering conventions (TDD workflow, error-handling rules, style) are
documented in [`AGENTS.md`](AGENTS.md). CI runs `cargo fmt --check`,
`cargo clippy --all-targets -- -D warnings`, and `cargo test`; all three
need to pass.

## License

GPL-3.0-or-later. See [`COPYING`](COPYING) for the full license text.
