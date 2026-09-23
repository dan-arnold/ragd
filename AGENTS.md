# Agent/contributor conventions

These are the standing engineering conventions for this repo. Whatever can be
mechanically enforced lives in `rustfmt.toml` / clippy lint attributes in
`src/lib.rs` and is checked in CI; everything else is documented here.

## Workflow: TDD

Write tests before the implementation they cover, for as much of the code as
feasible:

- Write unit tests against a component's intended interface first, then
  implement until they pass.
- Unit tests for a module live in that module via `#[cfg(test)]`.
- Once components interact across modules, add integration tests in
  `tests/` at the crate root — don't wait until the whole thing is built to
  start testing.
- Use `proptest` for invariants that should hold across a range of inputs
  (e.g. "applying the same chunk set twice is a no-op", "chunking is stable
  for unchanged input"), not just hand-picked examples.

## Errors vs. panics

- `Result<T, RagdError>` (see `src/error.rs`, built on `thiserror`) for every
  recoverable failure. Never `.unwrap()`/`.expect()` in non-test code —
  `clippy::unwrap_used` and `clippy::expect_used` are warned on for exactly
  this reason. Test modules are exempt (mark them
  `#[allow(clippy::unwrap_used, clippy::expect_used)]`) since asserting
  preconditions there is the point.
- `panic!` is reserved strictly for programmer errors — broken invariants or
  violated preconditions that indicate a bug, not an expected failure mode.
  Example: dereferencing an `Option` that's structurally guaranteed to be
  `Some` by a prior check, or a function documented to require a non-negative
  index being called with a negative one. A bad HTTP request, a missing
  file, or a failed embed call are *not* panics — those are `Result`s.

## Style

- Prefer `&T` over `T` in function signatures unless ownership is actually
  needed.
- Shared state across tasks: `Arc<T>` for shared ownership,
  `tokio::sync::Mutex`/`RwLock` for interior mutability (this is an async/
  tokio codebase — use the async-aware primitives, not `std::sync`'s, for
  anything held across an `.await`).
- Prefer iterator chains over manual loops.
- `Option<T>` for nullable values.
- Prefer enums over boolean flags for state (e.g. resource indexing state is
  an enum, not a pile of booleans).
- `///` doc comments on items (functions, structs, etc.); `//!` at the top of
  a file for module/crate-level docs. Use `# Examples` blocks where they
  double as doctests (`cargo test` runs them, so they can't silently rot).

## CI

CI runs, and all must pass:

- `cargo fmt --check`
- `cargo clippy --all-targets -- -D warnings`
- `cargo test`
