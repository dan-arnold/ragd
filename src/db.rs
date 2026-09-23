//! Storage layer: LanceDB-backed resource and chunk tables, with atomic
//! upsert-and-prune semantics to avoid the duplicate/orphaned-vector bug
//! this daemon replaces (see `AGENTS.md` and the project plan for context).
//!
//! Implemented in a later step; this module is a placeholder so the crate
//! compiles while other pieces of the scaffold land first.
