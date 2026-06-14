# Contributing to c0

Thanks for your interest! c0 is a knowledge-graph memory system for LLMs, written in Rust.

## Getting set up

```bash
docker compose up -d            # Neo4j
ollama pull nomic-embed-text    # embeddings
cargo build                     # default build
cargo build --features sessions # with the optional Claude Code session adapter
cargo test --all-features
```

You'll need Rust (2024 edition / 1.85+), Docker, and Ollama. An `ANTHROPIC_API_KEY` is optional
(only the reflection classifier and concept extraction use it).

## Before you open a PR

- `cargo fmt --all` — the CI checks formatting.
- `cargo clippy --all-features` — keep it clean. The crate forbids `unsafe` and lints with
  Clippy `pedantic`.
- `cargo build --features sessions` as well as the default build.
- `cargo test --all-features`.
- Update the README if you change commands or behavior.

## Architecture, in one breath

`src/graph.rs` holds the retrieval core (Cypher, BM25/vector/RRF, temporal filters); `src/embeddings.rs`
is the Ollama client + cosine similarity; `src/reflector.rs` is the dead-end → learn loop. Source-specific
integrations (like the optional `sessions` feature) live behind cargo features — that's the pattern for
new adapters.

## Reporting bugs / requesting features

Use the issue templates. For bugs, `c0 health` output and your Neo4j/Ollama versions help a lot.

By contributing, you agree your work is licensed under the project's [MIT License](./LICENSE).
