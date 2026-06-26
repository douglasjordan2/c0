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

Run the exact CI pipeline locally — `scripts/ci.sh` mirrors `.github/workflows/ci.yml`:

```bash
./scripts/ci.sh   # fmt --check, clippy, default + sessions builds, tests
```

Or enable the git hooks once so this runs automatically (fast checks on commit, full pipeline on push):

```bash
git config core.hooksPath .githooks
```

- `pre-commit` runs `cargo fmt --all --check` + `cargo clippy` (seconds).
- `pre-push` runs the full `scripts/ci.sh`.

The rule of thumb: **what's _required_ is what CI checks.** Everything CI enforces is
listed here; anything below this that CI can't run (the live Docker stack) is
*recommended*, not a merge blocker.

What CI enforces:
- `cargo fmt --all` — the CI checks formatting.
- `cargo clippy --all-features` — keep it clean. The crate forbids `unsafe` and lints with
  Clippy `pedantic`.
- `cargo build --features sessions` as well as the default build.
- `cargo test --all-features`.
- **Retrieval gate** — the `eval-gate` job stands up an ephemeral Neo4j and runs
  `c0 eval --no-embeddings` (fulltext-only, no model needed), failing the build if
  `recall@k` regresses. This is the automated guard for the retrieval cascade; see
  [EVAL.md](EVAL.md).
- Update the README if you change commands or behavior.

## Testing against a live stack (Docker)

`scripts/ci.sh` covers fmt/lint/build/unit-tests — but the unit tests never touch
a database. For changes to retrieval, the graph, or the sessions feature, also run
the **live integration harness**. It stands up a throwaway Neo4j + a real Ollama,
seeds a small graph by driving real `c0` commands, and asserts that real commands
return the right results:

```bash
./scripts/docker-test.sh
```

This brings up `docker-compose.test.yml` — isolated from your dev setup (separate
volumes, no published ports), so it won't touch your real Neo4j — builds c0 with
`--features sessions`, runs `scripts/integration-test.sh`, and exits non-zero if any
assertion fails. The first run pulls `nomic-embed-text` (~270 MB, then cached in a
named volume).

What it checks today: graph traversal (`walk`), vector search over **real
embeddings**, BM25 keyword search, a temporal supersession, and session indexing
with embedded turns. The graph and embeddings are real; the non-deterministic chat
LLM (concept extraction / enrichment) is **not required** — set
`C0_ENRICH_CHAT_MODEL=<ollama-model>` to also run live enrichment and assert it links
concepts.

**Run this when you change retrieval _behavior_** — the Cypher, the BM25/vector/RRF
ranking, or the temporal filters in `src/graph.rs` — or the sessions feature. Adding a
read-only command *over* existing retrieval doesn't require it: the automated retrieval
gate (above) already covers that path. CI can't run this live stack (it needs a real
Ollama), so it's the one check that's on you locally rather than enforced server-side.

When you change a command's behavior, add or update an assertion in
`scripts/integration-test.sh`; extend `scripts/seed-test-graph.sh` and
`tests/fixtures/` if you need new seed data.

## Architecture, in one breath

`src/graph.rs` holds the retrieval core (Cypher, BM25/vector/RRF, temporal filters); `src/embeddings.rs`
is the Ollama client + cosine similarity; `src/reflector.rs` is the dead-end → learn loop. Source-specific
integrations (like the optional `sessions` feature) live behind cargo features — that's the pattern for
new adapters.

## Reporting bugs / requesting features

Use the issue templates. For bugs, `c0 health` output and your Neo4j/Ollama versions help a lot.

By contributing, you agree your work is licensed under the project's [MIT License](./LICENSE).
