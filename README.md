# c0

**An external memory for LLMs** — a bi-temporal knowledge graph with hybrid (keyword + vector) retrieval and a self-improving reflection loop.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Built with Rust](https://img.shields.io/badge/built%20with-Rust-orange.svg)](https://www.rust-lang.org/)

---

## Why

Language models are **stateless between sessions** and their training data **goes stale**. The usual fix — stuffing documents into a vector store — retrieves blobs of prose and has no notion of how knowledge changes over time.

c0 takes a different approach. It stores knowledge as a **graph of concepts and the relationships between them**, retrieves the relevant subgraph on demand, and tracks how each fact evolves. The result is a persistent, *correctable* memory layer you can query in natural language and grow as you work.

## How it works

```
query ──▶ ❶ exact match ─▶ ❷ keyword (BM25) ─▶ ❸ hybrid (BM25 + vector, fused by RRF)
                                                          │
                                                          ▼
                                          resolve to a concept node in Neo4j
                                                          │
                                                          ▼
                                     traverse the graph for related context  ──▶  answer
                                                          │
                                              (no match) ▼
                                          reflection loop: learn from the miss
```

- **Graph storage (Neo4j).** Knowledge lives as `Concept` nodes and typed relationships, not as text chunks — so retrieval can *traverse* from one idea to related ones.
- **Hybrid retrieval.** A tiered cascade: exact match → keyword (Lucene/BM25) → **hybrid**, which runs keyword *and* vector search and merges them with **Reciprocal Rank Fusion (RRF)**. Keyword nails exact names and identifiers; vectors catch synonyms and paraphrase; fusion gets the best of both without normalizing incompatible score scales.
- **Bi-temporal.** Every concept carries two independent timestamps — when it was *recorded* (transaction time) and when it is *true* (valid time) — so you can run point-in-time ("as-of") queries, **supersede** a concept when it evolves, or **invalidate** it with a causal audit trail. Nothing is deleted; it's time-bounded.
- **Self-improving reflection loop.** When a lookup finds nothing, the dead end is queued, and an LLM classifies it: **commit** a genuinely new, reusable concept, **discard** noise, or **queue** the uncertain ones for human review.

## Requirements

- **Rust** (2024 edition — 1.85+)
- **Neo4j 5** — a `docker-compose.yml` is included
- **[Ollama](https://ollama.com/)** for local embeddings (default model: `nomic-embed-text`)
- *(optional)* an **Anthropic API key** — used by the reflection classifier and concept extraction; c0 runs without it (those features degrade gracefully)

## Quickstart

```bash
# 1. Start Neo4j (binds to localhost only)
docker compose up -d

# 2. Pull the embedding model
ollama pull nomic-embed-text

# 3. Build & install
cargo install --path .

# 4. Point c0 at Neo4j (defaults shown; the bundled compose uses no auth)
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER=""        # empty for the bundled docker-compose
export NEO4J_PASSWORD=""
# export ANTHROPIC_API_KEY="sk-..."   # optional, for the reflector/extraction

# 5. Create indexes (vector + fulltext), then a namespace
c0 migrate
c0 init --namespace my-project

# 6. Add knowledge and recall it
c0 add concept "reciprocal rank fusion" -d "Rank-based fusion of multiple result lists; score = weight/(k+rank)."
c0 relate "reciprocal rank fusion" USED_BY "hybrid search"
c0 walk "hybrid search"
```

## Core commands

| Command | What it does |
|---|---|
| `c0 walk <topic>` | Recall: resolve a concept (exact → keyword → hybrid) and traverse for context |
| `c0 walk <topic> --as-of <date>` | Point-in-time recall (bi-temporal) |
| `c0 search <query>` | Hybrid search without traversal (`--vector-only` / `--keyword-only`) |
| `c0 add concept <name> -d "<desc>"` | Add a concept (embedded on write) |
| `c0 add patch <name> --content "<text>"` | Add a knowledge patch that corrects/augments a concept |
| `c0 relate <a> <TYPE> <b>` | Create a typed relationship |
| `c0 supersede <old> --with <new>` | Mark a concept evolved into a newer one |
| `c0 invalidate concept <name> --reason "<why>"` | Retract a concept with a causal trail |
| `c0 describe <concept> "<new desc>"` | Update a description (and re-embed) |
| `c0 reflector ...` | Inspect/process the dead-end → learn loop |
| `c0 health --fix` | Check Neo4j / Ollama / indexes |
| `c0 export` · `c0 audit` · `c0 move` | Maintenance utilities |

Run `c0 --help` for the full set.

## Configuration

c0 reads connection details from the environment, with a per-namespace `.c0/config.toml` for local settings:

| Variable | Default | Purpose |
|---|---|---|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection |
| `NEO4J_USER` / `NEO4J_PASSWORD` | empty | Neo4j auth |
| `ANTHROPIC_API_KEY` | — | Optional; reflector classification & extraction |

Embedding host/model (Ollama) default to `http://localhost:11434` and `nomic-embed-text`, and are configurable.

## Optional: Claude Code session indexing

If you use [Claude Code](https://claude.com/claude-code), an optional feature indexes your session transcripts into the graph so you can semantically search past conversations and jump back into the right one. It's **off by default** (it couples to Claude Code's transcript format); enable it explicitly:

```bash
cargo install --path . --features sessions
```

This is the reference example of c0's **source-adapter** pattern — the same shape any "fill the graph from <source>" integration would take.

## Architecture notes

The retrieval core lives in `src/graph.rs` (Cypher queries, the BM25/vector/RRF functions, temporal filters) and `src/embeddings.rs` (Ollama client + cosine similarity). The reflection loop is in `src/reflector.rs`. Hybrid search defaults — `alpha = 0.4` (keyword vs. vector weight), `k = 60` (the canonical RRF constant), a `0.3` vector threshold — are defined in `HybridSearchConfig`.

## Contributing

Issues and PRs welcome. The codebase forbids `unsafe` and lints with Clippy `pedantic`; please keep new code warning-clean and run `cargo build --features sessions` as well as the default build.

## License

[MIT](./LICENSE)
