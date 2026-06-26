# c0-eval — is retrieval surfacing the *right* concept?

`c0 eval` is an offline retrieval-quality harness for c0's concept-resolution
cascade (exact → fulltext BM25 → hybrid BM25+vector via RRF, temporal-aware).

It is the sibling of [`c0 bench`](BENCH.md), and they answer different questions:

| | question | metric |
|---|---|---|
| `c0 bench` | does the memory layer beat a flat vector store **end-to-end**? | LLM-judged answer accuracy, across arms |
| `c0 eval`  | given a query, does the **right concept** rank in the top _k_? | recall@k, MRR (intrinsic retrieval) |

Bench can tell you the *answer* was wrong; eval tells you *why* — that retrieval
put the wrong context in front of the model in the first place. That is the
failure mode the issue calls a **bad hit**: the wrong concept surfacing with
confidence. It is invisible to dogfooding and to bench's end-to-end score, and
it is exactly what a hand-tuned cascade regresses on silently.

## What it measures

For each `query → expected concept(s)` pair in the golden set, the harness runs
the **real cascade** and scores the ranked candidate list:

- **recall@k** — did an expected concept land in the top _k_? (the headline)
- **MRR** — reciprocal rank of the first expected concept, averaged. How *highly*
  was it ranked, not just whether it appeared.
- **precision@k** — fraction of the top _k_ that were expected. Reported as a
  secondary trend: with one relevant concept per query it is bounded by `1/k`, so
  it is informative for movement, not as an absolute.

The fixture is the **same synthetic world** [`c0 bench`](BENCH.md) seeds — invented
entities (Quorrin Labs, Project Zephyr, Driftwood) no model has memorised — so the
golden `query → concept` pairs are versioned right alongside the corpus. Re-seed
is idempotent (`--seed`).

## The cascade under test

`eval` mirrors the real resolution priority and returns the ordered concept names
the cascade surfaces, deduped in tier order:

1. **exact** — substring match on concept name (`search_concepts`)
2. **fulltext** — BM25 over the `concept_fulltext` index (`search_concepts_fulltext`)
3. **hybrid** — RRF fusion of fulltext + vector (`search_hybrid`)

Point-in-time queries (`as_of`) resolve solely through the **temporal hybrid**
tier (`search_hybrid_temporal`), since the exact/fulltext tiers do not apply
validity filtering — an as-of query must return the one tenure node valid on that
date, not all three.

## Usage

```bash
c0 eval --seed --k 3            # full cascade incl. vector + temporal (needs Ollama)
c0 eval --seed --no-embeddings  # fulltext-only: Neo4j only, no Ollama/API
c0 eval --judge                 # + opt-in LLM-as-judge context-relevance pass
c0 eval --min-recall 0.8        # gate: non-zero exit if recall@k drops below 0.8
```

`--no-embeddings` forces the fulltext-only path. Queries that need the vector or
temporal tiers are **skipped and counted** (never silently dropped as misses), so
the reported recall is honest about what ran.

`--judge` adds an opt-in LLM-as-judge pass that grades whether the top hit is
on-topic for the query (context relevance / faithfulness). It uses the existing
LLM client and **degrades gracefully to a no-op** when no provider is configured —
consistent with the rest of c0's local-first posture.

## Local-first by design, and the CI gate

The metric path is **local-first**: the exact and fulltext tiers need only Neo4j,
so the gate runs with no model dependency at all. CI (`.github/workflows/ci.yml`,
job `eval-gate`) spins up a `neo4j:5` service, bootstraps the indexes with
`c0 migrate`, and runs:

```bash
c0 eval --seed --no-embeddings --k 3 --min-recall 0.8
```

A regression in the exact/fulltext tiers, the fixture, or the cascade ordering
drops recall below the threshold and **fails the build**. The vector/RRF tier is
covered separately by the `reciprocal_rank_fusion` unit tests and by local full
runs; the eval metric functions (recall@k, MRR, precision@k) are themselves
unit-tested as pure functions, so they run in CI without a database.

## Reference numbers (local)

| path | recall@3 | MRR |
|---|:---:|:---:|
| full cascade (exact → fulltext → hybrid, temporal) | 1.000 | 0.955 |
| fulltext-only (`--no-embeddings`, CI gate path) | 0.875 | 0.812 |

The one fulltext-only miss is a fact stored in **patch content** ("Driftwood's
storage engine was designed by Tomas Reyne") that the concept-name/description
BM25 index can't see — precisely the gap the vector tier closes, which is why the
full path recovers it.

## Non-goals (for now)

- Online / production-signal evaluation.
- Full [RAGAS](https://github.com/explodinggradients/ragas) integration.
- A large golden set — the fixture is deliberately small and readable; grow the
  `GOLDEN` array in `src/eval.rs` as the cascade's behaviour space grows.
