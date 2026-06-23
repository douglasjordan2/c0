# c0-bench — does structured memory actually beat a vector store?

`c0 bench` is a small, reproducible benchmark that measures what c0 adds on top
of (a) a bare language model with no memory, and (b) a naive flat vector store —
the obvious "why not just embed everything and retrieve top-k?" baseline.

The headline question it answers with numbers, not assertions:

> A vector store can retrieve facts. Where does it actually fall down, and does
> c0's structure (a temporal graph with corrections) fix it?

## TL;DR result

Local run, LLM-judged, 10 questions across 4 categories, 3 trials per question
(majority vote):

| category       | bare model | flat vector RAG | flat RAG + reranker |     c0     |
|----------------|:----------:|:---------------:|:-------------------:|:----------:|
| simple recall  | 0/3 (0%)   |   3/3 (100%)    |     3/3 (100%)      | 3/3 (100%) |
| multi-hop      | 0/2 (0%)   |    1/2 (50%)    |      0/2 (0%)       | 2/2 (100%) |
| **correction** | 0/2 (0%)   |  **0/2 (0%)**   |    **0/2 (0%)**     | **2/2 (100%)** |
| **temporal**   | 0/3 (0%)   |  **0/3 (0%)**   |    **0/3 (0%)**     | **3/3 (100%)** |
| **overall**    | 0/10 (0%)  |  **4/10 (40%)** |    **3/10 (30%)**   | **10/10 (100%)** |

**The vector arms pass simple recall and score 0% on correction and temporal.**
The clearest case is **temporal**: asked *"who led Project Zephyr in 2020?"* the
flat store retrieves all three (undated) leadership facts and answers *"the
context lists three people…"* — it has no notion of an effective date. c0 stores
each tenure with `valid_at`/`expired_at` and an `as-of` walk returns the one
person who held the role on that date. **Correction** is the same shape: the flat
store holds both the old and new value as equal blobs and reports a
*"contradiction"*; c0 knows (via the `corrects` edge) which value is current.

**Adding an LLM reranker doesn't help.** The `flat RAG + reranker` arm pulls a
wider candidate pool and asks the model to pick the most relevant passages — yet
correction and temporal stay at 0%. Reranking *reorders* passages; it cannot
synthesize an effective date or a supersession signal that isn't in the text.
The gap isn't retrieval quality, it's **representation** — the whole argument for
a temporal graph. (Multi-hop varies by ±1 between the flat arms at N=2; that's
noise, not a reranker effect.)

## Why the facts are invented

Every entity in the benchmark — Quorrin Labs, Project Zephyr, Driftwood — is
fictional. That is deliberate: no language model has seen these facts in
training, so the score can't be contaminated by the model's prior knowledge. A
bare model can only hallucinate or refuse (hence 0% across the board), which
means any points the other arms score are attributable to the **memory layer**,
not the model.

## The arms

| arm           | what it is                                                                    |
|---------------|-------------------------------------------------------------------------------|
| `bare`        | the model alone, no context. Floor.                                           |
| `flat_rag`    | embed every fact as a prose blob, cosine-retrieve top-k, stuff into context. The vector-store baseline. |
| `flat_rerank` | `flat_rag` with a wider candidate pool (top-8) re-ranked by the LLM down to top-4. The "but a real stack uses a reranker" arm. |
| `c0`          | the real c0 retrieval cascade (exact → fulltext → hybrid), temporal- and patch-aware. |

Every arm answers the **same** questions and is graded by the **same** LLM judge.
The vector arms and the c0 arm are given the **same ground truth** — the only
difference is how that knowledge is represented and retrieved.

## The four categories

- **simple recall** — one stored fact. Any memory should pass; the bare model can't.
- **multi-hop** — requires following relationships (`Tomas Reyne → Project Marlowe → Driftwood`). Tests graph traversal vs. flat similarity.
- **correction** — a fact that was later revised. The stale value lives in the
  concept description, the new value in a correction patch, with **no cue words**
  ("obsolete," "outdated") in either. A flat store sees two equally-plausible blobs
  and can't tell which wins; c0 knows the patch supersedes the description.
- **temporal** — time-versioned facts queried with an effective date. This is the
  category a flat vector store structurally cannot represent.

## Running it

```bash
# 1. seed the synthetic world into the `c0-bench` namespace (idempotent)
c0 bench --seed-only

# 2. run all three arms
c0 bench --arms bare,flat_rag,c0

# or seed + run all four arms, 3 trials per question (majority vote)
c0 bench --seed --arms bare,flat_rag,flat_rerank,c0 --trials 3
```

`--trials N` runs each question N times and takes the majority verdict, which
damps the LLM's run-to-run variance; the headline table above used `--trials 3`.

**Requirements:** a running Neo4j (the c0 graph) and an embeddings endpoint
(Ollama). The answer/judge model follows your c0 config — it uses the
`claude` CLI if configured, otherwise a local Ollama model. The run is resilient
to transient LLM/embedding failures (it retries, then records a single failed
cell rather than aborting).

## How c0 retrieval works here

- **recall / correction** — facts are stored as *patches* anchored to concepts; a
  walk surfaces the patch body.
- **multi-hop** — `traverse_temporal` follows outbound relationships to the
  configured depth.
- **temporal** — each leadership tenure is a distinct concept with `valid_at`, and
  `supersede_concept` sets the previous one's `expired_at` to the new start date.
  An `as-of` walk filters to `valid_at <= as_of < expired_at`.

## Limitations / honesty notes

- **Small set (10 questions).** It's an illustrative benchmark, not a leaderboard.
  The categories are designed to isolate capabilities, not to estimate accuracy on
  real workloads.
- **The bare arm is a floor by construction** — with synthetic facts it is expected
  to score 0; it exists to show the questions are unanswerable without memory. (A
  handful of bare cells also hit transient `claude` CLI errors, which are recorded
  as incorrect — immaterial, since bare is 0 either way.)
- **LLM-as-judge** introduces some grading noise; the judge is prompted to grade on
  the key fact only and to treat refusals as incorrect. `--trials N` with majority
  vote is the mitigation; the headline numbers use `--trials 3`.
- **The vector baselines are honest but not exhaustive.** `flat_rag` is plain
  embed → cosine top-k; `flat_rerank` adds an LLM reranker. A production stack
  could also add explicit metadata filters and recency heuristics — but those are
  per-case bolt-ons that re-implement, by hand, the temporal and corrective
  structure c0 represents natively. The reranker arm already shows that improving
  *retrieval ranking* does nothing for correction/temporal, because the gap is
  representation, not ranking.

## Why this exists

c0 is a memory layer for language models. "It helps" is easy to assert and hard
to believe. This makes the claim falsifiable: seed a controlled world, ask
questions a vector store *should* be able to answer, and show precisely where it
can't — and that c0 can.
