//! `c0 bench` — a reproducible benchmark for the value c0 adds over a bare model
//! and over a naive vector store.
//!
//! It seeds a small *synthetic* knowledge world (invented entities no model has
//! seen in training) into a dedicated `c0-bench` namespace, then asks the same
//! questions three ways:
//!
//!   * **bare**     — the model alone (no memory). Can only hallucinate or refuse.
//!   * **flat_rag** — naive vector RAG over the same facts as prose blobs. The
//!                    "why not just a vector store?" baseline. Has no notion of
//!                    supersession or effective dates.
//!   * **c0**       — the real c0 retrieval cascade (exact → fulltext → hybrid),
//!                    temporal-aware, patch-aware.
//!
//! Each answer is graded 0/1 by an LLM judge. Results are broken down by
//! category so the interesting signal is visible: all three arms look similar on
//! simple recall and diverge sharply on correction and temporal questions, which
//! only c0 can represent.
//!
//! Because every fact is synthetic, the score is not "model trivia knowledge" —
//! it isolates what the *memory layer* contributes.

use anyhow::Result;
use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use neo4rs::Graph;
use std::collections::BTreeMap;

use crate::claude::LlmClient;
use crate::config::SemanticConfig;
use crate::embeddings::{self, OllamaClient};
use crate::graph;
use neo4rs::query;

const NAMESPACE: &str = "c0-bench";

// ---------------------------------------------------------------------------
// The synthetic knowledge world.
//
// Knowledge content lives in PATCHES (c0's idiomatic unit of fact). Concepts are
// graph anchors carrying a short description + embedding so retrieval can find
// them. Relations enable multi-hop. Temporal facts are distinct dated concept
// nodes chained with supersession.
// ---------------------------------------------------------------------------

/// (name, description) — anchors. Descriptions are deliberately thin; the
/// substantive facts are in PATCHES so the c0 arm and flat-RAG arm see the same
/// ground truth.
const CONCEPTS: &[(&str, &str)] = &[
    ("Quorrin Labs", "A fictional applied-research lab."),
    // Descriptions for corrected concepts carry the ORIGINAL (now stale) value,
    // with NO lexical cue that it is outdated. The current value lives only in a
    // correction patch. A flat store sees both as equal blobs; only c0 knows
    // which one supersedes.
    (
        "Project Zephyr",
        "A Quorrin Labs networking project that uses a mesh network topology.",
    ),
    ("Project Marlowe", "A Quorrin Labs database project."),
    (
        "Driftwood",
        "A columnar time-series database, distributed under the GPL license.",
    ),
    ("Halberd", "A coastal city in Norvenia."),
    ("Tomas Reyne", "An engineer at Quorrin Labs."),
];

/// (from, REL_TYPE, to)
const RELATIONS: &[(&str, &str, &str)] = &[
    ("Project Zephyr", "DEVELOPED_BY", "Quorrin Labs"),
    ("Project Marlowe", "DEVELOPED_BY", "Quorrin Labs"),
    ("Project Marlowe", "PRODUCES", "Driftwood"),
    ("Quorrin Labs", "LOCATED_IN", "Halberd"),
    ("Tomas Reyne", "WORKS_ON", "Project Marlowe"),
];

/// (patch_name, corrects_concept, content, valid_at_or_empty). The patch content
/// is the current truth. Correction patches deliberately contain NO cue words
/// ("obsolete", "outdated", "was relicensed") — they simply state the new value.
/// A flat store that ingested both the concept description and this patch has two
/// equally-plausible blobs and no signal which is current. c0 presents the patch
/// as the authoritative current layer (it knows, via the `corrects` edge and
/// `valid_at`, that the patch supersedes the description).
const PATCHES: &[(&str, &str, &str, &str)] = &[
    // plain facts (simple recall) — attached as patches so `c0 walk` surfaces them
    (
        "quorrin-hq",
        "Quorrin Labs",
        "Quorrin Labs is headquartered in the city of Halberd, in Norvenia.",
        "",
    ),
    (
        "driftwood-author",
        "Driftwood",
        "Driftwood's storage engine was designed by the engineer Tomas Reyne.",
        "",
    ),
    // corrections — a later fact overrides the concept description, no cue words
    (
        "zephyr-topology",
        "Project Zephyr",
        "Project Zephyr uses a star topology with a single coordinator node called 'Anchor'.",
        "2023-05-01",
    ),
    (
        "driftwood-license",
        "Driftwood",
        "Driftwood is distributed under the Apache-2.0 license.",
        "2024-02-01",
    ),
];

/// Bi-temporal leadership of Project Zephyr. Each tenure is a distinct concept
/// node so `valid_at`/`expired_at` can bound it; an `as-of` walk returns whoever
/// held the role on that date. Listed oldest-first. The flat-RAG arm sees these
/// as undated, contradictory prose ("led by X" three times) and so cannot answer
/// an as-of question correctly — which is the point.
///
/// (concept_name, valid_at, [for flat-RAG prose] plain statement)
const TENURES: &[(&str, &str, &str)] = &[
    (
        "Zephyr lead: Mira Calden",
        "2019-06-01",
        "Project Zephyr is led by Mira Calden.",
    ),
    (
        "Zephyr lead: Selka Voss",
        "2022-03-15",
        "Project Zephyr is led by Selka Voss.",
    ),
    (
        "Zephyr lead: Ade Okonkwo",
        "2025-01-10",
        "Project Zephyr is led by Ade Okonkwo.",
    ),
];

fn parse_date(s: &str) -> Result<DateTime<Utc>> {
    let d = NaiveDate::parse_from_str(s, "%Y-%m-%d")?
        .and_hms_opt(0, 0, 0)
        .ok_or_else(|| anyhow::anyhow!("bad date"))?;
    Ok(Utc.from_utc_datetime(&d))
}

// ---------------------------------------------------------------------------
// Seeding
// ---------------------------------------------------------------------------

/// Embed with retries — the configured Ollama endpoint may be a remote
/// (Tailnet) host that drops connections transiently.
async fn embed_retry(client: &OllamaClient, text: &str) -> Result<Vec<f32>> {
    let mut last = None;
    for attempt in 0..3 {
        match client.embed(text).await {
            Ok(v) => return Ok(v),
            Err(e) => last = Some(e),
        }
        tokio::time::sleep(std::time::Duration::from_secs(attempt + 1)).await;
    }
    Err(last.unwrap_or_else(|| anyhow::anyhow!("embed failed")))
}

async fn embed_text(embedder: &Option<OllamaClient>, text: &str) -> Option<Vec<f32>> {
    match embedder {
        Some(c) => embed_retry(c, text).await.ok(),
        None => None,
    }
}

/// Wipe the benchmark namespace so re-seeding exactly reflects the corpus
/// (including changed descriptions, patch content, and temporal state).
async fn reset(graph: &Graph) -> Result<()> {
    graph
        .run(
            query(
                "MATCH (n) WHERE (n:Concept OR n:KnowledgePatch) AND n.namespace = $ns \
                 DETACH DELETE n",
            )
            .param("ns", NAMESPACE),
        )
        .await?;
    Ok(())
}

pub async fn seed(graph: &Graph, sem: &SemanticConfig) -> Result<()> {
    let ns = vec![NAMESPACE.to_string()];
    let embedder = OllamaClient::from_config(sem);
    if embedder.is_none() {
        eprintln!("⚠️  embeddings disabled; c0 retrieval will rely on fulltext only");
    }

    println!("Seeding namespace '{NAMESPACE}' (clean slate)...");
    reset(graph).await?;

    // concepts
    for (name, desc) in CONCEPTS {
        let emb = embed_text(&embedder, &format!("{name}: {desc}")).await;
        graph::add_concept(
            graph,
            name,
            NAMESPACE,
            Some(desc),
            None,
            None,
            emb.as_deref(),
            None,
        )
        .await?;
    }

    // relations
    for (from, rel, to) in RELATIONS {
        graph::relate(graph, from, rel, to, &ns).await?;
    }

    // patches (facts + corrections)
    for (name, corrects, content, valid_at) in PATCHES {
        let valid = if valid_at.is_empty() {
            None
        } else {
            Some(parse_date(valid_at)?)
        };
        graph::add_patch(
            graph,
            name,
            Some(corrects),
            None,
            Some(content),
            NAMESPACE,
            None,
            None,
            valid,
        )
        .await?;
    }

    // temporal tenures: insert each dated concept, link it to Project Zephyr,
    // then expire the previous one as of the new one's start date.
    let mut prev: Option<&str> = None;
    for (name, valid_at, stmt) in TENURES {
        let valid_dt = parse_date(valid_at)?;
        let emb = embed_text(&embedder, name).await;
        // The description holds the actual fact; temporal filtering selects the
        // one tenure node valid on the queried date, so its description directly
        // states the answer.
        graph::add_concept(
            graph,
            name,
            NAMESPACE,
            Some(stmt),
            None,
            None,
            emb.as_deref(),
            Some(valid_dt),
        )
        .await?;
        graph::relate(graph, "Project Zephyr", "LED_BY", name, &ns).await?;
        if let Some(prev_name) = prev {
            // previous tenure expires when this one begins
            graph::supersede_concept(graph, prev_name, name, &ns, Some(valid_dt)).await?;
        }
        prev = Some(name);
    }

    println!(
        "  seeded {} concepts, {} relations, {} patches, {} tenures",
        CONCEPTS.len(),
        RELATIONS.len(),
        PATCHES.len(),
        TENURES.len()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Arm: c0 retrieval (real cascade: exact → fulltext → hybrid), temporal-aware
// ---------------------------------------------------------------------------

async fn c0_context(
    graph: &Graph,
    sem: &SemanticConfig,
    topic: &str,
    depth: u32,
    as_of: Option<&str>,
) -> Result<String> {
    let ns = vec![NAMESPACE.to_string()];
    let temporal = graph::TemporalQuery {
        as_of: as_of.map(parse_date).transpose()?,
        include_expired: false,
    };

    // 1. resolve the topic to a concept (exact, then fulltext, then hybrid)
    let mut concept = topic.to_string();
    let mut patches = graph::get_patches_temporal(graph, &concept, &ns, &temporal).await?;
    let mut connected = graph::traverse_temporal(graph, &concept, depth, &ns, &temporal).await?;
    let mut desc = graph::get_concept_description(graph, &concept, &ns).await?;

    if patches.is_empty() && connected.is_empty() && desc.is_none() {
        let ft = graph::search_concepts_fulltext(graph, topic, 5, &ns).await?;
        if let Some(best) = ft.first() {
            concept = best.name.clone();
        } else if let Some(client) = OllamaClient::from_config(sem)
            && let Ok(q) = embed_retry(&client, topic).await
        {
            let hybrid = graph::HybridSearchConfig::default();
            let hits = graph::search_hybrid_temporal(graph, topic, &q, &ns, &temporal, &hybrid)
                .await
                .unwrap_or_default();
            if let Some((name, _)) = hits.first() {
                concept = name.clone();
            }
        }
        patches = graph::get_patches_temporal(graph, &concept, &ns, &temporal).await?;
        connected = graph::traverse_temporal(graph, &concept, depth, &ns, &temporal).await?;
        desc = graph::get_concept_description(graph, &concept, &ns).await?;
    }

    // 2. assemble the retrievable context: concept + patches + connected nodes
    //    (with their descriptions and patches), all temporal-filtered.
    // c0 knows, via the `corrects` edge and temporal validity, that a patch is
    // the authoritative *current* value — so it presents the patch as overriding
    // the (possibly stale) concept description. This metadata is exactly what a
    // flat vector store lacks.
    let mut out = String::new();
    // When the query is time-scoped, say so — the facts below are exactly those
    // c0 found valid on that date, so the model should answer for that date.
    if let Some(date) = as_of {
        out.push_str(&format!("As of {date}, the following is true:\n"));
    }
    if let Some(d) = &desc {
        out.push_str(&format!("{concept}: {d}\n"));
    }
    if !patches.is_empty() {
        out.push_str("[CURRENT KNOWLEDGE — the following supersedes the description above]\n");
        for p in &patches {
            if let Some(c) = &p.content {
                out.push_str(&format!("- {c}\n"));
            }
        }
    }
    for name in &connected {
        // traverse follows every edge type, including HAS_PATCH; keep only real
        // concepts (those with a description) so patch nodes don't leak in.
        let Some(d) = graph::get_concept_description(graph, name, &ns).await? else {
            continue;
        };
        out.push_str(&format!("{name}: {d}\n"));
        let np = graph::get_patches_temporal(graph, name, &ns, &temporal).await?;
        if !np.is_empty() {
            out.push_str(&format!(
                "[CURRENT KNOWLEDGE for {name} — supersedes the above]\n"
            ));
            for p in &np {
                if let Some(c) = &p.content {
                    out.push_str(&format!("- {c}\n"));
                }
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Arm: flat vector RAG over the same facts as prose
// ---------------------------------------------------------------------------

fn flat_chunks() -> Vec<String> {
    let mut v = Vec::new();
    for (name, desc) in CONCEPTS {
        v.push(format!("{name}: {desc}"));
    }
    for (from, rel, to) in RELATIONS {
        v.push(format!(
            "{from} {} {to}.",
            rel.to_lowercase().replace('_', " ")
        ));
    }
    for (_n, _c, content, _v) in PATCHES {
        v.push(content.to_string());
    }
    // temporal facts WITHOUT dates: a flat store has no effective-date concept,
    // so all three look equally relevant to an as-of question.
    for (_n, _v, stmt) in TENURES {
        v.push(stmt.to_string());
    }
    v
}

struct FlatIndex {
    chunks: Vec<(Vec<f32>, String)>,
}

impl FlatIndex {
    async fn build(client: &OllamaClient) -> Result<Self> {
        let mut chunks = Vec::new();
        for c in flat_chunks() {
            let v = embed_retry(client, &c).await?;
            chunks.push((v, c));
        }
        Ok(Self { chunks })
    }

    /// Top-n chunk texts by cosine similarity to the question.
    async fn candidates(
        &self,
        client: &OllamaClient,
        question: &str,
        n: usize,
    ) -> Result<Vec<String>> {
        let q = embed_retry(client, question).await?;
        let mut scored: Vec<(f32, &String)> = self
            .chunks
            .iter()
            .map(|(v, t)| (embeddings::cosine_similarity(&q, v), t))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored.iter().take(n).map(|(_, t)| (*t).clone()).collect())
    }

    async fn retrieve(&self, client: &OllamaClient, question: &str, k: usize) -> Result<String> {
        Ok(self
            .candidates(client, question, k)
            .await?
            .iter()
            .map(|t| format!("- {t}"))
            .collect::<Vec<_>>()
            .join("\n"))
    }
}

/// LLM reranker — the "but a real vector stack uses a reranker!" arm. Takes a
/// wider candidate pool and asks the model to pick the k most relevant chunks.
/// This reorders blobs by relevance, but it cannot invent an effective date or a
/// supersession signal that isn't in the text — so it does not help on temporal
/// or correction questions. That is exactly the point.
async fn rerank(
    llm: &LlmClient,
    question: &str,
    candidates: &[String],
    k: usize,
) -> Result<String> {
    let listed = candidates
        .iter()
        .enumerate()
        .map(|(i, c)| format!("{}. {c}", i + 1))
        .collect::<Vec<_>>()
        .join("\n");
    let prompt = format!(
        "Rank the numbered passages by how useful they are for answering the question. \
         Reply with ONLY the {k} most useful passage numbers, most useful first, \
         comma-separated (e.g. \"3,1,5\").\n\nQuestion: {question}\n\nPassages:\n{listed}"
    );
    let out = generate_retry(llm, &prompt).await?;
    let mut picked: Vec<String> = out
        .split(|c: char| !c.is_ascii_digit())
        .filter_map(|tok| tok.parse::<usize>().ok())
        .filter(|&i| i >= 1 && i <= candidates.len())
        .map(|i| candidates[i - 1].clone())
        .collect();
    picked.dedup();
    if picked.is_empty() {
        // fall back to the cosine order if the model didn't return usable indices
        picked = candidates.iter().take(k).cloned().collect();
    }
    Ok(picked
        .iter()
        .take(k)
        .map(|t| format!("- {t}"))
        .collect::<Vec<_>>()
        .join("\n"))
}

// ---------------------------------------------------------------------------
// Answering + judging
// ---------------------------------------------------------------------------

const ANSWER_BARE: &str = "Answer the question in one or two sentences. If you do not know, say \"I don't know.\"\n\nQuestion: {Q}";
const ANSWER_CTX: &str = "Answer the question using ONLY the context below. If the context does not contain \
     the answer, say \"I don't know.\"\n\nContext:\n{CTX}\n\nQuestion: {Q}";

/// Generate with retries — the underlying LLM (claude CLI or ollama) can fail
/// transiently (overload, rate limit). A benchmark must not abort on one flake.
async fn generate_retry(llm: &LlmClient, prompt: &str) -> Result<String> {
    let mut last = None;
    for attempt in 0..3 {
        match llm.generate(prompt, None).await {
            Ok(r) if !r.is_error => return Ok(r.result),
            Ok(r) => last = Some(anyhow::anyhow!("llm returned error: {}", r.result)),
            Err(e) => last = Some(e),
        }
        tokio::time::sleep(std::time::Duration::from_secs(2 * (attempt + 1))).await;
    }
    Err(last.unwrap_or_else(|| anyhow::anyhow!("llm failed")))
}

async fn answer(llm: &LlmClient, question: &str, ctx: Option<&str>) -> Result<String> {
    let prompt = match ctx {
        None => ANSWER_BARE.replace("{Q}", question),
        Some(c) => ANSWER_CTX
            .replace(
                "{CTX}",
                if c.trim().is_empty() {
                    "(no results)"
                } else {
                    c
                },
            )
            .replace("{Q}", question),
    };
    generate_retry(llm, &prompt).await
}

const JUDGE: &str = "You are grading a factual answer against the reference answer.\n\n\
Question: {Q}\n\nReference (correct) answer: {GOLD}\n\nCandidate answer: {CAND}\n\n\
Grade ONLY on whether the candidate states the key fact(s) in the reference answer. \
Ignore wording and extra detail. A refusal or \"I don't know\" is INCORRECT. A \
confidently wrong fact is INCORRECT.\n\nReason in one sentence, then output a final \
line that is exactly CORRECT or INCORRECT.";

async fn judge(llm: &LlmClient, question: &str, gold: &str, cand: &str) -> Result<bool> {
    let prompt = JUDGE
        .replace("{Q}", question)
        .replace("{GOLD}", gold)
        .replace("{CAND}", cand);
    let out = generate_retry(llm, &prompt).await?;
    let last = out.trim().lines().last().unwrap_or("").to_uppercase();
    Ok(last.contains("CORRECT") && !last.contains("INCORRECT"))
}

// ---------------------------------------------------------------------------
// Questions
// ---------------------------------------------------------------------------

struct Question {
    id: &'static str,
    category: &'static str,
    question: &'static str,
    gold: &'static str,
    topic: &'static str, // natural phrase the c0 arm resolves from
    depth: u32,
    as_of: Option<&'static str>,
}

const QUESTIONS: &[Question] = &[
    Question {
        id: "recall-1",
        category: "simple_recall",
        question: "In which city is Quorrin Labs headquartered?",
        gold: "Halberd (in Norvenia).",
        topic: "Quorrin Labs headquarters",
        depth: 2,
        as_of: None,
    },
    Question {
        id: "recall-2",
        category: "simple_recall",
        question: "What kind of database is Driftwood?",
        gold: "A columnar time-series database.",
        topic: "Driftwood database",
        depth: 2,
        as_of: None,
    },
    Question {
        id: "recall-3",
        category: "simple_recall",
        question: "Who designed Driftwood's storage engine?",
        gold: "Tomas Reyne.",
        topic: "Driftwood storage engine",
        depth: 2,
        as_of: None,
    },
    Question {
        id: "hop-1",
        category: "multi_hop",
        question: "What database does the project that Tomas Reyne works on produce?",
        gold: "Driftwood (a columnar time-series database).",
        topic: "Tomas Reyne",
        depth: 2,
        as_of: None,
    },
    Question {
        id: "hop-2",
        category: "multi_hop",
        question: "In what city does the engineer who designed Driftwood's storage engine work?",
        gold: "Halberd.",
        topic: "Tomas Reyne",
        depth: 3,
        as_of: None,
    },
    Question {
        id: "correct-1",
        category: "correction",
        question: "What network topology does Project Zephyr currently use?",
        gold: "A star topology with a coordinator node called 'Anchor' (no longer mesh).",
        topic: "Project Zephyr topology",
        depth: 2,
        as_of: None,
    },
    Question {
        id: "correct-2",
        category: "correction",
        question: "Under what license is Driftwood distributed today?",
        gold: "Apache-2.0 (relicensed from GPL in 2024).",
        topic: "Driftwood license",
        depth: 2,
        as_of: None,
    },
    Question {
        id: "temporal-1",
        category: "temporal",
        question: "Who led Project Zephyr in 2020?",
        gold: "Mira Calden.",
        topic: "Project Zephyr",
        depth: 1,
        as_of: Some("2020-06-01"),
    },
    Question {
        id: "temporal-2",
        category: "temporal",
        question: "Who led Project Zephyr in mid-2023?",
        gold: "Selka Voss.",
        topic: "Project Zephyr",
        depth: 1,
        as_of: Some("2023-07-01"),
    },
    Question {
        id: "temporal-3",
        category: "temporal",
        question: "Who leads Project Zephyr now (2026)?",
        gold: "Ade Okonkwo.",
        topic: "Project Zephyr",
        depth: 1,
        as_of: Some("2026-06-01"),
    },
];

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

#[derive(Default, Clone)]
struct Tally {
    correct: u32,
    total: u32,
}

pub async fn run(graph: &Graph, arms: &[String], do_seed: bool, trials: u32) -> Result<()> {
    let sem = SemanticConfig::load();
    let llm = LlmClient::for_task(&sem, "bench", sem.claude.timeout_secs);

    if do_seed {
        seed(graph, &sem).await?;
        println!();
    }

    // build flat index once if either vector arm is requested
    let flat = if arms.iter().any(|a| a == "flat_rag" || a == "flat_rerank") {
        match OllamaClient::from_config(&sem) {
            Some(c) => Some((FlatIndex::build(&c).await?, c)),
            None => {
                eprintln!("⚠️  embeddings disabled; skipping flat_rag/flat_rerank arms");
                None
            }
        }
    } else {
        None
    };

    // tallies[category][arm]
    let mut tallies: BTreeMap<&str, BTreeMap<String, Tally>> = BTreeMap::new();

    println!(
        "provider={}  arms={:?}\n",
        sem.claude.provider_for("bench"),
        arms
    );

    for q in QUESTIONS {
        for arm in arms {
            // Retrieval can fail transiently (remote embedder); record as
            // incorrect and continue rather than aborting the whole run.
            let ctx: Result<Option<String>> = match arm.as_str() {
                "bare" => Ok(None),
                "c0" => c0_context(graph, &sem, q.topic, q.depth, q.as_of)
                    .await
                    .map(Some),
                "flat_rag" => match &flat {
                    Some((idx, client)) => idx.retrieve(client, q.question, 4).await.map(Some),
                    None => continue,
                },
                "flat_rerank" => match &flat {
                    Some((idx, client)) => match idx.candidates(client, q.question, 8).await {
                        Ok(cands) => rerank(&llm, q.question, &cands, 4).await.map(Some),
                        Err(e) => Err(e),
                    },
                    None => continue,
                },
                _ => continue,
            };
            // A persistent LLM/retrieval failure is recorded as incorrect, not
            // fatal, so one flaky call never discards the rest of the run.
            if std::env::var("C0_BENCH_DEBUG").is_ok() {
                if let Ok(Some(c)) = &ctx {
                    eprintln!("\n--- ctx [{} {}] ---\n{c}---", q.id, arm);
                }
            }
            // Vote over `trials` answer+judge passes (context built once) and take
            // the majority verdict — this damps the LLM's run-to-run noise.
            let (ok, votes, summary) = match ctx {
                Err(e) => (false, 0, format!("[retrieval error: {e}]")),
                Ok(ctx) => {
                    let mut yes = 0u32;
                    let mut last = String::new();
                    for _ in 0..trials {
                        match answer(&llm, q.question, ctx.as_deref()).await {
                            Ok(ans) => {
                                last = ans
                                    .trim()
                                    .lines()
                                    .next()
                                    .unwrap_or("")
                                    .chars()
                                    .take(66)
                                    .collect();
                                match judge(&llm, q.question, q.gold, &ans).await {
                                    Ok(true) => yes += 1,
                                    Ok(false) => {}
                                    Err(e) => last = format!("[judge error: {e}]"),
                                }
                            }
                            Err(e) => last = format!("[answer error: {e}]"),
                        }
                    }
                    (yes * 2 > trials, yes, last)
                }
            };

            let t = tallies
                .entry(q.category)
                .or_default()
                .entry(arm.clone())
                .or_default();
            t.total += 1;
            t.correct += ok as u32;

            let vote = if trials > 1 {
                format!("({votes}/{trials}) ")
            } else {
                String::new()
            };
            println!(
                "[{:<10}] {:<9} {}  {vote}{}",
                q.id,
                arm,
                if ok { "✓" } else { "✗" },
                summary
            );
        }
    }

    report(&tallies, arms);
    Ok(())
}

fn report(tallies: &BTreeMap<&str, BTreeMap<String, Tally>>, arms: &[String]) {
    let cats = ["simple_recall", "multi_hop", "correction", "temporal"];
    println!("\n  c0-bench — accuracy by category (LLM-judged)\n");

    print!("  {:<16}", "category");
    for a in arms {
        print!("{:>14}", a);
    }
    println!();
    print!("  {:-<16}", "");
    for _ in arms {
        print!("{:->14}", "");
    }
    println!();

    let mut totals: BTreeMap<&String, Tally> = BTreeMap::new();
    for cat in cats {
        let Some(by_arm) = tallies.get(cat) else {
            continue;
        };
        print!("  {cat:<16}");
        for a in arms {
            let t = by_arm.get(a).cloned().unwrap_or_default();
            let agg = totals.entry(a).or_default();
            agg.correct += t.correct;
            agg.total += t.total;
            let cell = if t.total > 0 {
                format!(
                    "{}/{} ({:.0}%)",
                    t.correct,
                    t.total,
                    100.0 * t.correct as f32 / t.total as f32
                )
            } else {
                "-".to_string()
            };
            print!("{cell:>14}");
        }
        println!();
    }
    print!("  {:-<16}", "");
    for _ in arms {
        print!("{:->14}", "");
    }
    println!();
    print!("  {:<16}", "OVERALL");
    for a in arms {
        let t = totals.get(a).cloned().unwrap_or_default();
        let cell = if t.total > 0 {
            format!(
                "{}/{} ({:.0}%)",
                t.correct,
                t.total,
                100.0 * t.correct as f32 / t.total as f32
            )
        } else {
            "-".to_string()
        };
        print!("{cell:>14}");
    }
    println!("\n");
}
