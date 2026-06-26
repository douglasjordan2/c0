//! `c0 eval` — an offline retrieval-quality harness for the concept-resolution
//! cascade (exact → fulltext → hybrid BM25+vector via RRF, temporal-aware).
//!
//! Where [`crate::bench`] asks *"does the memory layer beat a flat vector store
//! end-to-end?"* (LLM-judged answer accuracy across arms), this harness asks the
//! narrower, intrinsic question the cascade is actually responsible for:
//!
//! > Given a natural-language query, does the **right concept** rank in the top *k*
//! > of what retrieval returns?
//!
//! That isolates the failure mode bench can't see directly: a *bad hit* — the
//! wrong context surfacing with confidence. The metrics are the standard IR pair:
//!
//!   * **recall@k** — did an expected concept land in the top *k*?
//!   * **MRR**      — how highly was the first expected concept ranked?
//!   * **precision@k** (reported, secondary) — fraction of the top *k* that were
//!     expected. With one relevant concept per query this is bounded by `1/k`, so
//!     it is informative as a trend, not an absolute.
//!
//! The fixture is the same synthetic world [`crate::bench`] seeds (invented
//! entities no model has memorised), so `query → expected concept` pairs are
//! versioned alongside it. The metric path is **local-first**: exact + fulltext
//! tiers need only Neo4j, so a CI gate can run with embeddings disabled
//! (`--no-embeddings`) and still catch fulltext/cascade regressions without an
//! Ollama or API dependency. Queries that exercise the vector/temporal tiers are
//! flagged and skipped (counted, never silently dropped) when embeddings are off.
//!
//! An optional `--judge` pass uses the existing LLM client to grade context
//! relevance (faithfulness); it degrades gracefully to a no-op when no provider
//! is configured.

use anyhow::Result;
use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use neo4rs::Graph;
use std::collections::HashSet;

use crate::bench;
use crate::claude::LlmClient;
use crate::config::SemanticConfig;
use crate::embeddings::OllamaClient;
use crate::graph;

const NAMESPACE: &str = bench::NAMESPACE;

// ---------------------------------------------------------------------------
// Golden set — query → expected concept(s), versioned with the bench fixture.
//
// `expected` is a set: any one of the names counts as a hit for recall@k, and
// MRR uses the best-ranked of them. Several queries deliberately accept either
// the directly-named concept or a one-hop neighbour (e.g. "who designed
// Driftwood's storage engine" → "Tomas Reyne" *or* "Driftwood"), because the
// cascade resolving to either is a legitimate entry point for the walk.
// ---------------------------------------------------------------------------

struct Golden {
    id: &'static str,
    query: &'static str,
    /// Acceptable concept names (case-insensitive). Any match is a hit.
    expected: &'static [&'static str],
    /// Point-in-time scope. When set, the query is resolved through the
    /// temporal hybrid tier, which requires embeddings.
    as_of: Option<&'static str>,
}

const GOLDEN: &[Golden] = &[
    Golden {
        id: "g-hq",
        query: "Where is Quorrin Labs headquartered?",
        expected: &["Quorrin Labs", "Halberd"],
        as_of: None,
    },
    Golden {
        id: "g-driftwood-kind",
        query: "columnar time-series database",
        expected: &["Driftwood"],
        as_of: None,
    },
    Golden {
        id: "g-driftwood-author",
        query: "who designed Driftwood's storage engine",
        expected: &["Tomas Reyne", "Driftwood"],
        as_of: None,
    },
    Golden {
        id: "g-zephyr-topology",
        query: "Project Zephyr network topology",
        expected: &["Project Zephyr"],
        as_of: None,
    },
    Golden {
        id: "g-driftwood-license",
        query: "Driftwood software license",
        expected: &["Driftwood"],
        as_of: None,
    },
    Golden {
        id: "g-engineer",
        query: "engineer who works at Quorrin Labs",
        expected: &["Tomas Reyne"],
        as_of: None,
    },
    Golden {
        id: "g-city",
        query: "coastal city in Norvenia",
        expected: &["Halberd"],
        as_of: None,
    },
    Golden {
        id: "g-marlowe",
        query: "Quorrin Labs database project",
        expected: &["Project Marlowe"],
        as_of: None,
    },
    // Temporal: the correct hit is a specific dated tenure node. Resolving these
    // requires the temporal hybrid tier (and therefore embeddings).
    Golden {
        id: "g-lead-2020",
        query: "who leads Project Zephyr",
        expected: &["Zephyr lead: Mira Calden"],
        as_of: Some("2020-06-01"),
    },
    Golden {
        id: "g-lead-2023",
        query: "who leads Project Zephyr",
        expected: &["Zephyr lead: Selka Voss"],
        as_of: Some("2023-07-01"),
    },
    Golden {
        id: "g-lead-2026",
        query: "who leads Project Zephyr",
        expected: &["Zephyr lead: Ade Okonkwo"],
        as_of: Some("2026-06-01"),
    },
];

fn parse_date(s: &str) -> Result<DateTime<Utc>> {
    let d = NaiveDate::parse_from_str(s, "%Y-%m-%d")?
        .and_hms_opt(0, 0, 0)
        .ok_or_else(|| anyhow::anyhow!("bad date"))?;
    Ok(Utc.from_utc_datetime(&d))
}

// ---------------------------------------------------------------------------
// Metrics — pure functions over a ranked candidate list (unit-tested below).
// ---------------------------------------------------------------------------

fn is_expected(name: &str, expected: &[&str]) -> bool {
    expected.iter().any(|e| e.eq_ignore_ascii_case(name))
}

/// 1.0 if any expected name appears in the top `k` of `ranked`, else 0.0.
fn recall_at_k(ranked: &[String], expected: &[&str], k: usize) -> f32 {
    let hit = ranked.iter().take(k).any(|n| is_expected(n, expected));
    if hit { 1.0 } else { 0.0 }
}

/// Reciprocal of the 1-indexed rank of the first expected name; 0.0 if absent.
fn reciprocal_rank(ranked: &[String], expected: &[&str]) -> f32 {
    for (i, n) in ranked.iter().enumerate() {
        if is_expected(n, expected) {
            return 1.0 / (i as f32 + 1.0);
        }
    }
    0.0
}

/// Fraction of the top `k` that are expected.
fn precision_at_k(ranked: &[String], expected: &[&str], k: usize) -> f32 {
    if k == 0 {
        return 0.0;
    }
    let hits = ranked
        .iter()
        .take(k)
        .filter(|n| is_expected(n, expected))
        .count();
    hits as f32 / k as f32
}

// ---------------------------------------------------------------------------
// The cascade under test — produce a ranked candidate list for a query.
// ---------------------------------------------------------------------------

async fn embed(client: &OllamaClient, text: &str) -> Result<Vec<f32>> {
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

fn push_unique(out: &mut Vec<String>, seen: &mut HashSet<String>, name: String) {
    if seen.insert(name.to_lowercase()) {
        out.push(name);
    }
}

/// Mirror the real cascade priority and return the ordered concept names it
/// surfaces: exact (substring) → fulltext BM25 → hybrid RRF, deduped in that
/// order. For point-in-time queries the temporal hybrid tier is used on its own,
/// since the exact/fulltext tiers do not apply validity filtering.
///
/// Returns `Ok(None)` when the query needs embeddings (`as_of`, or the
/// non-temporal vector tier) but none are available — the caller counts it as
/// skipped rather than a miss.
async fn cascade_ranked(
    graph: &Graph,
    embedder: Option<&OllamaClient>,
    query: &str,
    as_of: Option<&str>,
) -> Result<Option<Vec<String>>> {
    let ns = vec![NAMESPACE.to_string()];
    let config = graph::HybridSearchConfig::default();

    // Temporal queries resolve solely through the temporal hybrid tier.
    if let Some(date) = as_of {
        let Some(client) = embedder else {
            return Ok(None);
        };
        let temporal = graph::TemporalQuery {
            as_of: Some(parse_date(date)?),
            include_expired: false,
        };
        let q = embed(client, query).await?;
        let hits = graph::search_hybrid_temporal(graph, query, &q, &ns, &temporal, &config).await?;
        return Ok(Some(hits.into_iter().map(|(name, _)| name).collect()));
    }

    let mut out = Vec::new();
    let mut seen = HashSet::new();

    // 1. exact substring tier
    for name in graph::search_concepts(graph, query, &ns).await? {
        push_unique(&mut out, &mut seen, name);
    }
    // 2. fulltext BM25 tier
    for r in graph::search_concepts_fulltext(graph, query, config.fulltext_limit, &ns).await? {
        push_unique(&mut out, &mut seen, r.name);
    }
    // 3. hybrid RRF tier (vector + fulltext) — needs embeddings
    if let Some(client) = embedder {
        let q = embed(client, query).await?;
        for r in graph::search_hybrid(graph, query, &q, config.vector_limit, &ns, &config).await? {
            push_unique(&mut out, &mut seen, r.name);
        }
    }
    Ok(Some(out))
}

// ---------------------------------------------------------------------------
// Optional LLM-as-judge: grade context relevance of the top hit (opt-in).
// ---------------------------------------------------------------------------

const JUDGE: &str = "You are checking whether a retrieved knowledge-graph concept is \
relevant to a search query.\n\nQuery: {Q}\n\nRetrieved concept: {NAME}\nDescription: {DESC}\n\n\
Answer whether this concept is a relevant, on-topic result for the query. Reason in one \
sentence, then output a final line that is exactly RELEVANT or IRRELEVANT.";

async fn judge_relevance(llm: &LlmClient, graph: &Graph, query: &str, top: &str) -> Result<bool> {
    let ns = vec![NAMESPACE.to_string()];
    let desc = graph::get_concept_description(graph, top, &ns)
        .await?
        .unwrap_or_else(|| "(no description)".to_string());
    let prompt = JUDGE
        .replace("{Q}", query)
        .replace("{NAME}", top)
        .replace("{DESC}", &desc);
    let out = llm.generate(&prompt, None).await?;
    if out.is_error {
        return Err(anyhow::anyhow!("judge llm error: {}", out.result));
    }
    let last = out
        .result
        .trim()
        .lines()
        .last()
        .unwrap_or("")
        .to_uppercase();
    Ok(last.contains("RELEVANT") && !last.contains("IRRELEVANT"))
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

pub struct EvalOpts {
    pub do_seed: bool,
    pub k: usize,
    pub judge: bool,
    /// Force the fulltext-only path (CI / no Ollama). Skips embedding-dependent
    /// queries and counts them, rather than scoring them as misses.
    pub no_embeddings: bool,
    /// Fail (non-zero exit) if aggregate recall@k falls below this threshold.
    pub min_recall: Option<f32>,
}

pub async fn run(graph: &Graph, opts: &EvalOpts) -> Result<()> {
    let mut sem = SemanticConfig::load();
    if opts.no_embeddings {
        sem.enabled = false;
    }

    if opts.do_seed {
        bench::seed(graph, &sem).await?;
        println!();
    }

    let embedder = OllamaClient::from_config(&sem);
    if embedder.is_none() {
        eprintln!(
            "ℹ️  embeddings disabled — running fulltext-only; vector/temporal queries are skipped.\n"
        );
    }
    let llm = opts
        .judge
        .then(|| LlmClient::for_task(&sem, "eval", sem.claude.timeout_secs));

    let k = opts.k.max(1);
    println!("  c0-eval — retrieval quality over the c0-bench fixture (k={k})\n");
    println!(
        "  {:<18} {:>8} {:>6} {:>6}  result",
        "query", "recall@k", "RR", "P@k"
    );
    println!(
        "  {:-<18} {:->8} {:->6} {:->6}  {:-<24}",
        "", "", "", "", ""
    );

    let mut sum_recall = 0.0f32;
    let mut sum_rr = 0.0f32;
    let mut sum_prec = 0.0f32;
    let mut scored = 0u32;
    let mut skipped = 0u32;
    let mut judged_ok = 0u32;
    let mut judged_total = 0u32;

    for g in GOLDEN {
        let ranked = match cascade_ranked(graph, embedder.as_ref(), g.query, g.as_of).await {
            Ok(Some(r)) => r,
            Ok(None) => {
                skipped += 1;
                println!(
                    "  {:<18} {:>8} {:>6} {:>6}  (skipped — needs embeddings)",
                    g.id, "-", "-", "-"
                );
                continue;
            }
            Err(e) => {
                // A transient retrieval failure is a miss for this query, not a
                // fatal abort of the whole run.
                scored += 1;
                println!(
                    "  {:<18} {:>8} {:>6} {:>6}  [error: {e}]",
                    g.id, "0.00", "0.00", "0.00"
                );
                continue;
            }
        };

        let recall = recall_at_k(&ranked, g.expected, k);
        let rr = reciprocal_rank(&ranked, g.expected);
        let prec = precision_at_k(&ranked, g.expected, k);
        sum_recall += recall;
        sum_rr += rr;
        sum_prec += prec;
        scored += 1;

        let top = ranked
            .first()
            .cloned()
            .unwrap_or_else(|| "(none)".to_string());
        let mark = if recall > 0.0 { "✓" } else { "✗" };
        println!(
            "  {:<18} {:>8.2} {:>6.2} {:>6.2}  {mark} top={top}",
            g.id, recall, rr, prec
        );

        if let Some(llm) = &llm {
            if let Some(first) = ranked.first() {
                judged_total += 1;
                match judge_relevance(llm, graph, g.query, first).await {
                    Ok(true) => judged_ok += 1,
                    Ok(false) => {}
                    Err(e) => eprintln!("    judge error [{}]: {e}", g.id),
                }
            }
        }
    }

    let recall = if scored > 0 {
        sum_recall / scored as f32
    } else {
        0.0
    };
    let mrr = if scored > 0 {
        sum_rr / scored as f32
    } else {
        0.0
    };
    let prec = if scored > 0 {
        sum_prec / scored as f32
    } else {
        0.0
    };

    println!("\n  scored {scored} queries ({skipped} skipped)");
    println!("  recall@{k} = {recall:.3}   MRR = {mrr:.3}   precision@{k} = {prec:.3}");
    if judged_total > 0 {
        println!(
            "  context relevance (LLM-judged) = {judged_ok}/{judged_total} ({:.0}%)",
            100.0 * judged_ok as f32 / judged_total as f32
        );
    }

    if let Some(min) = opts.min_recall {
        if recall < min {
            anyhow::bail!("recall@{k} {recall:.3} is below the gate threshold {min:.3}");
        }
        println!("  ✓ gate passed (recall@{k} {recall:.3} ≥ {min:.3})");
    }
    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn recall_hits_within_k_only() {
        let ranked = names(&["A", "B", "Target", "C"]);
        assert_eq!(recall_at_k(&ranked, &["Target"], 3), 1.0);
        assert_eq!(recall_at_k(&ranked, &["Target"], 2), 0.0);
        assert_eq!(recall_at_k(&ranked, &["Missing"], 4), 0.0);
    }

    #[test]
    fn recall_is_case_insensitive() {
        let ranked = names(&["quorrin labs"]);
        assert_eq!(recall_at_k(&ranked, &["Quorrin Labs"], 1), 1.0);
    }

    #[test]
    fn reciprocal_rank_uses_first_hit() {
        let ranked = names(&["A", "Target", "Other"]);
        assert_eq!(reciprocal_rank(&ranked, &["Target"]), 0.5);
        // first of multiple acceptable names wins
        assert_eq!(reciprocal_rank(&ranked, &["Other", "Target"]), 0.5);
        assert_eq!(reciprocal_rank(&names(&["Target"]), &["Target"]), 1.0);
        assert_eq!(reciprocal_rank(&ranked, &["Nope"]), 0.0);
    }

    #[test]
    fn precision_counts_hits_over_k() {
        let ranked = names(&["Target", "B", "Target2", "C"]);
        assert_eq!(precision_at_k(&ranked, &["Target", "Target2"], 4), 0.5);
        assert_eq!(precision_at_k(&ranked, &["Target"], 1), 1.0);
        assert_eq!(precision_at_k(&ranked, &["X"], 0), 0.0);
    }
}
