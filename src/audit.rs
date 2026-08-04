use anyhow::Result;
use neo4rs::{Graph, query};
use serde::Serialize;

use crate::embeddings::cosine_similarity;

#[derive(Debug, Clone, Serialize)]
pub struct StalenessCandidate {
    pub name: String,
    pub namespace: String,
    pub last_updated: Option<String>,
    pub age_days: Option<i64>,
    pub incoming_count: i64,
    pub outgoing_count: i64,
    pub similar_to: Option<String>,
    pub similarity: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StalenessReport {
    pub namespace: String,
    pub threshold_days: u32,
    pub age_stale: Vec<StalenessCandidate>,
    pub orphaned: Vec<StalenessCandidate>,
    pub supersession_candidates: Vec<StalenessCandidate>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NamespaceIssue {
    pub concept_name: String,
    pub current_namespace: String,
    pub suggested_namespace: Option<String>,
    pub issue_type: String,
    pub same_ns_relationships: i64,
    pub other_ns_relationships: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct NamespaceStats {
    pub namespace: String,
    pub concept_count: i64,
    pub patch_count: i64,
    pub orphaned_count: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct NamespaceReport {
    pub global_refugees: Vec<NamespaceIssue>,
    pub prefix_mismatches: Vec<NamespaceIssue>,
    pub cross_namespace: Vec<NamespaceIssue>,
    pub stats: Vec<NamespaceStats>,
}

async fn find_age_stale(
    graph: &Graph,
    namespaces: &[String],
    days: u32,
) -> Result<Vec<StalenessCandidate>> {
    let q = query(
        r"
        MATCH (c:Concept)
        WHERE c.namespace IN $namespaces
          AND c.invalid_at IS NULL
          AND c.expired_at IS NULL
        OPTIONAL MATCH (c)-[:HAS_PATCH]->(p:KnowledgePatch)
        WHERE p.invalid_at IS NULL
        WITH c, max(p.valid_at) AS max_patch_valid
        WITH c, CASE
            WHEN max_patch_valid IS NOT NULL AND c.updated_at IS NOT NULL AND max_patch_valid > c.updated_at THEN max_patch_valid
            ELSE COALESCE(c.updated_at, max_patch_valid, c.valid_at, c.created_at, datetime())
        END AS last_update
        WHERE last_update < datetime() - duration({days: $days})
        OPTIONAL MATCH (c)<-[in_rel]-()
        OPTIONAL MATCH (c)-[out_rel]->()
        RETURN c.name AS name, c.namespace AS namespace,
               toString(last_update) AS last_updated,
               duration.inDays(last_update, datetime()).days AS age_days,
               count(DISTINCT in_rel) AS incoming_count,
               count(DISTINCT out_rel) AS outgoing_count
        ORDER BY age_days DESC
        LIMIT 50
        ",
    )
    .param("namespaces", namespaces.to_vec())
    .param("days", i64::from(days));

    let mut result = graph.execute(q).await?;
    let mut candidates = Vec::new();

    while let Some(row) = result.next().await? {
        candidates.push(StalenessCandidate {
            name: row.get::<String>("name").unwrap_or_default(),
            namespace: row.get::<String>("namespace").unwrap_or_default(),
            last_updated: row.get::<String>("last_updated").ok(),
            age_days: row.get::<i64>("age_days").ok(),
            incoming_count: row.get::<i64>("incoming_count").unwrap_or(0),
            outgoing_count: row.get::<i64>("outgoing_count").unwrap_or(0),
            similar_to: None,
            similarity: None,
        });
    }

    Ok(candidates)
}

async fn find_orphaned(graph: &Graph, namespaces: &[String]) -> Result<Vec<StalenessCandidate>> {
    let q = query(
        r"
        MATCH (c:Concept)
        WHERE c.namespace IN $namespaces
          AND c.invalid_at IS NULL
        OPTIONAL MATCH (c)<-[in_rel]-()
        WITH c, count(in_rel) AS incoming
        WHERE incoming = 0
        OPTIONAL MATCH (c)-[out_rel]->()
        RETURN c.name AS name, c.namespace AS namespace,
               toString(COALESCE(c.updated_at, c.valid_at, c.created_at)) AS last_updated,
               0 AS incoming_count,
               count(DISTINCT out_rel) AS outgoing_count
        ORDER BY c.name
        LIMIT 50
        ",
    )
    .param("namespaces", namespaces.to_vec());

    let mut result = graph.execute(q).await?;
    let mut candidates = Vec::new();

    while let Some(row) = result.next().await? {
        candidates.push(StalenessCandidate {
            name: row.get::<String>("name").unwrap_or_default(),
            namespace: row.get::<String>("namespace").unwrap_or_default(),
            last_updated: row.get::<String>("last_updated").ok(),
            age_days: None,
            incoming_count: row.get::<i64>("incoming_count").unwrap_or(0),
            outgoing_count: row.get::<i64>("outgoing_count").unwrap_or(0),
            similar_to: None,
            similarity: None,
        });
    }

    Ok(candidates)
}

#[derive(Debug)]
struct ConceptWithEmbedding {
    name: String,
    namespace: String,
    valid_at: Option<String>,
    embedding: Vec<f32>,
    outgoing_rel_types: std::collections::HashSet<String>,
}

fn jaccard(a: &std::collections::HashSet<String>, b: &std::collections::HashSet<String>) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let intersection = a.intersection(b).count() as f32;
    let union = a.union(b).count() as f32;
    if union == 0.0 {
        0.0
    } else {
        intersection / union
    }
}

async fn find_supersession_candidates(
    graph: &Graph,
    namespaces: &[String],
) -> Result<Vec<StalenessCandidate>> {
    let q = query(
        r"
        MATCH (c:Concept)
        WHERE c.namespace IN $namespaces
          AND c.invalid_at IS NULL
          AND c.expired_at IS NULL
          AND c.embedding IS NOT NULL
        OPTIONAL MATCH (c)-[r]->()
        WITH c, collect(DISTINCT type(r)) AS rel_types
        RETURN c.name AS name, c.namespace AS namespace,
               toString(c.valid_at) AS valid_at,
               c.embedding AS embedding,
               rel_types AS rel_types
        ORDER BY c.valid_at
        ",
    )
    .param("namespaces", namespaces.to_vec());

    let mut result = graph.execute(q).await?;
    let mut concepts: Vec<ConceptWithEmbedding> = Vec::new();

    while let Some(row) = result.next().await? {
        if let Ok(embedding) = row.get::<Vec<f64>>("embedding") {
            let rel_types: std::collections::HashSet<String> = row
                .get::<Vec<String>>("rel_types")
                .unwrap_or_default()
                .into_iter()
                .collect();
            concepts.push(ConceptWithEmbedding {
                name: row.get::<String>("name").unwrap_or_default(),
                namespace: row.get::<String>("namespace").unwrap_or_default(),
                valid_at: row.get::<String>("valid_at").ok(),
                embedding: embedding.iter().map(|x| *x as f32).collect(),
                outgoing_rel_types: rel_types,
            });
        }
    }

    let mut candidates = Vec::new();
    let similarity_threshold = 0.90;
    let role_overlap_min = 0.30;

    for i in 0..concepts.len() {
        for j in (i + 1)..concepts.len() {
            let older = &concepts[i];
            let newer = &concepts[j];

            let sim = cosine_similarity(&older.embedding, &newer.embedding);
            if sim <= similarity_threshold {
                continue;
            }

            let role_overlap = jaccard(&older.outgoing_rel_types, &newer.outgoing_rel_types);
            if role_overlap < role_overlap_min {
                continue;
            }

            candidates.push(StalenessCandidate {
                name: older.name.clone(),
                namespace: older.namespace.clone(),
                last_updated: older.valid_at.clone(),
                age_days: None,
                incoming_count: 0,
                outgoing_count: 0,
                similar_to: Some(newer.name.clone()),
                similarity: Some(sim),
            });
        }
    }

    candidates.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    candidates.truncate(20);

    Ok(candidates)
}

pub async fn staleness(
    graph: &Graph,
    namespace: &str,
    namespaces: &[String],
    days: u32,
    json: bool,
) -> Result<()> {
    let ns_filter: Vec<String> = if namespace == "global" {
        namespaces.to_vec()
    } else {
        vec![namespace.to_string()]
    };

    let age_stale = find_age_stale(graph, &ns_filter, days).await?;
    let orphaned = find_orphaned(graph, &ns_filter).await?;
    let supersession_candidates = find_supersession_candidates(graph, &ns_filter).await?;

    let report = StalenessReport {
        namespace: namespace.to_string(),
        threshold_days: days,
        age_stale,
        orphaned,
        supersession_candidates,
    };

    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    println!("C0 Staleness Audit");
    println!("═══════════════════════════════════════");
    println!("Namespace: {namespace} (threshold: {days} days)");

    println!();
    if report.age_stale.is_empty() {
        println!("⏰ STALE CONCEPTS: none");
    } else {
        println!("⏰ STALE CONCEPTS ({} found):", report.age_stale.len());
        for c in &report.age_stale {
            let age_str = c
                .age_days
                .map_or_else(|| "unknown age".to_string(), |d| format!("{d} days old"));
            println!("  {} [{}] - {}", c.name, c.namespace, age_str);
        }
    }

    println!();
    if report.orphaned.is_empty() {
        println!("👻 ORPHANED: none");
    } else {
        println!("👻 ORPHANED ({} found):", report.orphaned.len());
        for c in &report.orphaned {
            println!(
                "  {} [{}] - {} incoming, {} outgoing",
                c.name, c.namespace, c.incoming_count, c.outgoing_count
            );
        }
    }

    println!();
    if report.supersession_candidates.is_empty() {
        println!("🔀 SUPERSESSION CANDIDATES: none");
    } else {
        println!(
            "🔀 SUPERSESSION CANDIDATES ({} pairs):",
            report.supersession_candidates.len()
        );
        for c in &report.supersession_candidates {
            if let (Some(similar), Some(sim)) = (&c.similar_to, c.similarity) {
                println!(
                    "  {} may supersede {} ({:.0}% similar)",
                    similar,
                    c.name,
                    sim * 100.0
                );
            }
        }
    }

    println!();
    println!("═══════════════════════════════════════");
    let total =
        report.age_stale.len() + report.orphaned.len() + report.supersession_candidates.len();
    println!("Total candidates for review: {total}");

    Ok(())
}

#[derive(Debug, Clone, Serialize)]
pub struct EnrichEdge {
    pub from: String,
    pub from_namespace: String,
    pub to: String,
    pub to_namespace: String,
    pub score: f32,
    pub kind: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct EnrichReport {
    pub namespace: String,
    pub run_id: String,
    pub dry_run: bool,
    pub orphaned_before: i64,
    pub orphaned_after: i64,
    pub edges: Vec<EnrichEdge>,
}

async fn count_orphaned(graph: &Graph, namespace: &str) -> Result<i64> {
    let q = query(
        r"
        MATCH (c:Concept {namespace: $namespace})
        WHERE c.invalid_at IS NULL AND c.expired_at IS NULL
          AND NOT (c)--(:Concept)
        RETURN count(c) AS n
        ",
    )
    .param("namespace", namespace.to_string());

    let mut result = graph.execute(q).await?;
    match result.next().await? {
        Some(row) => Ok(row.get::<i64>("n").unwrap_or(0)),
        None => Ok(0),
    }
}

/// Same-namespace proposals: each orphan's nearest in-namespace neighbours
/// at or above `threshold`, capped at `max_links` per orphan.
async fn propose_same_namespace(
    graph: &Graph,
    namespace: &str,
    threshold: f32,
    max_links: usize,
) -> Result<Vec<EnrichEdge>> {
    // `max_links` is interpolated into the slice bound rather than passed as a
    // param; it is a trusted `usize` from a CLI arg, never user-controlled input.
    let cypher = format!(
        "MATCH (q:Concept {{namespace: $namespace}})
         WHERE q.invalid_at IS NULL AND q.expired_at IS NULL
           AND q.embedding IS NOT NULL AND NOT (q)--(:Concept)
         CALL db.index.vector.queryNodes('concept_embedding', 8, q.embedding)
           YIELD node, score
         WHERE node.name <> q.name AND score >= $threshold
           AND node.invalid_at IS NULL AND node.expired_at IS NULL
           AND node.namespace = q.namespace
         WITH q, node, score ORDER BY score DESC
         WITH q, collect({{n: node.name, s: score}})[0..{max_links}] AS top
         UNWIND top AS t
         RETURN q.name AS from, t.n AS to, t.s AS score"
    );

    let mut result = graph
        .execute(
            query(&cypher)
                .param("namespace", namespace.to_string())
                .param("threshold", f64::from(threshold)),
        )
        .await?;

    let mut edges = Vec::new();
    while let Some(row) = result.next().await? {
        edges.push(EnrichEdge {
            from: row.get::<String>("from").unwrap_or_default(),
            from_namespace: namespace.to_string(),
            to: row.get::<String>("to").unwrap_or_default(),
            to_namespace: namespace.to_string(),
            score: row.get::<f64>("score").unwrap_or(0.0) as f32,
            kind: "same".to_string(),
        });
    }
    Ok(edges)
}

/// Cross-namespace bridges: a single highest-scoring neighbour in another
/// namespace at or above `threshold`, for orphans with no same-namespace match.
async fn propose_cross_namespace(
    graph: &Graph,
    namespace: &str,
    threshold: f32,
) -> Result<Vec<EnrichEdge>> {
    let q = query(
        r"
        MATCH (q:Concept {namespace: $namespace})
        WHERE q.invalid_at IS NULL AND q.expired_at IS NULL
          AND q.embedding IS NOT NULL AND NOT (q)--(:Concept)
        CALL db.index.vector.queryNodes('concept_embedding', 8, q.embedding)
          YIELD node, score
        WHERE node.name <> q.name AND score >= $threshold
          AND node.invalid_at IS NULL AND node.expired_at IS NULL
          AND node.namespace <> q.namespace
        WITH q, node, score ORDER BY score DESC
        WITH q, collect({n: node.name, ns: node.namespace, s: score})[0] AS best
        RETURN q.name AS from, best.n AS to, best.ns AS to_namespace, best.s AS score
        ",
    )
    .param("namespace", namespace.to_string())
    .param("threshold", f64::from(threshold));

    let mut result = graph.execute(q).await?;
    let mut edges = Vec::new();
    while let Some(row) = result.next().await? {
        edges.push(EnrichEdge {
            from: row.get::<String>("from").unwrap_or_default(),
            from_namespace: namespace.to_string(),
            to: row.get::<String>("to").unwrap_or_default(),
            to_namespace: row.get::<String>("to_namespace").unwrap_or_default(),
            score: row.get::<f64>("score").unwrap_or(0.0) as f32,
            kind: "cross".to_string(),
        });
    }
    Ok(edges)
}

/// Create one tagged RELATED_TO edge. Returns true if a new edge was written.
async fn apply_edge(graph: &Graph, edge: &EnrichEdge, run_id: &str) -> Result<bool> {
    let q = query(
        r"
        MATCH (a:Concept {name: $from, namespace: $from_ns}),
              (b:Concept {name: $to, namespace: $to_ns})
        MERGE (a)-[r:RELATED_TO]->(b)
        ON CREATE SET r.auto_enriched = true, r.enrich_score = $score, r.enrich_run = $run
        RETURN r.enrich_run = $run AS created
        ",
    )
    .param("from", edge.from.clone())
    .param("from_ns", edge.from_namespace.clone())
    .param("to", edge.to.clone())
    .param("to_ns", edge.to_namespace.clone())
    .param("score", f64::from(edge.score))
    .param("run", run_id.to_string());

    let mut result = graph.execute(q).await?;
    match result.next().await? {
        Some(row) => Ok(row.get::<bool>("created").unwrap_or(false)),
        None => Ok(false),
    }
}

/// Every namespace that currently holds at least one live knowledge-orphaned
/// concept. Backs `--graph-wide`, which sweeps the whole graph instead of just
/// the caller's context-known namespaces. Concepts with a NULL namespace are
/// skipped: the per-namespace MATCH can't target them, same as the other paths.
pub async fn orphan_namespaces(graph: &Graph) -> Result<Vec<String>> {
    let q = query(
        r"
        MATCH (c:Concept)
        WHERE c.invalid_at IS NULL AND c.expired_at IS NULL
          AND c.namespace IS NOT NULL AND NOT (c)--(:Concept)
        RETURN DISTINCT c.namespace AS ns
        ORDER BY ns
        ",
    );

    let mut result = graph.execute(q).await?;
    let mut namespaces = Vec::new();
    while let Some(row) = result.next().await? {
        if let Ok(ns) = row.get::<String>("ns") {
            namespaces.push(ns);
        }
    }
    Ok(namespaces)
}

pub async fn enrich(
    graph: &Graph,
    targets: &[String],
    same_threshold: f32,
    cross_threshold: f32,
    max_links: usize,
    dry_run: bool,
    json: bool,
) -> Result<()> {
    // Timestamp for readability + PID so two invocations in the same second
    // (e.g. a scripted loop) can't share a run-id and clobber each other's rollback.
    let run_id = format!(
        "enrich-{}-{}",
        chrono::Utc::now().format("%Y%m%d-%H%M%S"),
        std::process::id()
    );
    let mut reports: Vec<EnrichReport> = Vec::new();

    for namespace in targets {
        let before = count_orphaned(graph, namespace).await?;

        let same = propose_same_namespace(graph, namespace, same_threshold, max_links).await?;
        // A cross-namespace bridge is a fallback only for orphans with no
        // in-namespace match, mirroring the original enrichment script.
        let covered: std::collections::HashSet<String> =
            same.iter().map(|e| e.from.clone()).collect();
        let cross: Vec<EnrichEdge> = propose_cross_namespace(graph, namespace, cross_threshold)
            .await?
            .into_iter()
            .filter(|e| !covered.contains(&e.from))
            .collect();

        let mut edges: Vec<EnrichEdge> = same;
        edges.extend(cross);

        if !dry_run {
            let mut written = Vec::with_capacity(edges.len());
            for edge in edges {
                if apply_edge(graph, &edge, &run_id).await? {
                    written.push(edge);
                }
            }
            edges = written;
        }

        let after = if dry_run {
            before
        } else {
            count_orphaned(graph, namespace).await?
        };

        reports.push(EnrichReport {
            namespace: namespace.clone(),
            run_id: run_id.clone(),
            dry_run,
            orphaned_before: before,
            orphaned_after: after,
            edges,
        });
    }

    if json {
        println!("{}", serde_json::to_string_pretty(&reports)?);
        return Ok(());
    }

    println!(
        "C0 Orphan Enrichment{}",
        if dry_run { " (dry run)" } else { "" }
    );
    println!("═══════════════════════════════════════");
    let mut total_before = 0;
    let mut total_after = 0;
    let mut total_edges = 0;
    for r in &reports {
        let same_n = r.edges.iter().filter(|e| e.kind == "same").count();
        let cross_n = r.edges.iter().filter(|e| e.kind == "cross").count();
        total_before += r.orphaned_before;
        total_after += r.orphaned_after;
        total_edges += r.edges.len();
        if dry_run {
            println!(
                "  {} - {} orphaned, would add {} edges ({} same-ns, {} cross-ns)",
                r.namespace,
                r.orphaned_before,
                r.edges.len(),
                same_n,
                cross_n
            );
        } else {
            println!(
                "  {} - {} → {} orphaned, +{} edges ({} same-ns, {} cross-ns)",
                r.namespace,
                r.orphaned_before,
                r.orphaned_after,
                r.edges.len(),
                same_n,
                cross_n
            );
        }
    }

    println!();
    println!("═══════════════════════════════════════");
    if dry_run {
        println!("Total: {total_before} orphaned, would add {total_edges} edges");
        println!("Re-run without --dry-run to apply.");
    } else {
        println!(
            "Total orphaned: {total_before} → {total_after}  (+{total_edges} edges, run-id: {run_id})"
        );
        println!("Rollback: c0 audit enrich --rollback {run_id}");
    }

    Ok(())
}

pub async fn enrich_rollback(graph: &Graph, run: Option<&str>, json: bool) -> Result<()> {
    let run_id = match run {
        Some(r) => r.to_string(),
        None => {
            let q = query(
                r"
                MATCH ()-[r:RELATED_TO {auto_enriched: true}]->()
                WHERE r.enrich_run IS NOT NULL
                RETURN r.enrich_run AS run
                ORDER BY run DESC LIMIT 1
                ",
            );
            let mut result = graph.execute(q).await?;
            match result.next().await? {
                Some(row) => row.get::<String>("run").unwrap_or_default(),
                None => {
                    if json {
                        println!("{}", serde_json::json!({"deleted": 0, "run_id": null}));
                    } else {
                        println!("No auto-enriched edges found to roll back.");
                    }
                    return Ok(());
                }
            }
        }
    };

    let q = query(
        r"
        MATCH ()-[r:RELATED_TO {auto_enriched: true, enrich_run: $run}]->()
        WITH collect(r) AS rels
        WITH rels, size(rels) AS n
        FOREACH (r IN rels | DELETE r)
        RETURN n
        ",
    )
    .param("run", run_id.clone());

    let mut result = graph.execute(q).await?;
    let deleted = match result.next().await? {
        Some(row) => row.get::<i64>("n").unwrap_or(0),
        None => 0,
    };

    if json {
        println!(
            "{}",
            serde_json::json!({"deleted": deleted, "run_id": run_id})
        );
    } else {
        println!("Rolled back {deleted} edges from run {run_id}.");
    }
    Ok(())
}

async fn find_global_refugees(graph: &Graph, namespaces: &[String]) -> Result<Vec<NamespaceIssue>> {
    let known_prefixes: Vec<&str> = namespaces
        .iter()
        .filter(|ns| *ns != "global")
        .map(std::string::String::as_str)
        .collect();

    if known_prefixes.is_empty() {
        return Ok(Vec::new());
    }

    let pattern = format!(
        "(?i)^({})[-_].*",
        known_prefixes
            .iter()
            .map(|s| regex::escape(s))
            .collect::<Vec<_>>()
            .join("|")
    );

    let q = query(
        r"
        MATCH (c:Concept {namespace: 'global'})
        WHERE c.name =~ $pattern
        RETURN c.name AS name
        ORDER BY c.name
        LIMIT 50
        ",
    )
    .param("pattern", pattern.clone());

    let mut result = graph.execute(q).await?;
    let mut issues = Vec::new();

    while let Some(row) = result.next().await? {
        let name: String = row.get("name").unwrap_or_default();
        let suggested = known_prefixes.iter().find(|prefix| {
            name.to_lowercase()
                .starts_with(&format!("{}-", prefix.to_lowercase()))
                || name
                    .to_lowercase()
                    .starts_with(&format!("{}_", prefix.to_lowercase()))
        });

        issues.push(NamespaceIssue {
            concept_name: name,
            current_namespace: "global".to_string(),
            suggested_namespace: suggested.map(std::string::ToString::to_string),
            issue_type: "global_refugee".to_string(),
            same_ns_relationships: 0,
            other_ns_relationships: 0,
        });
    }

    Ok(issues)
}

async fn find_prefix_mismatches(
    graph: &Graph,
    namespaces: &[String],
) -> Result<Vec<NamespaceIssue>> {
    let q = query(
        r"
        MATCH (c:Concept)
        WHERE c.namespace IN $namespaces
          AND c.namespace <> 'global'
        WITH c, split(c.name, '-')[0] AS prefix
        WHERE prefix <> c.namespace
          AND prefix IN $namespaces
        RETURN c.name AS name, c.namespace AS current_namespace,
               prefix AS suggested_namespace
        ORDER BY suggested_namespace, c.name
        LIMIT 50
        ",
    )
    .param("namespaces", namespaces.to_vec());

    let mut result = graph.execute(q).await?;
    let mut issues = Vec::new();

    while let Some(row) = result.next().await? {
        issues.push(NamespaceIssue {
            concept_name: row.get::<String>("name").unwrap_or_default(),
            current_namespace: row.get::<String>("current_namespace").unwrap_or_default(),
            suggested_namespace: row.get::<String>("suggested_namespace").ok(),
            issue_type: "prefix_mismatch".to_string(),
            same_ns_relationships: 0,
            other_ns_relationships: 0,
        });
    }

    Ok(issues)
}

async fn find_cross_namespace(graph: &Graph, namespaces: &[String]) -> Result<Vec<NamespaceIssue>> {
    let q = query(
        r"
        MATCH (c:Concept)
        WHERE c.namespace IN $namespaces
        OPTIONAL MATCH (c)-[]-(related:Concept)
        WHERE related.namespace IN $namespaces
        WITH c,
             count(CASE WHEN related.namespace = c.namespace THEN 1 END) AS same_ns,
             count(CASE WHEN related.namespace <> c.namespace THEN 1 END) AS other_ns,
             collect(DISTINCT related.namespace) AS related_namespaces
        WHERE other_ns > same_ns AND other_ns > 2
        RETURN c.name AS name, c.namespace AS current_namespace,
               same_ns, other_ns,
               [ns IN related_namespaces WHERE ns <> c.namespace][0] AS dominant_other
        ORDER BY other_ns DESC
        LIMIT 50
        ",
    )
    .param("namespaces", namespaces.to_vec());

    let mut result = graph.execute(q).await?;
    let mut issues = Vec::new();

    while let Some(row) = result.next().await? {
        issues.push(NamespaceIssue {
            concept_name: row.get::<String>("name").unwrap_or_default(),
            current_namespace: row.get::<String>("current_namespace").unwrap_or_default(),
            suggested_namespace: row.get::<String>("dominant_other").ok(),
            issue_type: "cross_namespace".to_string(),
            same_ns_relationships: row.get::<i64>("same_ns").unwrap_or(0),
            other_ns_relationships: row.get::<i64>("other_ns").unwrap_or(0),
        });
    }

    Ok(issues)
}

async fn get_namespace_stats(graph: &Graph, namespaces: &[String]) -> Result<Vec<NamespaceStats>> {
    let q = query(
        r"
        MATCH (c:Concept)
        WHERE c.namespace IN $namespaces
        OPTIONAL MATCH (c)-[:HAS_PATCH]->(p:KnowledgePatch)
        OPTIONAL MATCH (c)<-[in_rel]-()
        WITH c.namespace AS namespace,
             count(DISTINCT c) AS concepts,
             count(DISTINCT p) AS patches,
             sum(CASE WHEN in_rel IS NULL THEN 1 ELSE 0 END) AS orphaned
        RETURN namespace, concepts AS concept_count,
               patches AS patch_count, orphaned AS orphaned_count
        ORDER BY concepts DESC
        ",
    )
    .param("namespaces", namespaces.to_vec());

    let mut result = graph.execute(q).await?;
    let mut stats = Vec::new();

    while let Some(row) = result.next().await? {
        stats.push(NamespaceStats {
            namespace: row.get::<String>("namespace").unwrap_or_default(),
            concept_count: row.get::<i64>("concept_count").unwrap_or(0),
            patch_count: row.get::<i64>("patch_count").unwrap_or(0),
            orphaned_count: row.get::<i64>("orphaned_count").unwrap_or(0),
        });
    }

    Ok(stats)
}

pub async fn namespaces(
    graph: &Graph,
    namespaces: &[String],
    suggest: bool,
    json: bool,
) -> Result<()> {
    let global_refugees = find_global_refugees(graph, namespaces).await?;
    let prefix_mismatches = find_prefix_mismatches(graph, namespaces).await?;
    let cross_namespace = find_cross_namespace(graph, namespaces).await?;
    let stats = get_namespace_stats(graph, namespaces).await?;

    let report = NamespaceReport {
        global_refugees,
        prefix_mismatches,
        cross_namespace,
        stats,
    };

    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    println!("C0 Namespace Audit");
    println!("═══════════════════════════════════════");

    println!();
    println!("📊 NAMESPACE HEALTH:");
    for s in &report.stats {
        let orphan_pct = if s.concept_count > 0 {
            (s.orphaned_count as f64 / s.concept_count as f64 * 100.0) as u32
        } else {
            0
        };
        let warning = if orphan_pct > 25 { " ⚠️" } else { "" };
        println!(
            "  {}: {} concepts, {} patches ({}% orphaned){}",
            s.namespace, s.concept_count, s.patch_count, orphan_pct, warning
        );
    }

    println!();
    if report.global_refugees.is_empty() {
        println!("📍 GLOBAL REFUGEES: none");
    } else {
        println!(
            "📍 GLOBAL REFUGEES ({} found):",
            report.global_refugees.len()
        );
        for issue in &report.global_refugees {
            let suggestion = if suggest {
                issue
                    .suggested_namespace
                    .as_ref()
                    .map(|s| format!(" → suggested: {s}"))
                    .unwrap_or_default()
            } else {
                String::new()
            };
            println!("  {}{}", issue.concept_name, suggestion);
        }
    }

    println!();
    if report.prefix_mismatches.is_empty() {
        println!("🔀 PREFIX MISMATCHES: none");
    } else {
        println!(
            "🔀 PREFIX MISMATCHES ({} found):",
            report.prefix_mismatches.len()
        );
        for issue in &report.prefix_mismatches {
            let suggestion = issue
                .suggested_namespace
                .as_ref()
                .map(|s| format!(" → should be [{s}]"))
                .unwrap_or_default();
            println!(
                "  {} [{}]{}",
                issue.concept_name, issue.current_namespace, suggestion
            );
        }
    }

    println!();
    if report.cross_namespace.is_empty() {
        println!("🔗 CROSS-NAMESPACE: none");
    } else {
        println!(
            "🔗 CROSS-NAMESPACE ({} found):",
            report.cross_namespace.len()
        );
        for issue in &report.cross_namespace {
            let total = issue.same_ns_relationships + issue.other_ns_relationships;
            let other_pct = if total > 0 {
                (issue.other_ns_relationships as f64 / total as f64 * 100.0) as u32
            } else {
                0
            };
            let dominant = issue
                .suggested_namespace
                .as_ref()
                .map(|s| format!(" to {s}"))
                .unwrap_or_default();
            println!(
                "  {} [{}] - {}% relationships{}",
                issue.concept_name, issue.current_namespace, other_pct, dominant
            );
        }
    }

    println!();
    println!("═══════════════════════════════════════");
    let total = report.global_refugees.len()
        + report.prefix_mismatches.len()
        + report.cross_namespace.len();
    println!("Total namespace issues: {total}");

    Ok(())
}

#[derive(Debug, Clone, Serialize)]
pub struct DedupeMerge {
    pub name: String,
    pub duplicate_namespace: String,
    pub canonical_namespace: String,
    pub similarity: f32,
    pub relationships_kept: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct DedupeReport {
    pub run_id: String,
    pub dry_run: bool,
    pub threshold: f32,
    pub merged: Vec<DedupeMerge>,
    pub held_back: Vec<DedupeMerge>,
}

async fn propose_merges(
    graph: &Graph,
    threshold: f32,
    namespace: Option<&str>,
) -> Result<(Vec<DedupeMerge>, Vec<DedupeMerge>)> {
    let ns_clause = if namespace.is_some() {
        " AND d.namespace = $namespace"
    } else {
        ""
    };
    let cypher = format!(
        r"
        MATCH (c:Concept {{namespace: 'global'}})
        WHERE c.embedding IS NOT NULL AND c.expired_at IS NULL
        MATCH (d:Concept)
        WHERE d.name = c.name AND d.namespace <> 'global'
          AND d.embedding IS NOT NULL AND d.expired_at IS NULL{ns_clause}
        OPTIONAL MATCH (d)-[r]-()
        RETURN d.name AS name, d.namespace AS dup_ns, c.namespace AS canon_ns,
               c.embedding AS canon_emb, d.embedding AS dup_emb, count(r) AS rels
        "
    );
    let mut q = query(&cypher);
    if let Some(ns) = namespace {
        q = q.param("namespace", ns.to_string());
    }

    let mut result = graph.execute(q).await?;
    let (mut merge, mut hold) = (Vec::new(), Vec::new());
    while let Some(row) = result.next().await? {
        let canon_emb: Vec<f64> = row.get("canon_emb").unwrap_or_default();
        let dup_emb: Vec<f64> = row.get("dup_emb").unwrap_or_default();
        if canon_emb.is_empty() || dup_emb.is_empty() {
            continue;
        }
        let a: Vec<f32> = canon_emb.iter().map(|&x| x as f32).collect();
        let b: Vec<f32> = dup_emb.iter().map(|&x| x as f32).collect();
        let entry = DedupeMerge {
            name: row.get("name").unwrap_or_default(),
            duplicate_namespace: row.get("dup_ns").unwrap_or_default(),
            canonical_namespace: row.get("canon_ns").unwrap_or_default(),
            similarity: cosine_similarity(&a, &b),
            relationships_kept: row.get("rels").unwrap_or(0),
        };
        if entry.similarity >= threshold {
            merge.push(entry);
        } else {
            hold.push(entry);
        }
    }
    merge.sort_by(|x, y| y.similarity.total_cmp(&x.similarity));
    hold.sort_by(|x, y| x.similarity.total_cmp(&y.similarity));
    Ok((merge, hold))
}

async fn apply_merge(
    graph: &Graph,
    m: &DedupeMerge,
    run_id: &str,
    adopted: Option<(&str, Vec<f32>)>,
) -> Result<()> {
    let q = query(
        r"
        MATCH (d:Concept {name: $name, namespace: $dup_ns})
        MATCH (c:Concept {name: $name, namespace: 'global'})
        SET d.expired_at = datetime(), d.merged_into = c.namespace, d.merge_run = $run
        MERGE (d)-[r:SAME_AS]->(c)
        ON CREATE SET r.merge_run = $run
        RETURN d.name AS name
        ",
    )
    .param("name", m.name.clone())
    .param("dup_ns", m.duplicate_namespace.clone())
    .param("run", run_id.to_string());
    graph.execute(q).await?.next().await?;

    if let Some((description, embedding)) = adopted {
        let embedding_vec: Vec<f64> = embedding.iter().map(|&x| f64::from(x)).collect();
        let q = query(
            r"
            MATCH (c:Concept {name: $name, namespace: 'global'})
            WHERE c.merge_prev_description IS NULL
            SET c.merge_prev_description = c.description,
                c.merge_desc_run = $run,
                c.description = $description,
                c.embedding = $embedding,
                c.updated_at = datetime()
            RETURN c.name AS name
            ",
        )
        .param("name", m.name.clone())
        .param("run", run_id.to_string())
        .param("description", description.to_string())
        .param("embedding", embedding_vec);
        graph.execute(q).await?.next().await?;
    }
    Ok(())
}

async fn better_description(graph: &Graph, m: &DedupeMerge) -> Result<Option<String>> {
    let q = query(
        r"
        MATCH (d:Concept {name: $name, namespace: $dup_ns})
        MATCH (c:Concept {name: $name, namespace: 'global'})
        RETURN coalesce(d.description, '') AS dup, coalesce(c.description, '') AS canon
        ",
    )
    .param("name", m.name.clone())
    .param("dup_ns", m.duplicate_namespace.clone());
    let mut result = graph.execute(q).await?;
    match result.next().await? {
        Some(row) => {
            let dup: String = row.get("dup").unwrap_or_default();
            let canon: String = row.get("canon").unwrap_or_default();
            Ok((dup.len() > canon.len()).then_some(dup))
        }
        None => Ok(None),
    }
}

pub async fn dedupe(
    graph: &Graph,
    threshold: f32,
    namespace: Option<&str>,
    dry_run: bool,
    adopt: Option<&crate::embeddings::OllamaClient>,
    json: bool,
) -> Result<()> {
    let run_id = format!(
        "dedupe-{}-{}",
        chrono::Utc::now().format("%Y%m%d-%H%M%S"),
        std::process::id()
    );
    let (merges, held) = propose_merges(graph, threshold, namespace).await?;

    let mut adopted_count = 0usize;
    if !dry_run {
        for m in &merges {
            let mut adopted = None;
            if let Some(client) = adopt
                && let Some(better) = better_description(graph, m).await?
                && let Ok(embedding) = client.embed(&better).await
            {
                adopted = Some((better, embedding));
            }
            let adopted_ref = adopted.as_ref().map(|(d, e)| (d.as_str(), e.clone()));
            if adopted_ref.is_some() {
                adopted_count += 1;
            }
            apply_merge(graph, m, &run_id, adopted_ref).await?;
        }
    }

    let report = DedupeReport {
        run_id: run_id.clone(),
        dry_run,
        threshold,
        merged: merges.clone(),
        held_back: held.clone(),
    };

    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    let title = if dry_run {
        "C0 Duplicate Merge (dry run)"
    } else {
        "C0 Duplicate Merge"
    };
    println!("{title}");
    println!("═══════════════════════════════════════");
    let kept: i64 = merges.iter().map(|m| m.relationships_kept).sum();
    println!(
        "  {} duplicate(s) at or above {:.2} — {} relationship(s) preserved",
        merges.len(),
        threshold,
        kept
    );
    println!(
        "  {} held back below threshold (review by hand)",
        held.len()
    );
    for m in held.iter().take(20) {
        println!(
            "    {:.2}  {} [{}]",
            m.similarity, m.name, m.duplicate_namespace
        );
    }
    println!("═══════════════════════════════════════");
    if dry_run {
        println!("Re-run without --dry-run to apply.");
    } else {
        println!("Merged {} concept(s), run-id: {run_id}", merges.len());
        if adopt.is_some() {
            println!("Adopted {adopted_count} richer description(s) onto the global concept.");
        }
        println!("Rollback: c0 audit dedupe --rollback {run_id}");
    }
    Ok(())
}

pub async fn dedupe_rollback(graph: &Graph, run: Option<&str>, json: bool) -> Result<()> {
    let run_id = match run {
        Some(r) if !r.is_empty() => r.to_string(),
        _ => {
            let q = query(
                r"
                MATCH (d:Concept) WHERE d.merge_run IS NOT NULL
                RETURN d.merge_run AS run ORDER BY run DESC LIMIT 1
                ",
            );
            let mut result = graph.execute(q).await?;
            match result.next().await? {
                Some(row) => row.get::<String>("run").unwrap_or_default(),
                None => {
                    if json {
                        println!("{}", serde_json::json!({"restored": 0, "run_id": null}));
                    } else {
                        println!("No dedupe runs found.");
                    }
                    return Ok(());
                }
            }
        }
    };

    let q = query(
        r"
        MATCH (c:Concept {merge_desc_run: $run})
        SET c.description = c.merge_prev_description,
            c.merge_prev_description = null,
            c.merge_desc_run = null
        RETURN count(c) AS n
        ",
    )
    .param("run", run_id.clone());
    graph.execute(q).await?.next().await?;

    let q = query(
        r"
        MATCH (d:Concept {merge_run: $run})
        OPTIONAL MATCH (d)-[r:SAME_AS {merge_run: $run}]->()
        DELETE r
        SET d.expired_at = null, d.merged_into = null, d.merge_run = null
        RETURN count(DISTINCT d) AS restored
        ",
    )
    .param("run", run_id.clone());
    let mut result = graph.execute(q).await?;
    let restored: i64 = match result.next().await? {
        Some(row) => row.get("restored").unwrap_or(0),
        None => 0,
    };

    if json {
        println!(
            "{}",
            serde_json::json!({"restored": restored, "run_id": run_id})
        );
    } else {
        println!("Restored {restored} concept(s) from run {run_id}.");
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod dedupe_tests {
    use crate::embeddings::cosine_similarity;

    fn normalised(raw: f32) -> f32 {
        (1.0 + raw) / 2.0
    }

    #[test]
    fn default_threshold_matches_neo4j_normalised_085() {
        assert!((normalised(0.70) - 0.85).abs() < 1e-6);
    }

    #[test]
    fn identical_embeddings_score_one() {
        let v = vec![0.3f32, 0.4, 0.5];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn orthogonal_embeddings_fall_below_default_threshold() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        assert!(cosine_similarity(&a, &b) < 0.70);
    }
}
