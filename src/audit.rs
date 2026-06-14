use anyhow::Result;
use neo4rs::{query, Graph};
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
    if union == 0.0 { 0.0 } else { intersection / union }
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
                .age_days.map_or_else(|| "unknown age".to_string(), |d| format!("{d} days old"));
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
    let total = report.age_stale.len()
        + report.orphaned.len()
        + report.supersession_candidates.len();
    println!("Total candidates for review: {total}");

    Ok(())
}

async fn find_global_refugees(
    graph: &Graph,
    namespaces: &[String],
) -> Result<Vec<NamespaceIssue>> {
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

async fn find_cross_namespace(
    graph: &Graph,
    namespaces: &[String],
) -> Result<Vec<NamespaceIssue>> {
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
        println!("🔗 CROSS-NAMESPACE ({} found):", report.cross_namespace.len());
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
