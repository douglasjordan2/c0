use anyhow::{Result, anyhow};
use chrono::{DateTime, Utc};
use neo4rs::{Graph, query};
use std::collections::HashMap;
use std::env;

const DEFAULT_NEO4J_URI: &str = "bolt://localhost:7687";

#[derive(Debug, Clone, Default)]
pub struct TemporalQuery {
    pub as_of: Option<DateTime<Utc>>,
    pub include_expired: bool,
}

impl TemporalQuery {
    pub fn build_where_clause(&self, node_alias: &str) -> String {
        if self.include_expired {
            return String::new();
        }

        let as_of_dt = self
            .as_of
            .map_or_else(|| "datetime()".to_string(), |dt| dt.to_rfc3339());
        let as_of_expr = if self.as_of.is_some() {
            format!("datetime('{as_of_dt}')")
        } else {
            "datetime()".to_string()
        };

        format!(
            "({node_alias}.valid_at IS NULL OR {node_alias}.valid_at <= {as_of_expr}) AND \
             ({node_alias}.invalid_at IS NULL OR {node_alias}.invalid_at > {as_of_expr}) AND \
             ({node_alias}.expired_at IS NULL OR {node_alias}.expired_at > {as_of_expr})"
        )
    }

    pub fn build_and_clause(&self, node_alias: &str) -> String {
        let clause = self.build_where_clause(node_alias);
        if clause.is_empty() {
            String::new()
        } else {
            format!(" AND {clause}")
        }
    }
}

fn neo4j_config() -> (String, String, String) {
    let uri = env::var("NEO4J_URI").unwrap_or_else(|_| DEFAULT_NEO4J_URI.to_string());
    let user = env::var("NEO4J_USER").unwrap_or_default();
    let password = env::var("NEO4J_PASSWORD").unwrap_or_default();
    (uri, user, password)
}

#[derive(Debug, Clone)]
pub struct Patch {
    pub name: String,
    pub file: Option<String>,
    pub content: Option<String>,
    pub namespace: String,
    pub url: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SemanticTrigger {
    pub name: String,
    pub description: String,
    pub namespace: String,
    pub threshold: Option<f32>,
    pub similarity: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct LiveSource {
    pub name: String,
    pub url: String,
    pub source_type: String,
    pub namespace: String,
    pub last_indexed: Option<String>,
    pub linked_concept: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Session {
    pub session_id: String,
    pub slug: Option<String>,
    pub cwd: String,
    pub namespace: String,
    pub first_prompt: String,
    pub summary: Option<String>,
    pub git_branch: Option<String>,
    pub created_at: String,
    pub ended_at: Option<String>,
    pub message_count: Option<i64>,
    pub is_sidechain: bool,
}

#[derive(Debug, Clone)]
pub struct InvalidationRecord {
    pub name: String,
    pub invalid_at: Option<String>,
    pub invalidated_by: Option<String>,
    pub reason: Option<String>,
}

#[cfg(feature = "sessions")]
#[derive(Debug, Clone, Default)]
pub struct Turn {
    pub turn_id: String,
    pub session_id: String,
    pub namespace: String,
    pub role: String,
    pub text: String,
    pub model: Option<String>,
    pub timestamp: String,
    pub parent_turn_id: Option<String>,
    pub is_sidechain: bool,
    pub git_branch: Option<String>,
    pub cwd: Option<String>,
    pub input_tokens: Option<i64>,
    pub output_tokens: Option<i64>,
    pub cache_creation_tokens: Option<i64>,
    pub cache_read_tokens: Option<i64>,
    pub tool_use_count: i64,
    pub tool_use_names: Vec<String>,
}

#[cfg(feature = "sessions")]
#[derive(Debug, Clone)]
pub struct Reflection {
    pub reflection_id: String,
    pub turn_id: String,
    pub session_id: String,
    pub namespace: String,
    pub text: String,
    pub signature: Option<String>,
    pub timestamp: String,
}

#[cfg(feature = "sessions")]
#[derive(Debug, Clone)]
pub struct ToolCallRecord {
    pub tool_call_id: String,
    pub turn_id: String,
    pub session_id: String,
    pub namespace: String,
    pub name: String,
    pub input_json: String,
    pub timestamp: String,
}

#[cfg(feature = "sessions")]
#[derive(Debug, Clone)]
pub struct FileTouch {
    pub path: String,
    pub action: String,
}

#[cfg(feature = "sessions")]
#[derive(Debug, Clone)]
pub struct BashCall {
    pub cmd: String,
    pub description: Option<String>,
}

#[cfg(feature = "sessions")]
#[derive(Debug, Clone)]
pub struct ToolResultBackfill {
    pub tool_call_id: String,
    pub is_error: bool,
    pub error_text: Option<String>,
}

#[cfg(feature = "sessions")]
#[derive(Debug, Clone, Default)]
pub struct SessionAggregates {
    pub total_turns: i64,
    pub total_text_chars: i64,
    pub total_thinking_chars: i64,
    pub total_input_tokens: i64,
    pub total_output_tokens: i64,
    pub total_tool_calls: i64,
}

pub async fn connect() -> Result<Graph> {
    let (uri, user, password) = neo4j_config();
    let graph = Graph::new(&uri, &user, &password).await?;
    Ok(graph)
}

pub async fn ping(graph: &Graph) -> Result<()> {
    graph.run(query("RETURN 1")).await?;
    Ok(())
}

pub async fn ensure_event(
    graph: &Graph,
    name: &str,
    reason: Option<&str>,
    namespace: &str,
) -> Result<bool> {
    let mut result = graph
        .execute(
            query(
                "MERGE (e:Event {name: $name, namespace: $namespace})
             ON CREATE SET e.reason = $reason, e.created_at = datetime()
             RETURN e.name AS name,
                    CASE WHEN e.created_at = datetime() THEN true ELSE false END AS created",
            )
            .param("name", name)
            .param("namespace", namespace)
            .param("reason", reason.unwrap_or("")),
        )
        .await?;

    if let Some(row) = result.next().await? {
        let created: bool = row.get("created").unwrap_or(false);
        Ok(created)
    } else {
        Ok(false)
    }
}

pub async fn add_concept(
    graph: &Graph,
    name: &str,
    namespace: &str,
    description: Option<&str>,
    source: Option<&str>,
    url: Option<&str>,
    embedding: Option<&[f32]>,
    valid_at: Option<DateTime<Utc>>,
) -> Result<()> {
    let valid_at_clause = valid_at.map_or_else(
        || "datetime()".to_string(),
        |dt| format!("datetime('{}')", dt.to_rfc3339()),
    );

    let cypher = if embedding.is_some() {
        format!(
            "MERGE (c:Concept {{name: $name, namespace: $namespace}})
             ON CREATE SET c.description = $description, c.source = $source, c.url = $url, c.embedding = $embedding,
                           c.created_at = datetime(), c.updated_at = datetime(), c.valid_at = {valid_at_clause}, c.invalid_at = null, c.expired_at = null
             ON MATCH SET c.embedding = $embedding,
                          c.description = CASE WHEN $description = '' THEN c.description ELSE $description END,
                          c.source = CASE WHEN $source = '' THEN c.source ELSE $source END,
                          c.url = CASE WHEN $url = '' THEN c.url ELSE $url END,
                          c.updated_at = datetime()"
        )
    } else {
        format!(
            "MERGE (c:Concept {{name: $name, namespace: $namespace}})
             ON CREATE SET c.description = $description, c.source = $source, c.url = $url,
                           c.created_at = datetime(), c.updated_at = datetime(), c.valid_at = {valid_at_clause}, c.invalid_at = null, c.expired_at = null
             ON MATCH SET c.description = CASE WHEN $description = '' THEN c.description ELSE $description END,
                          c.source = CASE WHEN $source = '' THEN c.source ELSE $source END,
                          c.url = CASE WHEN $url = '' THEN c.url ELSE $url END,
                          c.updated_at = datetime()"
        )
    };

    let embedding_vec: Vec<f64> = embedding
        .map(|e| e.iter().map(|&x| f64::from(x)).collect())
        .unwrap_or_default();

    let q = query(&cypher)
        .param("name", name)
        .param("namespace", namespace)
        .param("description", description.unwrap_or(""))
        .param("source", source.unwrap_or(""))
        .param("url", url.unwrap_or(""))
        .param("embedding", embedding_vec);

    graph.run(q).await?;
    Ok(())
}

pub async fn update_concept_embedding(
    graph: &Graph,
    name: &str,
    namespace: &str,
    embedding: &[f32],
) -> Result<bool> {
    let embedding_vec: Vec<f64> = embedding.iter().map(|&x| f64::from(x)).collect();

    let mut result = graph
        .execute(
            query(
                "MATCH (c:Concept {name: $name, namespace: $namespace})
             SET c.embedding = $embedding
             RETURN c.name AS name",
            )
            .param("name", name)
            .param("namespace", namespace)
            .param("embedding", embedding_vec),
        )
        .await?;

    Ok(result.next().await?.is_some())
}

pub async fn update_concept_description(
    graph: &Graph,
    name: &str,
    namespace: &str,
    description: &str,
    embedding: &[f32],
) -> Result<bool> {
    let embedding_vec: Vec<f64> = embedding.iter().map(|&x| f64::from(x)).collect();

    let mut result = graph
        .execute(
            query(
                "MATCH (c:Concept {name: $name, namespace: $namespace})
             SET c.description = $description, c.embedding = $embedding, c.updated_at = datetime()
             RETURN c.name AS name",
            )
            .param("name", name)
            .param("namespace", namespace)
            .param("description", description)
            .param("embedding", embedding_vec),
        )
        .await?;

    Ok(result.next().await?.is_some())
}

pub async fn get_concept_description(
    graph: &Graph,
    name: &str,
    namespaces: &[String],
) -> Result<Option<String>> {
    let mut result = graph
        .execute(
            query(
                "MATCH (c:Concept {name: $name})
             WHERE c.namespace IN $namespaces
             RETURN c.description AS description",
            )
            .param("name", name)
            .param("namespaces", namespaces.to_vec()),
        )
        .await?;

    if let Some(row) = result.next().await? {
        let desc: String = row.get("description").unwrap_or_default();
        if desc.is_empty() {
            Ok(None)
        } else {
            Ok(Some(desc))
        }
    } else {
        Ok(None)
    }
}

pub async fn get_concept_embedding(
    graph: &Graph,
    name: &str,
    namespaces: &[String],
) -> Result<Option<Vec<f32>>> {
    let mut result = graph
        .execute(
            query(
                "MATCH (c:Concept {name: $name})
             WHERE c.namespace IN $namespaces AND c.embedding IS NOT NULL
             RETURN c.embedding AS embedding",
            )
            .param("name", name)
            .param("namespaces", namespaces.to_vec()),
        )
        .await?;

    if let Some(row) = result.next().await? {
        let embedding: Vec<f64> = row.get("embedding").unwrap_or_default();
        if embedding.is_empty() {
            Ok(None)
        } else {
            Ok(Some(embedding.iter().map(|&x| x as f32).collect()))
        }
    } else {
        Ok(None)
    }
}

pub async fn get_concepts_without_embeddings(
    graph: &Graph,
    namespaces: &[String],
) -> Result<Vec<(String, String)>> {
    let mut result = graph
        .execute(
            query(
                "MATCH (c:Concept)
             WHERE c.namespace IN $namespaces
               AND c.embedding IS NULL
             RETURN c.name AS name, c.namespace AS namespace",
            )
            .param("namespaces", namespaces.to_vec()),
        )
        .await?;

    let mut concepts = Vec::new();
    while let Some(row) = result.next().await? {
        let name: String = row.get("name").unwrap_or_default();
        let namespace: String = row.get("namespace").unwrap_or_default();
        concepts.push((name, namespace));
    }
    Ok(concepts)
}

pub async fn relate(
    graph: &Graph,
    from: &str,
    rel_type: &str,
    to: &str,
    namespaces: &[String],
) -> Result<()> {
    let cypher = if rel_type == "HAS_PATCH" {
        format!(
            "MATCH (a:Concept {{name: $from}}), (b:KnowledgePatch {{name: $to}}) \
             WHERE a.namespace IN $namespaces AND (b.namespace IS NULL OR b.namespace IN $namespaces) \
             CREATE (a)-[:{rel_type}]->(b) RETURN count(*) AS created"
        )
    } else {
        format!(
            "MATCH (a:Concept {{name: $from}}), (b:Concept {{name: $to}}) \
             WHERE a.namespace IN $namespaces AND b.namespace IN $namespaces \
             CREATE (a)-[:{rel_type}]->(b) RETURN count(*) AS created"
        )
    };
    let mut result = graph
        .execute(
            query(&cypher)
                .param("from", from)
                .param("to", to)
                .param("namespaces", namespaces.to_vec()),
        )
        .await?;

    let created = match result.next().await? {
        Some(row) => row.get::<i64>("created").unwrap_or(0),
        None => 0,
    };
    if created == 0 {
        let target_kind = if rel_type == "HAS_PATCH" {
            "patch"
        } else {
            "concept"
        };
        return Err(anyhow!(
            "Could not create relationship '{from}' -[{rel_type}]-> '{to}': \
             the source concept and/or target {target_kind} was not found in \
             namespaces {namespaces:?}. Create both first (e.g. `c0 add concept`)."
        ));
    }
    Ok(())
}

pub async fn traverse_temporal(
    graph: &Graph,
    start: &str,
    depth: u32,
    namespaces: &[String],
    temporal: &TemporalQuery,
) -> Result<Vec<String>> {
    let temporal_clause = temporal.build_and_clause("connected");

    let cypher = format!(
        "MATCH (start:Concept {{name: $start}})-[*1..{depth}]->(connected) \
         WHERE start.namespace IN $namespaces AND connected.namespace IN $namespaces{temporal_clause} \
         RETURN DISTINCT connected.name AS name"
    );
    let mut result = graph
        .execute(
            query(&cypher)
                .param("start", start)
                .param("namespaces", namespaces.to_vec()),
        )
        .await?;

    let mut names = Vec::new();
    while let Some(row) = result.next().await? {
        if let Ok(name) = row.get::<String>("name") {
            names.push(name);
        }
    }
    Ok(names)
}

pub async fn find_pattern(graph: &Graph, pattern: &str) -> Result<Vec<String>> {
    let mut result = graph.execute(query(pattern)).await?;

    let mut output = Vec::new();
    while let Some(row) = result.next().await? {
        let row_str = format!("{row:?}");
        output.push(row_str);
    }
    Ok(output)
}

/// Expand `~` and resolve a patch file path to absolute form (against the
/// current working directory if relative). Does not require the file to exist,
/// so it is safe to call before the file is written.
pub fn absolutize_patch_path(f: &str) -> String {
    let expanded = shellexpand::tilde(f).to_string();
    let p = std::path::Path::new(&expanded);
    if p.is_absolute() {
        expanded
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(p).to_string_lossy().to_string())
            .unwrap_or(expanded)
    }
}

pub async fn add_patch(
    graph: &Graph,
    name: &str,
    corrects: Option<&str>,
    file: Option<&str>,
    content: Option<&str>,
    namespace: &str,
    source: Option<&str>,
    url: Option<&str>,
    valid_at: Option<DateTime<Utc>>,
) -> Result<()> {
    // Absolutize the patch file path so the patch resolves from any CWD or
    // machine. Storing a relative/tilde path in the shared graph renders the
    // patch empty when walked from a different directory or host.
    let abs_file = file.map(absolutize_patch_path);

    let mut params = vec![
        ("name", name.to_string()),
        ("namespace", namespace.to_string()),
    ];
    // MERGE on identity (name, namespace); everything else is SET so re-running
    // updates in place instead of creating duplicate empty shells.
    let mut sets = vec![
        "p.invalid_at = null".to_string(),
        "p.expired_at = null".to_string(),
    ];

    if let Some(f) = &abs_file {
        params.push(("file", f.clone()));
        sets.push("p.patch_file = $file".to_string());
    }
    if let Some(c) = content {
        params.push(("content", c.to_string()));
        sets.push("p.content = $content".to_string());
    }
    if let Some(s) = source {
        params.push(("source", s.to_string()));
        sets.push("p.source = $source".to_string());
    }
    if let Some(u) = url {
        params.push(("url", u.to_string()));
        sets.push("p.url = $url".to_string());
    }

    let valid_at_str = valid_at.map_or_else(
        || "datetime()".to_string(),
        |dt| format!("datetime('{}')", dt.to_rfc3339()),
    );
    sets.push(format!("p.valid_at = {valid_at_str}"));

    let cypher = format!(
        "MERGE (p:KnowledgePatch {{name: $name, namespace: $namespace}})
         ON CREATE SET p.created_at = datetime()
         SET {}",
        sets.join(", ")
    );
    let mut q = query(&cypher);
    for (k, v) in &params {
        q = q.param(k, v.as_str());
    }
    graph.run(q).await?;

    if let Some(concept) = corrects {
        graph
            .run(
                query(
                    "MATCH (c:Concept {name: $concept}), (p:KnowledgePatch {name: $patch})
                 WHERE c.namespace IN $namespaces
                 MERGE (c)-[:HAS_PATCH]->(p)
                 MERGE (p)-[:CORRECTS]->(c)",
                )
                .param("concept", concept)
                .param("patch", name)
                .param(
                    "namespaces",
                    vec!["global".to_string(), namespace.to_string()],
                ),
            )
            .await?;
    }

    Ok(())
}

pub async fn link_patch(
    graph: &Graph,
    patch: &str,
    concept: &str,
    namespaces: &[String],
) -> Result<()> {
    graph
        .run(
            query(
                "MATCH (c:Concept {name: $concept}), (p:KnowledgePatch {name: $patch})
             WHERE c.namespace IN $namespaces
             MERGE (c)-[:HAS_PATCH]->(p)",
            )
            .param("concept", concept)
            .param("patch", patch)
            .param("namespaces", namespaces.to_vec()),
        )
        .await?;
    Ok(())
}

pub async fn get_patches_temporal(
    graph: &Graph,
    concept: &str,
    namespaces: &[String],
    temporal: &TemporalQuery,
) -> Result<Vec<Patch>> {
    let temporal_clause = temporal.build_and_clause("p");

    let cypher = format!(
        "MATCH (c:Concept {{name: $name}})-[:HAS_PATCH]->(p:KnowledgePatch)
         WHERE c.namespace IN $namespaces AND (p.namespace IS NULL OR p.namespace IN $namespaces){temporal_clause}
         RETURN p.name AS name, p.patch_file AS file, p.content AS content,
                COALESCE(p.namespace, 'global') AS namespace, p.url AS url,
                p.invalid_at AS invalid_at, p.expired_at AS expired_at
         UNION
         MATCH (c:Concept {{name: $name}})<-[:CORRECTS]-(p:KnowledgePatch)
         WHERE c.namespace IN $namespaces AND (p.namespace IS NULL OR p.namespace IN $namespaces){temporal_clause}
         RETURN p.name AS name, p.patch_file AS file, p.content AS content,
                COALESCE(p.namespace, 'global') AS namespace, p.url AS url,
                p.invalid_at AS invalid_at, p.expired_at AS expired_at"
    );

    let mut result = graph
        .execute(
            query(&cypher)
                .param("name", concept)
                .param("namespaces", namespaces.to_vec()),
        )
        .await?;

    let mut patches = Vec::new();
    while let Some(row) = result.next().await? {
        let name: String = row.get("name").unwrap_or_default();
        let file: Option<String> = row.get("file").ok();
        let content: Option<String> = row.get("content").ok();
        let namespace: String = row
            .get("namespace")
            .unwrap_or_else(|_| "global".to_string());
        let url: Option<String> = row.get("url").ok();
        patches.push(Patch {
            name,
            file,
            content,
            namespace,
            url,
        });
    }
    Ok(patches)
}

pub async fn list_patches(
    graph: &Graph,
    namespaces: &[String],
) -> Result<Vec<(String, Option<String>, Option<String>, String)>> {
    let mut result = graph
        .execute(
            query(
                "MATCH (p:KnowledgePatch)
             WHERE p.namespace IS NULL OR p.namespace IN $namespaces
             OPTIONAL MATCH (p)-[:CORRECTS]->(c:Concept)
             RETURN p.name AS name, p.patch_file AS file, c.name AS corrects,
                    COALESCE(p.namespace, 'global') AS namespace",
            )
            .param("namespaces", namespaces.to_vec()),
        )
        .await?;

    let mut patches = Vec::new();
    while let Some(row) = result.next().await? {
        let name: String = row.get("name").unwrap_or_default();
        let file: Option<String> = row.get("file").ok();
        let corrects: Option<String> = row.get("corrects").ok();
        let namespace: String = row
            .get("namespace")
            .unwrap_or_else(|_| "global".to_string());
        patches.push((name, file, corrects, namespace));
    }
    Ok(patches)
}

pub async fn search_concepts(
    graph: &Graph,
    term: &str,
    namespaces: &[String],
) -> Result<Vec<String>> {
    let mut result = graph
        .execute(
            query(
                "MATCH (c:Concept)
             WHERE toLower(c.name) CONTAINS toLower($term) AND c.namespace IN $namespaces
             OPTIONAL MATCH (c)-[:HAS_PATCH]->(p:KnowledgePatch)
             RETURN c.name AS name, count(p) AS patch_count
             ORDER BY patch_count DESC, size(c.name)",
            )
            .param("term", term)
            .param("namespaces", namespaces.to_vec()),
        )
        .await?;

    let mut names = Vec::new();
    while let Some(row) = result.next().await? {
        if let Ok(name) = row.get::<String>("name") {
            names.push(name);
        }
    }
    Ok(names)
}

pub async fn migrate_add_global_namespace(graph: &Graph) -> Result<u64> {
    graph
        .run(query(
            "MATCH (n) WHERE n.namespace IS NULL SET n.namespace = 'global'",
        ))
        .await?;
    Ok(0)
}

pub async fn count_nodes_without_namespace(graph: &Graph) -> Result<u64> {
    let mut result = graph
        .execute(query(
            "MATCH (n) WHERE n.namespace IS NULL RETURN count(n) AS count",
        ))
        .await?;

    if let Some(row) = result.next().await? {
        let count: i64 = row.get("count").unwrap_or(0);
        return Ok(count as u64);
    }
    Ok(0)
}

pub async fn count_nodes_without_temporal(graph: &Graph) -> Result<u64> {
    let mut result = graph
        .execute(query(
            "MATCH (n)
             WHERE (n:Concept OR n:KnowledgePatch) AND n.valid_at IS NULL
             RETURN count(n) AS count",
        ))
        .await?;

    if let Some(row) = result.next().await? {
        let count: i64 = row.get("count").unwrap_or(0);
        return Ok(count as u64);
    }
    Ok(0)
}

pub async fn migrate_add_temporal_fields(graph: &Graph) -> Result<u64> {
    graph
        .run(query(
            "MATCH (n)
         WHERE (n:Concept OR n:KnowledgePatch) AND n.valid_at IS NULL
         SET n.created_at = COALESCE(n.created_at, datetime()),
             n.valid_at = datetime(),
             n.invalid_at = null,
             n.expired_at = null",
        ))
        .await?;

    graph
        .run(query(
            "CREATE INDEX concept_valid_at IF NOT EXISTS FOR (c:Concept) ON (c.valid_at)",
        ))
        .await?;

    graph
        .run(query(
            "CREATE INDEX concept_invalid_at IF NOT EXISTS FOR (c:Concept) ON (c.invalid_at)",
        ))
        .await?;

    graph
        .run(query(
            "CREATE INDEX patch_valid_at IF NOT EXISTS FOR (p:KnowledgePatch) ON (p.valid_at)",
        ))
        .await?;

    Ok(0)
}

pub async fn invalidate_concept(
    graph: &Graph,
    name: &str,
    namespace: &str,
    invalid_at: Option<DateTime<Utc>>,
    invalidated_by: Option<&str>,
    reason: Option<&str>,
    namespaces: &[String],
) -> Result<Option<String>> {
    let invalid_at_clause = invalid_at.map_or_else(
        || "datetime()".to_string(),
        |dt| format!("datetime('{}')", dt.to_rfc3339()),
    );

    let cypher = format!(
        "MATCH (c:Concept {{name: $name, namespace: $namespace}})
         WHERE c.invalid_at IS NULL
         SET c.invalid_at = {invalid_at_clause}
         RETURN c.name AS name, toString(c.invalid_at) AS invalid_at"
    );

    let mut result = graph
        .execute(
            query(&cypher)
                .param("name", name)
                .param("namespace", namespace),
        )
        .await?;

    if let Some(row) = result.next().await? {
        let invalid_at_str: String = row.get("invalid_at").unwrap_or_default();

        if let Some(by_name) = invalidated_by {
            let target_exists = graph
                .execute(
                    query(
                        "MATCH (target)
                     WHERE target.name = $by_name
                       AND (target:Concept OR target:Event OR target:KnowledgePatch)
                       AND (target.namespace IN $namespaces OR target.namespace IS NULL)
                     RETURN target.name AS name
                     LIMIT 1",
                    )
                    .param("by_name", by_name)
                    .param("namespaces", namespaces.to_vec()),
                )
                .await?
                .next()
                .await?
                .is_some();

            if !target_exists {
                ensure_event(graph, by_name, reason, namespace).await?;
            }

            graph
                .run(
                    query(
                        "MATCH (c:Concept {name: $name, namespace: $namespace})
                     MATCH (target)
                     WHERE target.name = $by_name
                       AND (target:Concept OR target:Event OR target:KnowledgePatch)
                       AND (target.namespace IN $namespaces OR target.namespace IS NULL)
                     MERGE (c)-[:INVALIDATED_BY]->(target)",
                    )
                    .param("name", name)
                    .param("namespace", namespace)
                    .param("by_name", by_name)
                    .param("namespaces", namespaces.to_vec()),
                )
                .await?;
        }

        Ok(Some(invalid_at_str))
    } else {
        Ok(None)
    }
}

pub async fn invalidate_patch(
    graph: &Graph,
    name: &str,
    namespace: &str,
    invalid_at: Option<DateTime<Utc>>,
    invalidated_by: Option<&str>,
    reason: Option<&str>,
    namespaces: &[String],
) -> Result<Option<String>> {
    let invalid_at_clause = invalid_at.map_or_else(
        || "datetime()".to_string(),
        |dt| format!("datetime('{}')", dt.to_rfc3339()),
    );

    let cypher = format!(
        "MATCH (p:KnowledgePatch {{name: $name}})
         WHERE (p.namespace IS NULL OR p.namespace = $namespace) AND p.invalid_at IS NULL
         SET p.invalid_at = {invalid_at_clause}
         RETURN p.name AS name, toString(p.invalid_at) AS invalid_at"
    );

    let mut result = graph
        .execute(
            query(&cypher)
                .param("name", name)
                .param("namespace", namespace),
        )
        .await?;

    if let Some(row) = result.next().await? {
        let invalid_at_str: String = row.get("invalid_at").unwrap_or_default();

        if let Some(by_name) = invalidated_by {
            let target_exists = graph
                .execute(
                    query(
                        "MATCH (target)
                     WHERE target.name = $by_name
                       AND (target:Concept OR target:Event OR target:KnowledgePatch)
                       AND (target.namespace IN $namespaces OR target.namespace IS NULL)
                     RETURN target.name AS name
                     LIMIT 1",
                    )
                    .param("by_name", by_name)
                    .param("namespaces", namespaces.to_vec()),
                )
                .await?
                .next()
                .await?
                .is_some();

            if !target_exists {
                ensure_event(graph, by_name, reason, namespace).await?;
            }

            graph
                .run(
                    query(
                        "MATCH (p:KnowledgePatch {name: $name})
                     WHERE p.namespace IS NULL OR p.namespace = $namespace
                     MATCH (target)
                     WHERE target.name = $by_name
                       AND (target:Concept OR target:Event OR target:KnowledgePatch)
                       AND (target.namespace IN $namespaces OR target.namespace IS NULL)
                     MERGE (p)-[:INVALIDATED_BY]->(target)",
                    )
                    .param("name", name)
                    .param("namespace", namespace)
                    .param("by_name", by_name)
                    .param("namespaces", namespaces.to_vec()),
                )
                .await?;
        }

        Ok(Some(invalid_at_str))
    } else {
        Ok(None)
    }
}

pub async fn supersede_concept(
    graph: &Graph,
    old_name: &str,
    new_name: &str,
    namespaces: &[String],
    expired_at: Option<DateTime<Utc>>,
) -> Result<bool> {
    let expired_at_clause = expired_at.map_or_else(
        || "datetime()".to_string(),
        |dt| format!("datetime('{}')", dt.to_rfc3339()),
    );

    let cypher = format!(
        "MATCH (old:Concept {{name: $old_name}}), (new:Concept {{name: $new_name}})
         WHERE old.namespace IN $namespaces AND new.namespace IN $namespaces
           AND old.expired_at IS NULL
         SET old.expired_at = {expired_at_clause}
         MERGE (new)-[:SUPERSEDES]->(old)
         RETURN old.name AS old_name, new.name AS new_name"
    );

    let mut result = graph
        .execute(
            query(&cypher)
                .param("old_name", old_name)
                .param("new_name", new_name)
                .param("namespaces", namespaces.to_vec()),
        )
        .await?;

    Ok(result.next().await?.is_some())
}

pub async fn get_supersession_chain(
    graph: &Graph,
    name: &str,
    namespaces: &[String],
) -> Result<Vec<(String, Option<String>)>> {
    let mut result = graph
        .execute(
            query(
                "MATCH path = (current:Concept {name: $name})-[:SUPERSEDES*0..10]->(old:Concept)
             WHERE current.namespace IN $namespaces
             UNWIND nodes(path) AS node
             WITH DISTINCT node.name AS name, node.expired_at AS exp_at
             RETURN name, toString(exp_at) AS expired_at
             ORDER BY COALESCE(exp_at, datetime('9999-12-31')) DESC",
            )
            .param("name", name)
            .param("namespaces", namespaces.to_vec()),
        )
        .await?;

    let mut chain = Vec::new();
    while let Some(row) = result.next().await? {
        let name: String = row.get("name").unwrap_or_default();
        let expired_at: Option<String> = row.get("expired_at").ok();
        chain.push((name, expired_at));
    }
    Ok(chain)
}

pub async fn get_invalidation_chain(
    graph: &Graph,
    name: &str,
    namespaces: &[String],
) -> Result<Vec<InvalidationRecord>> {
    let mut result = graph
        .execute(
            query(
                "MATCH (n {name: $name})
             WHERE n.namespace IN $namespaces
               AND (n:Concept OR n:KnowledgePatch)
             OPTIONAL MATCH (n)-[:INVALIDATED_BY]->(cause)
             RETURN n.name AS name,
                    toString(n.invalid_at) AS invalid_at,
                    cause.name AS invalidated_by,
                    cause.reason AS reason",
            )
            .param("name", name)
            .param("namespaces", namespaces.to_vec()),
        )
        .await?;

    let mut records = Vec::new();
    while let Some(row) = result.next().await? {
        let name: String = row.get("name").unwrap_or_default();
        let invalid_at: Option<String> = row
            .get("invalid_at")
            .ok()
            .filter(|s: &String| !s.is_empty());
        let invalidated_by: Option<String> = row
            .get("invalidated_by")
            .ok()
            .filter(|s: &String| !s.is_empty());
        let reason: Option<String> = row.get("reason").ok().filter(|s: &String| !s.is_empty());

        records.push(InvalidationRecord {
            name,
            invalid_at,
            invalidated_by,
            reason,
        });
    }
    Ok(records)
}

pub async fn ensure_semantic_trigger_index(graph: &Graph) -> Result<()> {
    graph
        .run(query(
            "CREATE CONSTRAINT semantic_trigger_name IF NOT EXISTS
         FOR (t:SemanticTrigger) REQUIRE t.name IS UNIQUE",
        ))
        .await?;

    graph
        .run(query(
            "CREATE VECTOR INDEX semantic_trigger_embedding IF NOT EXISTS
         FOR (t:SemanticTrigger)
         ON t.embedding
         OPTIONS {indexConfig: {
           `vector.dimensions`: 768,
           `vector.similarity_function`: 'cosine'
         }}",
        ))
        .await?;

    Ok(())
}

pub async fn check_concept_duplicates(graph: &Graph) -> Result<Vec<(String, String, i64)>> {
    let mut result = graph
        .execute(query(
            "MATCH (c:Concept)
         WITH c.name AS name, c.namespace AS namespace, count(*) AS cnt
         WHERE cnt > 1
         RETURN name, namespace, cnt",
        ))
        .await?;

    let mut duplicates = Vec::new();
    while let Some(row) = result.next().await? {
        let name: String = row.get("name")?;
        let namespace: String = row.get("namespace")?;
        let cnt: i64 = row.get("cnt")?;
        duplicates.push((name, namespace, cnt));
    }
    Ok(duplicates)
}

pub async fn ensure_concept_unique_constraint(graph: &Graph) -> Result<()> {
    let duplicates = check_concept_duplicates(graph).await?;
    if !duplicates.is_empty() {
        eprintln!(
            "⚠️  Found {} duplicate concept(s) - constraint not created:",
            duplicates.len()
        );
        for (name, namespace, cnt) in &duplicates {
            eprintln!("   - {namespace}:{name} ({cnt} copies)");
        }
        eprintln!("   Fix duplicates first, then retry.");
        return Ok(());
    }

    graph
        .run(query(
            "CREATE CONSTRAINT concept_name_namespace IF NOT EXISTS
         FOR (c:Concept) REQUIRE (c.name, c.namespace) IS UNIQUE",
        ))
        .await?;
    Ok(())
}

pub async fn ensure_concept_embedding_index(graph: &Graph) -> Result<()> {
    graph
        .run(query(
            "CREATE VECTOR INDEX concept_embedding IF NOT EXISTS
         FOR (c:Concept)
         ON c.embedding
         OPTIONS {indexConfig: {
           `vector.dimensions`: 768,
           `vector.similarity_function`: 'cosine'
         }}",
        ))
        .await?;

    Ok(())
}

pub async fn ensure_concept_fulltext_index(graph: &Graph) -> Result<()> {
    graph
        .run(query(
            "CREATE FULLTEXT INDEX concept_fulltext IF NOT EXISTS
         FOR (c:Concept) ON EACH [c.name, c.description]",
        ))
        .await?;
    Ok(())
}

pub async fn ensure_patch_fulltext_index(graph: &Graph) -> Result<()> {
    graph
        .run(query(
            "CREATE FULLTEXT INDEX patch_fulltext IF NOT EXISTS
         FOR (p:KnowledgePatch) ON EACH [p.name, p.content]",
        ))
        .await?;
    Ok(())
}

pub async fn check_fulltext_index_exists(graph: &Graph) -> Result<bool> {
    let mut result = graph.execute(query(
        "SHOW INDEXES YIELD name, type WHERE type = 'FULLTEXT' AND name = 'concept_fulltext' RETURN name"
    )).await?;
    Ok(result.next().await?.is_some())
}

pub async fn add_semantic_trigger(
    graph: &Graph,
    name: &str,
    description: &str,
    embedding: &[f32],
    namespace: &str,
    threshold: Option<f32>,
) -> Result<()> {
    let embedding_vec: Vec<f64> = embedding.iter().map(|&x| f64::from(x)).collect();

    graph
        .run(
            query(
                "MERGE (t:SemanticTrigger {name: $name, namespace: $namespace})
             SET t.description = $description,
                 t.embedding = $embedding,
                 t.threshold = $threshold,
                 t.updated_at = datetime()",
            )
            .param("name", name)
            .param("description", description)
            .param("embedding", embedding_vec)
            .param("namespace", namespace)
            .param("threshold", threshold),
        )
        .await?;

    Ok(())
}

pub async fn remove_semantic_trigger(graph: &Graph, name: &str, namespace: &str) -> Result<bool> {
    let mut result = graph
        .execute(
            query(
                "MATCH (t:SemanticTrigger {name: $name, namespace: $namespace})
             DELETE t
             RETURN count(t) AS deleted",
            )
            .param("name", name)
            .param("namespace", namespace),
        )
        .await?;

    if let Some(row) = result.next().await? {
        let deleted: i64 = row.get("deleted").unwrap_or(0);
        return Ok(deleted > 0);
    }
    Ok(false)
}

pub async fn list_semantic_triggers(
    graph: &Graph,
    namespaces: &[String],
) -> Result<Vec<SemanticTrigger>> {
    let mut result = graph
        .execute(
            query(
                "MATCH (t:SemanticTrigger)
             WHERE t.namespace IN $namespaces
             RETURN t.name AS name, t.description AS description,
                    t.namespace AS namespace, t.threshold AS threshold
             ORDER BY t.name",
            )
            .param("namespaces", namespaces.to_vec()),
        )
        .await?;

    let mut triggers = Vec::new();
    while let Some(row) = result.next().await? {
        let name: String = row.get("name").unwrap_or_default();
        let description: String = row.get("description").unwrap_or_default();
        let namespace: String = row
            .get("namespace")
            .unwrap_or_else(|_| "global".to_string());
        let threshold: Option<f32> = row.get("threshold").ok();

        triggers.push(SemanticTrigger {
            name,
            description,
            namespace,
            threshold,
            similarity: None,
        });
    }
    Ok(triggers)
}

pub async fn find_similar_triggers(
    graph: &Graph,
    embedding: &[f32],
    default_threshold: f32,
    floor_threshold: f32,
    namespaces: &[String],
) -> Result<Vec<SemanticTrigger>> {
    let embedding_vec: Vec<f64> = embedding.iter().map(|&x| f64::from(x)).collect();

    let mut result = graph
        .execute(
            query(
                "CALL db.index.vector.queryNodes('semantic_trigger_embedding', 10, $embedding)
             YIELD node, score
             WHERE node.namespace IN $namespaces
               AND score >= $floor_threshold
             RETURN node.name AS name, node.description AS description,
                    node.namespace AS namespace, node.threshold AS trigger_threshold,
                    score AS similarity
             ORDER BY score DESC",
            )
            .param("embedding", embedding_vec)
            .param("floor_threshold", f64::from(floor_threshold))
            .param("namespaces", namespaces.to_vec()),
        )
        .await?;

    let mut triggers = Vec::new();
    while let Some(row) = result.next().await? {
        let name: String = row.get("name").unwrap_or_default();
        let description: String = row.get("description").unwrap_or_default();
        let namespace: String = row
            .get("namespace")
            .unwrap_or_else(|_| "global".to_string());
        let trigger_threshold: Option<f32> = row.get("trigger_threshold").ok();
        let similarity: f64 = row.get("similarity").unwrap_or(0.0);

        let effective_threshold = trigger_threshold.unwrap_or(default_threshold);
        if similarity as f32 >= effective_threshold {
            triggers.push(SemanticTrigger {
                name,
                description,
                namespace,
                threshold: trigger_threshold,
                similarity: Some(similarity as f32),
            });
        }
    }
    Ok(triggers)
}

pub async fn get_related_concept_names(
    graph: &Graph,
    name: &str,
    namespaces: &[String],
) -> Result<Vec<String>> {
    let mut result = graph
        .execute(
            query(
                "MATCH (c:Concept {name: $name})
             WHERE c.namespace IN $namespaces
             OPTIONAL MATCH (c)-[:RELATES_TO|DEPENDS_ON|USES]-(related:Concept)
             WHERE related.namespace IN $namespaces
             RETURN DISTINCT related.name AS name
             LIMIT 10",
            )
            .param("name", name)
            .param("namespaces", namespaces.to_vec()),
        )
        .await?;

    let mut names = Vec::new();
    while let Some(row) = result.next().await? {
        if let Ok(name) = row.get::<String>("name") {
            names.push(name);
        }
    }
    Ok(names)
}

pub async fn find_similar_concepts(
    graph: &Graph,
    embedding: &[f32],
    threshold: f32,
    namespaces: &[String],
) -> Result<Vec<(String, f32)>> {
    find_similar_concepts_temporal(
        graph,
        embedding,
        threshold,
        namespaces,
        &TemporalQuery::default(),
    )
    .await
}

pub async fn find_similar_concepts_temporal(
    graph: &Graph,
    embedding: &[f32],
    threshold: f32,
    namespaces: &[String],
    temporal: &TemporalQuery,
) -> Result<Vec<(String, f32)>> {
    let embedding_vec: Vec<f64> = embedding.iter().map(|&x| f64::from(x)).collect();
    let temporal_clause = temporal.build_and_clause("node");

    let cypher = format!(
        "CALL db.index.vector.queryNodes('concept_embedding', 10, $embedding)
         YIELD node, score
         WHERE node.namespace IN $namespaces
           AND score >= $threshold{temporal_clause}
         RETURN node.name AS name, score AS similarity
         ORDER BY score DESC"
    );

    let result = graph
        .execute(
            query(&cypher)
                .param("embedding", embedding_vec)
                .param("threshold", f64::from(threshold))
                .param("namespaces", namespaces.to_vec()),
        )
        .await;

    match result {
        Ok(mut rows) => {
            let mut concepts = Vec::new();
            while let Some(row) = rows.next().await? {
                let name: String = row.get("name").unwrap_or_default();
                let similarity: f64 = row.get("similarity").unwrap_or(0.0);
                concepts.push((name, similarity as f32));
            }
            Ok(concepts)
        }
        Err(_) => Ok(Vec::new()),
    }
}

pub async fn ensure_live_source_index(graph: &Graph) -> Result<()> {
    graph
        .run(query(
            "CREATE CONSTRAINT live_source_name IF NOT EXISTS
         FOR (s:LiveSource) REQUIRE (s.name, s.namespace) IS UNIQUE",
        ))
        .await?;

    graph
        .run(query(
            "CREATE VECTOR INDEX live_source_embedding IF NOT EXISTS
         FOR (s:LiveSource)
         ON s.embedding
         OPTIONS {indexConfig: {
           `vector.dimensions`: 768,
           `vector.similarity_function`: 'cosine'
         }}",
        ))
        .await?;

    Ok(())
}

pub async fn add_live_source(
    graph: &Graph,
    name: &str,
    url: &str,
    source_type: &str,
    namespace: &str,
    linked_concept: Option<&str>,
    embedding: Option<&[f32]>,
) -> Result<()> {
    let embedding_vec: Vec<f64> = embedding
        .map(|e| e.iter().map(|&x| f64::from(x)).collect())
        .unwrap_or_default();

    let cypher = if embedding.is_some() {
        "MERGE (s:LiveSource {name: $name, namespace: $namespace})
         SET s.url = $url,
             s.source_type = $source_type,
             s.linked_concept = $linked_concept,
             s.embedding = $embedding,
             s.last_indexed = datetime(),
             s.created_at = COALESCE(s.created_at, datetime())"
    } else {
        "MERGE (s:LiveSource {name: $name, namespace: $namespace})
         SET s.url = $url,
             s.source_type = $source_type,
             s.linked_concept = $linked_concept,
             s.created_at = COALESCE(s.created_at, datetime())"
    };

    graph
        .run(
            query(cypher)
                .param("name", name)
                .param("url", url)
                .param("source_type", source_type)
                .param("namespace", namespace)
                .param("linked_concept", linked_concept.unwrap_or(""))
                .param("embedding", embedding_vec),
        )
        .await?;

    if let Some(concept) = linked_concept
        && !concept.is_empty()
    {
        graph.run(
                query(
                    "MATCH (c:Concept {name: $concept}), (s:LiveSource {name: $source, namespace: $namespace})
                     WHERE c.namespace IN $namespaces
                     MERGE (c)-[:HAS_LIVE_SOURCE]->(s)"
                )
                .param("concept", concept)
                .param("source", name)
                .param("namespace", namespace)
                .param("namespaces", vec!["global".to_string(), namespace.to_string()])
            ).await?;
    }

    Ok(())
}

pub async fn update_live_source_embedding(
    graph: &Graph,
    name: &str,
    namespace: &str,
    embedding: &[f32],
) -> Result<bool> {
    let embedding_vec: Vec<f64> = embedding.iter().map(|&x| f64::from(x)).collect();

    let mut result = graph
        .execute(
            query(
                "MATCH (s:LiveSource {name: $name, namespace: $namespace})
             SET s.embedding = $embedding, s.last_indexed = datetime()
             RETURN s.name AS name",
            )
            .param("name", name)
            .param("namespace", namespace)
            .param("embedding", embedding_vec),
        )
        .await?;

    Ok(result.next().await?.is_some())
}

pub async fn get_live_source(
    graph: &Graph,
    name: &str,
    namespaces: &[String],
) -> Result<Option<LiveSource>> {
    let mut result = graph
        .execute(
            query(
                "MATCH (s:LiveSource {name: $name})
             WHERE s.namespace IN $namespaces
             RETURN s.name AS name, s.url AS url, s.source_type AS source_type,
                    s.namespace AS namespace, s.last_indexed AS last_indexed,
                    s.linked_concept AS linked_concept",
            )
            .param("name", name)
            .param("namespaces", namespaces.to_vec()),
        )
        .await?;

    if let Some(row) = result.next().await? {
        let name: String = row.get("name").unwrap_or_default();
        let url: String = row.get("url").unwrap_or_default();
        let source_type: String = row.get("source_type").unwrap_or_else(|_| "url".to_string());
        let namespace: String = row
            .get("namespace")
            .unwrap_or_else(|_| "global".to_string());
        let last_indexed: Option<String> = row.get("last_indexed").ok();
        let linked_concept: Option<String> = row
            .get("linked_concept")
            .ok()
            .filter(|s: &String| !s.is_empty());

        Ok(Some(LiveSource {
            name,
            url,
            source_type,
            namespace,
            last_indexed,
            linked_concept,
        }))
    } else {
        Ok(None)
    }
}

pub async fn list_live_sources(graph: &Graph, namespaces: &[String]) -> Result<Vec<LiveSource>> {
    let mut result = graph
        .execute(
            query(
                "MATCH (s:LiveSource)
             WHERE s.namespace IN $namespaces
             RETURN s.name AS name, s.url AS url, s.source_type AS source_type,
                    s.namespace AS namespace, s.last_indexed AS last_indexed,
                    s.linked_concept AS linked_concept
             ORDER BY s.name",
            )
            .param("namespaces", namespaces.to_vec()),
        )
        .await?;

    let mut sources = Vec::new();
    while let Some(row) = result.next().await? {
        let name: String = row.get("name").unwrap_or_default();
        let url: String = row.get("url").unwrap_or_default();
        let source_type: String = row.get("source_type").unwrap_or_else(|_| "url".to_string());
        let namespace: String = row
            .get("namespace")
            .unwrap_or_else(|_| "global".to_string());
        let last_indexed: Option<String> = row.get("last_indexed").ok();
        let linked_concept: Option<String> = row
            .get("linked_concept")
            .ok()
            .filter(|s: &String| !s.is_empty());

        sources.push(LiveSource {
            name,
            url,
            source_type,
            namespace,
            last_indexed,
            linked_concept,
        });
    }
    Ok(sources)
}

pub async fn remove_live_source(graph: &Graph, name: &str, namespace: &str) -> Result<bool> {
    let mut result = graph
        .execute(
            query(
                "MATCH (s:LiveSource {name: $name, namespace: $namespace})
             DETACH DELETE s
             RETURN count(s) AS deleted",
            )
            .param("name", name)
            .param("namespace", namespace),
        )
        .await?;

    if let Some(row) = result.next().await? {
        let deleted: i64 = row.get("deleted").unwrap_or(0);
        return Ok(deleted > 0);
    }
    Ok(false)
}

pub async fn find_similar_live_sources(
    graph: &Graph,
    embedding: &[f32],
    threshold: f32,
    namespaces: &[String],
    limit: usize,
) -> Result<Vec<(LiveSource, f32)>> {
    let embedding_vec: Vec<f64> = embedding.iter().map(|&x| f64::from(x)).collect();

    let result = graph
        .execute(
            query(
                "CALL db.index.vector.queryNodes('live_source_embedding', $limit, $embedding)
             YIELD node, score
             WHERE node.namespace IN $namespaces
               AND score >= $threshold
             RETURN node.name AS name, node.url AS url, node.source_type AS source_type,
                    node.namespace AS namespace, node.last_indexed AS last_indexed,
                    node.linked_concept AS linked_concept, score AS similarity
             ORDER BY score DESC",
            )
            .param("embedding", embedding_vec)
            .param("threshold", f64::from(threshold))
            .param("namespaces", namespaces.to_vec())
            .param("limit", limit as i64),
        )
        .await;

    match result {
        Ok(mut rows) => {
            let mut sources = Vec::new();
            while let Some(row) = rows.next().await? {
                let name: String = row.get("name").unwrap_or_default();
                let url: String = row.get("url").unwrap_or_default();
                let source_type: String =
                    row.get("source_type").unwrap_or_else(|_| "url".to_string());
                let namespace: String = row
                    .get("namespace")
                    .unwrap_or_else(|_| "global".to_string());
                let last_indexed: Option<String> = row.get("last_indexed").ok();
                let linked_concept: Option<String> = row
                    .get("linked_concept")
                    .ok()
                    .filter(|s: &String| !s.is_empty());
                let similarity: f64 = row.get("similarity").unwrap_or(0.0);

                sources.push((
                    LiveSource {
                        name,
                        url,
                        source_type,
                        namespace,
                        last_indexed,
                        linked_concept,
                    },
                    similarity as f32,
                ));
            }
            Ok(sources)
        }
        Err(_) => Ok(Vec::new()),
    }
}

pub async fn get_live_sources_for_concept(
    graph: &Graph,
    concept: &str,
    namespaces: &[String],
) -> Result<Vec<LiveSource>> {
    let mut result = graph
        .execute(
            query(
                "MATCH (c:Concept {name: $concept})-[:HAS_LIVE_SOURCE]->(s:LiveSource)
             WHERE c.namespace IN $namespaces
             RETURN s.name AS name, s.url AS url, s.source_type AS source_type,
                    s.namespace AS namespace, s.last_indexed AS last_indexed,
                    s.linked_concept AS linked_concept",
            )
            .param("concept", concept)
            .param("namespaces", namespaces.to_vec()),
        )
        .await?;

    let mut sources = Vec::new();
    while let Some(row) = result.next().await? {
        let name: String = row.get("name").unwrap_or_default();
        let url: String = row.get("url").unwrap_or_default();
        let source_type: String = row.get("source_type").unwrap_or_else(|_| "url".to_string());
        let namespace: String = row
            .get("namespace")
            .unwrap_or_else(|_| "global".to_string());
        let last_indexed: Option<String> = row.get("last_indexed").ok();
        let linked_concept: Option<String> = row
            .get("linked_concept")
            .ok()
            .filter(|s: &String| !s.is_empty());

        sources.push(LiveSource {
            name,
            url,
            source_type,
            namespace,
            last_indexed,
            linked_concept,
        });
    }
    Ok(sources)
}

#[derive(Debug, Clone)]
pub struct MoveResult {
    pub concept_moved: bool,
    pub patches_moved: i64,
    pub old_namespace: String,
    pub new_namespace: String,
}

pub async fn move_concept(
    graph: &Graph,
    name: &str,
    to_namespace: &str,
    include_patches: bool,
) -> Result<Option<MoveResult>> {
    let mut result = graph
        .execute(
            query(
                "MATCH (c:Concept {name: $name})
             WHERE c.namespace <> $to_namespace
             RETURN c.namespace AS old_namespace",
            )
            .param("name", name)
            .param("to_namespace", to_namespace),
        )
        .await?;

    let old_namespace: String = match result.next().await? {
        Some(row) => row.get("old_namespace").unwrap_or_default(),
        None => return Ok(None),
    };

    graph
        .run(
            query(
                "MATCH (c:Concept {name: $name})
             SET c.namespace = $to_namespace",
            )
            .param("name", name)
            .param("to_namespace", to_namespace),
        )
        .await?;

    let patches_moved = if include_patches {
        let mut patch_result = graph.execute(
            query(
                "MATCH (c:Concept {name: $name, namespace: $to_namespace})-[:HAS_PATCH]->(p:KnowledgePatch)
                 WHERE p.namespace = $old_namespace
                 SET p.namespace = $to_namespace
                 RETURN count(p) AS moved"
            )
            .param("name", name)
            .param("to_namespace", to_namespace)
            .param("old_namespace", old_namespace.clone())
        ).await?;

        patch_result
            .next()
            .await?
            .map_or(0, |r| r.get::<i64>("moved").unwrap_or(0))
    } else {
        0
    };

    Ok(Some(MoveResult {
        concept_moved: true,
        patches_moved,
        old_namespace,
        new_namespace: to_namespace.to_string(),
    }))
}

pub async fn move_concepts_by_prefix(
    graph: &Graph,
    prefix: &str,
    from_namespace: &str,
    to_namespace: &str,
    include_patches: bool,
) -> Result<(i64, i64)> {
    let pattern = format!("(?i)^{}[-_:].*", regex::escape(prefix));

    let mut result = graph
        .execute(
            query(
                "MATCH (c:Concept {namespace: $from_namespace})
             WHERE c.name =~ $pattern
             SET c.namespace = $to_namespace
             RETURN count(c) AS moved",
            )
            .param("from_namespace", from_namespace)
            .param("to_namespace", to_namespace)
            .param("pattern", pattern.clone()),
        )
        .await?;

    let concepts_moved: i64 = result
        .next()
        .await?
        .map_or(0, |r| r.get("moved").unwrap_or(0));

    let patches_moved =
        if include_patches {
            let mut patch_result = graph.execute(
            query(
                "MATCH (c:Concept {namespace: $to_namespace})-[:HAS_PATCH]->(p:KnowledgePatch)
                 WHERE c.name =~ $pattern AND p.namespace = $from_namespace
                 SET p.namespace = $to_namespace
                 RETURN count(p) AS moved"
            )
            .param("from_namespace", from_namespace)
            .param("to_namespace", to_namespace)
            .param("pattern", pattern.clone())
        ).await?;

            patch_result
                .next()
                .await?
                .map_or(0, |r| r.get::<i64>("moved").unwrap_or(0))
        } else {
            0
        };

    Ok((concepts_moved, patches_moved))
}

pub async fn list_concepts_by_prefix(
    graph: &Graph,
    prefix: &str,
    namespace: &str,
) -> Result<Vec<String>> {
    let pattern = format!("(?i)^{}[-_:].*", regex::escape(prefix));

    let mut result = graph
        .execute(
            query(
                "MATCH (c:Concept {namespace: $namespace})
             WHERE c.name =~ $pattern
             RETURN c.name AS name
             ORDER BY c.name",
            )
            .param("namespace", namespace)
            .param("pattern", pattern),
        )
        .await?;

    let mut names = Vec::new();
    while let Some(row) = result.next().await? {
        if let Ok(name) = row.get::<String>("name") {
            names.push(name);
        }
    }

    Ok(names)
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SearchResult {
    pub name: String,
    pub namespace: String,
    pub description: Option<String>,
    pub similarity: f32,
}

pub async fn search_concepts_semantic(
    graph: &Graph,
    embedding: &[f32],
    limit: usize,
    threshold: f32,
    namespaces: &[String],
) -> Result<Vec<SearchResult>> {
    let embedding_vec: Vec<f64> = embedding.iter().map(|&x| f64::from(x)).collect();

    let result = graph
        .execute(
            query(
                "CALL db.index.vector.queryNodes('concept_embedding', $limit, $embedding)
             YIELD node, score
             WHERE node.namespace IN $namespaces
               AND score >= $threshold
             RETURN node.name AS name, node.namespace AS namespace,
                    node.description AS description, score AS similarity
             ORDER BY score DESC",
            )
            .param("embedding", embedding_vec)
            .param("limit", limit as i64)
            .param("threshold", f64::from(threshold))
            .param("namespaces", namespaces.to_vec()),
        )
        .await;

    match result {
        Ok(mut rows) => {
            let mut results = Vec::new();
            while let Some(row) = rows.next().await? {
                let name: String = row.get("name").unwrap_or_default();
                let namespace: String = row.get("namespace").unwrap_or_default();
                let description: Option<String> = row
                    .get("description")
                    .ok()
                    .filter(|s: &String| !s.is_empty());
                let similarity: f64 = row.get("similarity").unwrap_or(0.0);
                results.push(SearchResult {
                    name,
                    namespace,
                    description,
                    similarity: similarity as f32,
                });
            }
            Ok(results)
        }
        Err(_) => Ok(Vec::new()),
    }
}

fn sanitize_lucene_query(input: &str) -> String {
    let special = [
        '+', '-', '&', '|', '!', '(', ')', '{', '}', '[', ']', '^', '"', '~', '*', '?', ':', '\\',
        '/',
    ];
    let mut out = String::with_capacity(input.len() * 2);
    for ch in input.chars() {
        if special.contains(&ch) {
            out.push('\\');
        }
        out.push(ch);
    }
    out
}

fn build_fulltext_query(input: &str) -> String {
    let sanitized = sanitize_lucene_query(input);
    let terms: Vec<&str> = sanitized.split_whitespace().collect();
    if terms.len() == 1 {
        let t = &terms[0];
        format!("name:{t}^3 OR description:{t} OR name:{t}~")
    } else {
        let phrase = terms.join(" ");
        let boosted: Vec<String> = terms
            .iter()
            .map(|t| format!("name:{t}^3 OR description:{t}"))
            .collect();
        format!("\"{}\"^5 OR {}", phrase, boosted.join(" OR "))
    }
}

pub async fn search_concepts_fulltext(
    graph: &Graph,
    query_text: &str,
    limit: usize,
    namespaces: &[String],
) -> Result<Vec<SearchResult>> {
    let ft_query = build_fulltext_query(query_text);

    let result = graph
        .execute(
            query(
                "CALL db.index.fulltext.queryNodes('concept_fulltext', $query, {limit: $limit})
             YIELD node, score
             WHERE node.namespace IN $namespaces
             RETURN node.name AS name, node.namespace AS namespace,
                    node.description AS description, score AS similarity
             ORDER BY score DESC",
            )
            .param("query", ft_query)
            .param("limit", limit as i64)
            .param("namespaces", namespaces.to_vec()),
        )
        .await;

    match result {
        Ok(mut rows) => {
            let mut results = Vec::new();
            while let Some(row) = rows.next().await? {
                let name: String = row.get("name").unwrap_or_default();
                let namespace: String = row.get("namespace").unwrap_or_default();
                let description: Option<String> = row
                    .get("description")
                    .ok()
                    .filter(|s: &String| !s.is_empty());
                let similarity: f64 = row.get("similarity").unwrap_or(0.0);
                results.push(SearchResult {
                    name,
                    namespace,
                    description,
                    similarity: similarity as f32,
                });
            }
            Ok(results)
        }
        Err(_) => Ok(Vec::new()),
    }
}

pub struct HybridSearchConfig {
    pub alpha: f32,
    pub k: f32,
    pub vector_limit: usize,
    pub fulltext_limit: usize,
    pub vector_threshold: f32,
}

impl Default for HybridSearchConfig {
    fn default() -> Self {
        Self {
            alpha: 0.4,
            k: 60.0,
            vector_limit: 20,
            fulltext_limit: 20,
            vector_threshold: 0.3,
        }
    }
}

fn reciprocal_rank_fusion(
    query_text: &str,
    keyword_results: &[SearchResult],
    vector_results: &[SearchResult],
    alpha: f32,
    k: f32,
) -> Vec<SearchResult> {
    let mut scores: HashMap<(String, String), (f32, Option<String>)> = HashMap::new();

    for (rank, r) in keyword_results.iter().enumerate() {
        let key = (r.name.clone(), r.namespace.clone());
        let rrf_score = alpha / (k + (rank + 1) as f32);
        let entry = scores.entry(key).or_insert((0.0, r.description.clone()));
        entry.0 += rrf_score;
    }

    for (rank, r) in vector_results.iter().enumerate() {
        let key = (r.name.clone(), r.namespace.clone());
        let rrf_score = (1.0 - alpha) / (k + (rank + 1) as f32);
        let entry = scores.entry(key).or_insert((0.0, r.description.clone()));
        entry.0 += rrf_score;
        if entry.1.is_none() {
            entry.1 = r.description.clone();
        }
    }

    // Raw RRF scores are compressed into [0, 1/(k+1)] because the formula only
    // uses rank position, not match magnitude. A perfect hit (rank #1 in both
    // lists) tops out near 0.0164 with default k, which sits in the same band as
    // semantic noise and reads as a miss to any absolute-threshold consumer.
    // Multiplying by (k+1) rescales to [0,1]: rank-#1-in-both -> 1.0,
    // keyword-only-#1 -> alpha, vector-only-#1 -> (1 - alpha).
    let norm = k + 1.0;
    let query_norm = query_text.trim().to_lowercase();

    let mut fused: Vec<SearchResult> = scores
        .into_iter()
        .map(|((name, namespace), (score, desc))| {
            // An exact (case-insensitive) name match must dominate regardless of
            // where the embedding model happened to rank it among semantic noise.
            let similarity = if name.trim().to_lowercase() == query_norm {
                1.0
            } else {
                (score * norm).min(1.0)
            };
            SearchResult {
                name,
                namespace,
                description: desc,
                similarity,
            }
        })
        .collect();
    fused.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    fused
}

pub async fn search_hybrid(
    graph: &Graph,
    query_text: &str,
    embedding: &[f32],
    limit: usize,
    namespaces: &[String],
    config: &HybridSearchConfig,
) -> Result<Vec<SearchResult>> {
    let ft_results =
        search_concepts_fulltext(graph, query_text, config.fulltext_limit, namespaces).await?;
    let vec_results = search_concepts_semantic(
        graph,
        embedding,
        config.vector_limit,
        config.vector_threshold,
        namespaces,
    )
    .await?;
    let mut fused = reciprocal_rank_fusion(
        query_text,
        &ft_results,
        &vec_results,
        config.alpha,
        config.k,
    );
    fused.truncate(limit);
    Ok(fused)
}

pub async fn search_hybrid_temporal(
    graph: &Graph,
    query_text: &str,
    embedding: &[f32],
    namespaces: &[String],
    temporal: &TemporalQuery,
    config: &HybridSearchConfig,
) -> Result<Vec<(String, f32)>> {
    let ft_results =
        search_concepts_fulltext(graph, query_text, config.fulltext_limit, namespaces).await?;
    let vec_raw = find_similar_concepts_temporal(
        graph,
        embedding,
        config.vector_threshold,
        namespaces,
        temporal,
    )
    .await
    .unwrap_or_default();

    let vec_results: Vec<SearchResult> = vec_raw
        .iter()
        .map(|(name, score)| SearchResult {
            name: name.clone(),
            namespace: String::new(),
            description: None,
            similarity: *score,
        })
        .collect();

    let fused = reciprocal_rank_fusion(
        query_text,
        &ft_results,
        &vec_results,
        config.alpha,
        config.k,
    );
    Ok(fused.into_iter().map(|r| (r.name, r.similarity)).collect())
}

pub async fn find_orphaned_concepts(
    graph: &Graph,
    namespaces: &[String],
) -> Result<Vec<(String, String)>> {
    let mut result = graph
        .execute(
            query(
                "MATCH (c:Concept)
             WHERE c.namespace IN $namespaces
             AND NOT (c)-[]-()
             RETURN c.name AS name, c.namespace AS namespace
             ORDER BY c.name",
            )
            .param("namespaces", namespaces.to_vec()),
        )
        .await?;

    let mut orphans = Vec::new();
    while let Some(row) = result.next().await? {
        let name: String = row.get("name").unwrap_or_default();
        let namespace: String = row.get("namespace").unwrap_or_default();
        orphans.push((name, namespace));
    }
    Ok(orphans)
}

pub async fn count_concepts_by_namespace(graph: &Graph) -> Result<Vec<(String, i64)>> {
    let mut result = graph
        .execute(query(
            "MATCH (c:Concept)
             RETURN c.namespace AS namespace, count(c) AS count
             ORDER BY count DESC",
        ))
        .await?;

    let mut counts = Vec::new();
    while let Some(row) = result.next().await? {
        let namespace: String = row.get("namespace").unwrap_or_default();
        let count: i64 = row.get("count").unwrap_or(0);
        counts.push((namespace, count));
    }
    Ok(counts)
}

pub async fn find_patches_with_files(
    graph: &Graph,
    namespaces: &[String],
) -> Result<Vec<(String, String, String)>> {
    let mut result = graph
        .execute(
            query(
                "MATCH (p:KnowledgePatch)
             WHERE (p.namespace IS NULL OR p.namespace IN $namespaces)
               AND p.patch_file IS NOT NULL AND p.patch_file <> ''
             RETURN p.name AS name, COALESCE(p.namespace, 'global') AS namespace,
                    p.patch_file AS file_path",
            )
            .param("namespaces", namespaces.to_vec()),
        )
        .await?;

    let mut patches = Vec::new();
    while let Some(row) = result.next().await? {
        let name: String = row.get("name").unwrap_or_default();
        let namespace: String = row.get("namespace").unwrap_or_default();
        let file_path: String = row.get("file_path").unwrap_or_default();
        patches.push((name, namespace, file_path));
    }
    Ok(patches)
}

pub async fn clear_patch_file_reference(
    graph: &Graph,
    patch_name: &str,
    namespace: &str,
) -> Result<bool> {
    let mut result = graph
        .execute(
            query(
                "MATCH (p:KnowledgePatch {name: $name})
             WHERE p.namespace IS NULL OR p.namespace = $namespace
             REMOVE p.patch_file
             RETURN p.name AS name",
            )
            .param("name", patch_name)
            .param("namespace", namespace),
        )
        .await?;

    Ok(result.next().await?.is_some())
}

/// Every KnowledgePatch across all namespaces, with its file ref and whether it
/// already has inline content. Used by `c0 backfill patch-content` to find
/// patches that render empty (no content + missing/relative patch_file).
pub async fn find_all_patches(
    graph: &Graph,
) -> Result<Vec<(String, String, Option<String>, bool)>> {
    let mut result = graph
        .execute(query(
            "MATCH (p:KnowledgePatch)
             RETURN p.name AS name, COALESCE(p.namespace, 'global') AS namespace,
                    p.patch_file AS file,
                    (p.content IS NOT NULL AND p.content <> '') AS has_content",
        ))
        .await?;

    let mut out = Vec::new();
    while let Some(row) = result.next().await? {
        let name: String = row.get("name").unwrap_or_default();
        let namespace: String = row.get("namespace").unwrap_or_default();
        let file: Option<String> = row.get("file").unwrap_or(None);
        let has_content: bool = row.get("has_content").unwrap_or(false);
        out.push((name, namespace, file, has_content));
    }
    Ok(out)
}

/// Inline a patch's content into the graph and normalize its file ref to an
/// absolute path, making it readable from any machine (the graph is shared but
/// patch files are stored on individual hosts).
pub async fn set_patch_content(
    graph: &Graph,
    name: &str,
    namespace: &str,
    content: &str,
    abs_file: &str,
) -> Result<bool> {
    let mut result = graph
        .execute(
            query(
                "MATCH (p:KnowledgePatch {name: $name})
             WHERE COALESCE(p.namespace, 'global') = $namespace
             SET p.content = $content, p.patch_file = $file
             RETURN p.name AS name",
            )
            .param("name", name)
            .param("namespace", namespace)
            .param("content", content)
            .param("file", abs_file),
        )
        .await?;

    Ok(result.next().await?.is_some())
}

pub async fn export_nodes_by_label(
    graph: &Graph,
    label: &str,
    namespace_filter: Option<&[String]>,
    no_embeddings: bool,
) -> Result<Vec<serde_json::Value>> {
    let cypher = if namespace_filter.is_some() {
        format!(
            "MATCH (n:{label})
             WHERE n.namespace IN $namespaces
             RETURN n, labels(n) AS labels"
        )
    } else {
        format!(
            "MATCH (n:{label})
             RETURN n, labels(n) AS labels"
        )
    };

    let ns = namespace_filter.map(|ns| ns.to_vec()).unwrap_or_default();

    let mut result = graph
        .execute(query(&cypher).param("namespaces", ns))
        .await?;

    let mut nodes = Vec::new();
    while let Some(row) = result.next().await? {
        let node: neo4rs::Node = row.get("n")?;
        let labels: Vec<String> = row.get("labels").unwrap_or_default();

        let mut props = serde_json::Map::new();
        for key in node.keys() {
            if no_embeddings && key == "embedding" {
                continue;
            }
            if let Ok(val) = node.get::<String>(key) {
                props.insert(key.to_string(), serde_json::Value::String(val));
            } else if let Ok(val) = node.get::<i64>(key) {
                props.insert(key.to_string(), serde_json::json!(val));
            } else if let Ok(val) = node.get::<f64>(key) {
                props.insert(key.to_string(), serde_json::json!(val));
            } else if let Ok(val) = node.get::<bool>(key) {
                props.insert(key.to_string(), serde_json::json!(val));
            } else if let Ok(val) = node.get::<Vec<f64>>(key) {
                props.insert(key.to_string(), serde_json::json!(val));
            }
        }

        nodes.push(serde_json::json!({
            "labels": labels,
            "properties": props,
        }));
    }
    Ok(nodes)
}

pub async fn export_all_relationships(
    graph: &Graph,
    namespace_filter: Option<&[String]>,
) -> Result<Vec<serde_json::Value>> {
    let cypher = if namespace_filter.is_some() {
        "MATCH (a)-[r]->(b)
         WHERE (a.namespace IS NULL OR a.namespace IN $namespaces)
           AND (b.namespace IS NULL OR b.namespace IN $namespaces)
         RETURN COALESCE(a.name, a.id, a.gid) AS start_name,
                labels(a) AS start_labels,
                type(r) AS rel_type,
                COALESCE(b.name, b.id, b.gid) AS end_name,
                labels(b) AS end_labels"
    } else {
        "MATCH (a)-[r]->(b)
         RETURN COALESCE(a.name, a.id, a.gid) AS start_name,
                labels(a) AS start_labels,
                type(r) AS rel_type,
                COALESCE(b.name, b.id, b.gid) AS end_name,
                labels(b) AS end_labels"
    };

    let ns = namespace_filter.map(|ns| ns.to_vec()).unwrap_or_default();

    let mut result = graph.execute(query(cypher).param("namespaces", ns)).await?;

    let mut rels = Vec::new();
    while let Some(row) = result.next().await? {
        let start_name: String = row.get("start_name").unwrap_or_default();
        let start_labels: Vec<String> = row.get("start_labels").unwrap_or_default();
        let rel_type: String = row.get("rel_type").unwrap_or_default();
        let end_name: String = row.get("end_name").unwrap_or_default();
        let end_labels: Vec<String> = row.get("end_labels").unwrap_or_default();

        rels.push(serde_json::json!({
            "start_name": start_name,
            "start_labels": start_labels,
            "rel_type": rel_type,
            "end_name": end_name,
            "end_labels": end_labels,
        }));
    }
    Ok(rels)
}

#[cfg(feature = "sessions")]
pub async fn add_session(
    graph: &Graph,
    session: &Session,
    embedding: Option<&[f32]>,
) -> Result<()> {
    let embedding_vec: Option<Vec<f64>> =
        embedding.map(|e| e.iter().map(|&x| f64::from(x)).collect());

    let cypher = if embedding_vec.is_some() {
        "MERGE (s:Session {session_id: $session_id})
         SET s.slug = $slug,
             s.cwd = $cwd,
             s.namespace = $namespace,
             s.first_prompt = $first_prompt,
             s.summary = $summary,
             s.git_branch = $git_branch,
             s.created_at = $created_at,
             s.ended_at = $ended_at,
             s.message_count = $message_count,
             s.is_sidechain = $is_sidechain,
             s.embedding = $embedding,
             s.indexed_at = datetime()"
    } else {
        "MERGE (s:Session {session_id: $session_id})
         SET s.slug = $slug,
             s.cwd = $cwd,
             s.namespace = $namespace,
             s.first_prompt = $first_prompt,
             s.summary = $summary,
             s.git_branch = $git_branch,
             s.created_at = $created_at,
             s.ended_at = $ended_at,
             s.message_count = $message_count,
             s.is_sidechain = $is_sidechain,
             s.indexed_at = datetime()"
    };

    let mut q = query(cypher)
        .param("session_id", session.session_id.as_str())
        .param("slug", session.slug.as_deref().unwrap_or(""))
        .param("cwd", session.cwd.as_str())
        .param("namespace", session.namespace.as_str())
        .param("first_prompt", session.first_prompt.as_str())
        .param("summary", session.summary.as_deref().unwrap_or(""))
        .param("git_branch", session.git_branch.as_deref().unwrap_or(""))
        .param("created_at", session.created_at.as_str())
        .param("ended_at", session.ended_at.as_deref().unwrap_or(""))
        .param("message_count", session.message_count.unwrap_or(0))
        .param("is_sidechain", session.is_sidechain);

    if let Some(ref emb) = embedding_vec {
        q = q.param("embedding", emb.clone());
    }

    graph.run(q).await?;
    Ok(())
}

#[cfg(feature = "sessions")]
pub async fn get_sessions(
    graph: &Graph,
    namespaces: &[String],
    limit: usize,
) -> Result<Vec<Session>> {
    let ns: Vec<&str> = namespaces.iter().map(String::as_str).collect();
    let filter_ns = !ns.is_empty();

    let cypher = if filter_ns {
        "MATCH (s:Session)
         WHERE s.namespace IN $namespaces
         RETURN s.session_id AS session_id, s.slug AS slug, s.cwd AS cwd,
                s.namespace AS namespace, s.first_prompt AS first_prompt,
                s.summary AS summary, s.git_branch AS git_branch,
                s.created_at AS created_at, s.ended_at AS ended_at,
                s.message_count AS message_count, s.is_sidechain AS is_sidechain
         ORDER BY s.created_at DESC
         LIMIT $limit"
    } else {
        "MATCH (s:Session)
         RETURN s.session_id AS session_id, s.slug AS slug, s.cwd AS cwd,
                s.namespace AS namespace, s.first_prompt AS first_prompt,
                s.summary AS summary, s.git_branch AS git_branch,
                s.created_at AS created_at, s.ended_at AS ended_at,
                s.message_count AS message_count, s.is_sidechain AS is_sidechain
         ORDER BY s.created_at DESC
         LIMIT $limit"
    };

    let mut result = graph
        .execute(
            query(cypher)
                .param("namespaces", ns)
                .param("limit", limit as i64),
        )
        .await?;

    let mut sessions = Vec::new();
    while let Some(row) = result.next().await? {
        sessions.push(Session {
            session_id: row.get("session_id").unwrap_or_default(),
            slug: row.get::<String>("slug").ok().filter(|s| !s.is_empty()),
            cwd: row.get("cwd").unwrap_or_default(),
            namespace: row.get("namespace").unwrap_or_default(),
            first_prompt: row.get("first_prompt").unwrap_or_default(),
            summary: row.get::<String>("summary").ok().filter(|s| !s.is_empty()),
            git_branch: row
                .get::<String>("git_branch")
                .ok()
                .filter(|s| !s.is_empty()),
            created_at: row.get("created_at").unwrap_or_default(),
            ended_at: row.get::<String>("ended_at").ok().filter(|s| !s.is_empty()),
            message_count: row.get("message_count").ok(),
            is_sidechain: row.get("is_sidechain").unwrap_or(false),
        });
    }
    Ok(sessions)
}

#[cfg(feature = "sessions")]
pub async fn search_sessions_hybrid(
    graph: &Graph,
    query_text: &str,
    embedding: Option<&[f32]>,
    limit: usize,
    namespaces: &[String],
) -> Result<Vec<(Session, f64)>> {
    let ns: Vec<&str> = namespaces.iter().map(String::as_str).collect();
    let filter_ns = !ns.is_empty();

    let mut fulltext_results: HashMap<String, (Session, f64)> = HashMap::new();
    let escaped = query_text.replace('\\', "\\\\").replace('"', "\\\"");
    let ft_query = format!("{escaped}~");

    let ft_cypher = if filter_ns {
        "CALL db.index.fulltext.queryNodes('session_fulltext', $query)
         YIELD node, score
         WHERE node.namespace IN $namespaces
         RETURN node.session_id AS session_id, node.slug AS slug, node.cwd AS cwd,
                node.namespace AS namespace, node.first_prompt AS first_prompt,
                node.summary AS summary, node.git_branch AS git_branch,
                node.created_at AS created_at, node.ended_at AS ended_at,
                node.message_count AS message_count, node.is_sidechain AS is_sidechain,
                score
         LIMIT $limit"
    } else {
        "CALL db.index.fulltext.queryNodes('session_fulltext', $query)
         YIELD node, score
         RETURN node.session_id AS session_id, node.slug AS slug, node.cwd AS cwd,
                node.namespace AS namespace, node.first_prompt AS first_prompt,
                node.summary AS summary, node.git_branch AS git_branch,
                node.created_at AS created_at, node.ended_at AS ended_at,
                node.message_count AS message_count, node.is_sidechain AS is_sidechain,
                score
         LIMIT $limit"
    };

    if let Ok(mut result) = graph
        .execute(
            query(ft_cypher)
                .param("query", ft_query.as_str())
                .param("namespaces", ns.clone())
                .param("limit", (limit * 2) as i64),
        )
        .await
    {
        while let Ok(Some(row)) = result.next().await {
            let session_id: String = row.get("session_id").unwrap_or_default();
            let score: f64 = row.get("score").unwrap_or(0.0);
            let session = Session {
                session_id: session_id.clone(),
                slug: row.get::<String>("slug").ok().filter(|s| !s.is_empty()),
                cwd: row.get("cwd").unwrap_or_default(),
                namespace: row.get("namespace").unwrap_or_default(),
                first_prompt: row.get("first_prompt").unwrap_or_default(),
                summary: row.get::<String>("summary").ok().filter(|s| !s.is_empty()),
                git_branch: row
                    .get::<String>("git_branch")
                    .ok()
                    .filter(|s| !s.is_empty()),
                created_at: row.get("created_at").unwrap_or_default(),
                ended_at: row.get::<String>("ended_at").ok().filter(|s| !s.is_empty()),
                message_count: row.get("message_count").ok(),
                is_sidechain: row.get("is_sidechain").unwrap_or(false),
            };
            fulltext_results.insert(session_id, (session, score));
        }
    }

    if let Some(emb) = embedding {
        let embedding_vec: Vec<f64> = emb.iter().map(|&x| f64::from(x)).collect();

        let vec_cypher = if filter_ns {
            "CALL db.index.vector.queryNodes('session_embedding', $limit, $embedding)
             YIELD node, score
             WHERE node.namespace IN $namespaces
             RETURN node.session_id AS session_id, node.slug AS slug, node.cwd AS cwd,
                    node.namespace AS namespace, node.first_prompt AS first_prompt,
                    node.summary AS summary, node.git_branch AS git_branch,
                    node.created_at AS created_at, node.ended_at AS ended_at,
                    node.message_count AS message_count, node.is_sidechain AS is_sidechain,
                    score"
        } else {
            "CALL db.index.vector.queryNodes('session_embedding', $limit, $embedding)
             YIELD node, score
             RETURN node.session_id AS session_id, node.slug AS slug, node.cwd AS cwd,
                    node.namespace AS namespace, node.first_prompt AS first_prompt,
                    node.summary AS summary, node.git_branch AS git_branch,
                    node.created_at AS created_at, node.ended_at AS ended_at,
                    node.message_count AS message_count, node.is_sidechain AS is_sidechain,
                    score"
        };

        if let Ok(mut result) = graph
            .execute(
                query(vec_cypher)
                    .param("embedding", embedding_vec)
                    .param("namespaces", ns)
                    .param("limit", (limit * 2) as i64),
            )
            .await
        {
            let ft_count = fulltext_results.len();
            while let Ok(Some(row)) = result.next().await {
                let session_id: String = row.get("session_id").unwrap_or_default();
                let vec_score: f64 = row.get("score").unwrap_or(0.0);

                if let Some(entry) = fulltext_results.get_mut(&session_id) {
                    entry.1 = (entry.1 + vec_score) / 2.0;
                } else {
                    let session = Session {
                        session_id: session_id.clone(),
                        slug: row.get::<String>("slug").ok().filter(|s| !s.is_empty()),
                        cwd: row.get("cwd").unwrap_or_default(),
                        namespace: row.get("namespace").unwrap_or_default(),
                        first_prompt: row.get("first_prompt").unwrap_or_default(),
                        summary: row.get::<String>("summary").ok().filter(|s| !s.is_empty()),
                        git_branch: row
                            .get::<String>("git_branch")
                            .ok()
                            .filter(|s| !s.is_empty()),
                        created_at: row.get("created_at").unwrap_or_default(),
                        ended_at: row.get::<String>("ended_at").ok().filter(|s| !s.is_empty()),
                        message_count: row.get("message_count").ok(),
                        is_sidechain: row.get("is_sidechain").unwrap_or(false),
                    };
                    let boost = if ft_count > 0 {
                        vec_score * 0.8
                    } else {
                        vec_score
                    };
                    fulltext_results.insert(session_id, (session, boost));
                }
            }
        }
    }

    let mut results: Vec<(Session, f64)> = fulltext_results.into_values().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(limit);
    Ok(results)
}

#[cfg(feature = "sessions")]
pub async fn ensure_session_indexes(graph: &Graph) -> Result<()> {
    graph
        .run(query(
            "CREATE CONSTRAINT session_id_unique IF NOT EXISTS
         FOR (s:Session) REQUIRE s.session_id IS UNIQUE",
        ))
        .await?;

    graph
        .run(query(
            "CREATE VECTOR INDEX session_embedding IF NOT EXISTS
         FOR (s:Session)
         ON s.embedding
         OPTIONS {indexConfig: {
           `vector.dimensions`: 768,
           `vector.similarity_function`: 'cosine'
         }}",
        ))
        .await?;

    graph
        .run(query(
            "CREATE FULLTEXT INDEX session_fulltext IF NOT EXISTS
         FOR (s:Session) ON EACH [s.first_prompt, s.summary, s.slug]",
        ))
        .await?;

    Ok(())
}

#[cfg(feature = "sessions")]
const COMMAND_CMD_MAX_LEN: usize = 4096;
#[cfg(feature = "sessions")]
const TOOLCALL_INPUT_MAX_LEN: usize = 4096;
#[cfg(feature = "sessions")]
const TOOLCALL_ERROR_MAX_LEN: usize = 2048;

#[cfg(feature = "sessions")]
fn truncate_str(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let mut end = max;
        while !s.is_char_boundary(end) && end > 0 {
            end -= 1;
        }
        s[..end].to_string()
    }
}

#[cfg(feature = "sessions")]
pub async fn add_turn(graph: &Graph, turn: &Turn, embedding: Option<&[f32]>) -> Result<()> {
    let embedding_vec: Option<Vec<f64>> =
        embedding.map(|e| e.iter().map(|&x| f64::from(x)).collect());

    let cypher = if embedding_vec.is_some() {
        "MERGE (t:Turn {turn_id: $turn_id})
         SET t.session_id = $session_id,
             t.namespace = $namespace,
             t.role = $role,
             t.text = $text,
             t.model = $model,
             t.timestamp = $timestamp,
             t.parent_turn_id = $parent_turn_id,
             t.is_sidechain = $is_sidechain,
             t.git_branch = $git_branch,
             t.cwd = $cwd,
             t.input_tokens = $input_tokens,
             t.output_tokens = $output_tokens,
             t.cache_creation_tokens = $cache_creation_tokens,
             t.cache_read_tokens = $cache_read_tokens,
             t.tool_use_count = $tool_use_count,
             t.tool_use_names = $tool_use_names,
             t.embedding = $embedding,
             t.created_at = coalesce(t.created_at, datetime()),
             t.updated_at = datetime()
         WITH t
         MATCH (s:Session {session_id: $session_id})
         MERGE (s)-[:HAS_TURN]->(t)"
    } else {
        "MERGE (t:Turn {turn_id: $turn_id})
         SET t.session_id = $session_id,
             t.namespace = $namespace,
             t.role = $role,
             t.text = $text,
             t.model = $model,
             t.timestamp = $timestamp,
             t.parent_turn_id = $parent_turn_id,
             t.is_sidechain = $is_sidechain,
             t.git_branch = $git_branch,
             t.cwd = $cwd,
             t.input_tokens = $input_tokens,
             t.output_tokens = $output_tokens,
             t.cache_creation_tokens = $cache_creation_tokens,
             t.cache_read_tokens = $cache_read_tokens,
             t.tool_use_count = $tool_use_count,
             t.tool_use_names = $tool_use_names,
             t.created_at = coalesce(t.created_at, datetime()),
             t.updated_at = datetime()
         WITH t
         MATCH (s:Session {session_id: $session_id})
         MERGE (s)-[:HAS_TURN]->(t)"
    };

    let names: Vec<&str> = turn.tool_use_names.iter().map(String::as_str).collect();

    let mut q = query(cypher)
        .param("turn_id", turn.turn_id.as_str())
        .param("session_id", turn.session_id.as_str())
        .param("namespace", turn.namespace.as_str())
        .param("role", turn.role.as_str())
        .param("text", turn.text.as_str())
        .param("model", turn.model.as_deref().unwrap_or(""))
        .param("timestamp", turn.timestamp.as_str())
        .param(
            "parent_turn_id",
            turn.parent_turn_id.as_deref().unwrap_or(""),
        )
        .param("is_sidechain", turn.is_sidechain)
        .param("git_branch", turn.git_branch.as_deref().unwrap_or(""))
        .param("cwd", turn.cwd.as_deref().unwrap_or(""))
        .param("input_tokens", turn.input_tokens.unwrap_or(0))
        .param("output_tokens", turn.output_tokens.unwrap_or(0))
        .param(
            "cache_creation_tokens",
            turn.cache_creation_tokens.unwrap_or(0),
        )
        .param("cache_read_tokens", turn.cache_read_tokens.unwrap_or(0))
        .param("tool_use_count", turn.tool_use_count)
        .param("tool_use_names", names);

    if let Some(ref emb) = embedding_vec {
        q = q.param("embedding", emb.clone());
    }

    graph.run(q).await?;
    Ok(())
}

#[cfg(feature = "sessions")]
pub async fn add_reflection(
    graph: &Graph,
    reflection: &Reflection,
    embedding: Option<&[f32]>,
) -> Result<()> {
    let embedding_vec: Option<Vec<f64>> =
        embedding.map(|e| e.iter().map(|&x| f64::from(x)).collect());

    let cypher = if embedding_vec.is_some() {
        "MERGE (r:Reflection {reflection_id: $reflection_id})
         SET r.turn_id = $turn_id,
             r.session_id = $session_id,
             r.namespace = $namespace,
             r.text = $text,
             r.signature = $signature,
             r.timestamp = $timestamp,
             r.embedding = $embedding,
             r.created_at = coalesce(r.created_at, datetime())
         WITH r
         MATCH (t:Turn {turn_id: $turn_id})
         MERGE (t)-[:HAS_REFLECTION]->(r)"
    } else {
        "MERGE (r:Reflection {reflection_id: $reflection_id})
         SET r.turn_id = $turn_id,
             r.session_id = $session_id,
             r.namespace = $namespace,
             r.text = $text,
             r.signature = $signature,
             r.timestamp = $timestamp,
             r.created_at = coalesce(r.created_at, datetime())
         WITH r
         MATCH (t:Turn {turn_id: $turn_id})
         MERGE (t)-[:HAS_REFLECTION]->(r)"
    };

    let mut q = query(cypher)
        .param("reflection_id", reflection.reflection_id.as_str())
        .param("turn_id", reflection.turn_id.as_str())
        .param("session_id", reflection.session_id.as_str())
        .param("namespace", reflection.namespace.as_str())
        .param("text", reflection.text.as_str())
        .param("signature", reflection.signature.as_deref().unwrap_or(""))
        .param("timestamp", reflection.timestamp.as_str());

    if let Some(ref emb) = embedding_vec {
        q = q.param("embedding", emb.clone());
    }

    graph.run(q).await?;
    Ok(())
}

#[cfg(feature = "sessions")]
pub async fn add_toolcall(
    graph: &Graph,
    call: &ToolCallRecord,
    file_touches: &[FileTouch],
    bash: Option<&BashCall>,
) -> Result<()> {
    let truncated_input = truncate_str(&call.input_json, TOOLCALL_INPUT_MAX_LEN);

    graph
        .run(
            query(
                "MERGE (tc:ToolCall {tool_call_id: $tool_call_id})
             SET tc.turn_id = $turn_id,
                 tc.session_id = $session_id,
                 tc.namespace = $namespace,
                 tc.name = $name,
                 tc.input_json = $input_json,
                 tc.timestamp = $timestamp,
                 tc.created_at = coalesce(tc.created_at, datetime())
             WITH tc
             MATCH (t:Turn {turn_id: $turn_id})
             MERGE (t)-[:CALLED]->(tc)",
            )
            .param("tool_call_id", call.tool_call_id.as_str())
            .param("turn_id", call.turn_id.as_str())
            .param("session_id", call.session_id.as_str())
            .param("namespace", call.namespace.as_str())
            .param("name", call.name.as_str())
            .param("input_json", truncated_input.as_str())
            .param("timestamp", call.timestamp.as_str()),
        )
        .await?;

    for touch in file_touches {
        graph
            .run(
                query(
                    "MERGE (f:File {path: $path})
                 ON CREATE SET f.created_at = datetime(), f.updated_at = datetime()
                 ON MATCH SET f.updated_at = datetime()
                 WITH f
                 MATCH (tc:ToolCall {tool_call_id: $tool_call_id})
                 MERGE (tc)-[r:TOUCHED]->(f)
                 SET r.action = $action",
                )
                .param("path", touch.path.as_str())
                .param("tool_call_id", call.tool_call_id.as_str())
                .param("action", touch.action.as_str()),
            )
            .await?;
    }

    if let Some(bc) = bash {
        let truncated_cmd = truncate_str(&bc.cmd, COMMAND_CMD_MAX_LEN);
        graph
            .run(
                query(
                    "MERGE (c:Command {cmd: $cmd})
                 ON CREATE SET c.first_seen_at = datetime(), c.last_seen_at = datetime()
                 ON MATCH SET c.last_seen_at = datetime()
                 WITH c
                 MATCH (tc:ToolCall {tool_call_id: $tool_call_id})
                 MERGE (tc)-[r:RAN]->(c)
                 SET r.description = $description",
                )
                .param("cmd", truncated_cmd.as_str())
                .param("tool_call_id", call.tool_call_id.as_str())
                .param("description", bc.description.as_deref().unwrap_or("")),
            )
            .await?;
    }

    Ok(())
}

#[cfg(feature = "sessions")]
pub async fn backfill_toolcall_result(graph: &Graph, backfill: &ToolResultBackfill) -> Result<()> {
    let truncated_err = backfill
        .error_text
        .as_deref()
        .map(|s| truncate_str(s, TOOLCALL_ERROR_MAX_LEN))
        .unwrap_or_default();

    graph
        .run(
            query(
                "MATCH (tc:ToolCall {tool_call_id: $tool_call_id})
             SET tc.is_error = $is_error,
                 tc.error_text = $error_text",
            )
            .param("tool_call_id", backfill.tool_call_id.as_str())
            .param("is_error", backfill.is_error)
            .param("error_text", truncated_err.as_str()),
        )
        .await?;
    Ok(())
}

#[cfg(feature = "sessions")]
pub async fn build_reply_chain(graph: &Graph, session_id: &str) -> Result<u64> {
    let mut result = graph
        .execute(
            query(
                "MATCH (t:Turn {session_id: $session_id})
             WHERE t.parent_turn_id IS NOT NULL AND t.parent_turn_id <> ''
             MATCH (p:Turn {turn_id: t.parent_turn_id})
             MERGE (t)-[:REPLIES_TO]->(p)
             RETURN count(*) AS n",
            )
            .param("session_id", session_id),
        )
        .await?;

    if let Some(row) = result.next().await? {
        let n: i64 = row.get("n").unwrap_or(0);
        Ok(n.max(0) as u64)
    } else {
        Ok(0)
    }
}

#[cfg(feature = "sessions")]
pub async fn delete_session_turns(graph: &Graph, session_id: &str) -> Result<u64> {
    let mut result = graph
        .execute(
            query(
                "MATCH (s:Session {session_id: $session_id})-[:HAS_TURN]->(t:Turn)
             OPTIONAL MATCH (t)-[:HAS_REFLECTION]->(r:Reflection)
             OPTIONAL MATCH (t)-[:CALLED]->(tc:ToolCall)
             WITH t, collect(DISTINCT r) AS reflections, collect(DISTINCT tc) AS toolcalls
             DETACH DELETE t
             WITH reflections, toolcalls
             UNWIND reflections AS r
             DETACH DELETE r
             WITH toolcalls
             UNWIND toolcalls AS tc
             DETACH DELETE tc
             RETURN count(*) AS n",
            )
            .param("session_id", session_id),
        )
        .await?;

    if let Some(row) = result.next().await? {
        let n: i64 = row.get("n").unwrap_or(0);
        Ok(n.max(0) as u64)
    } else {
        Ok(0)
    }
}

#[cfg(feature = "sessions")]
pub async fn update_session_aggregates(
    graph: &Graph,
    session_id: &str,
    agg: &SessionAggregates,
) -> Result<()> {
    graph
        .run(
            query(
                "MATCH (s:Session {session_id: $session_id})
             SET s.total_turns = $total_turns,
                 s.total_text_chars = $total_text_chars,
                 s.total_thinking_chars = $total_thinking_chars,
                 s.total_input_tokens = $total_input_tokens,
                 s.total_output_tokens = $total_output_tokens,
                 s.total_tool_calls = $total_tool_calls,
                 s.deep_indexed_at = datetime()",
            )
            .param("session_id", session_id)
            .param("total_turns", agg.total_turns)
            .param("total_text_chars", agg.total_text_chars)
            .param("total_thinking_chars", agg.total_thinking_chars)
            .param("total_input_tokens", agg.total_input_tokens)
            .param("total_output_tokens", agg.total_output_tokens)
            .param("total_tool_calls", agg.total_tool_calls),
        )
        .await?;
    Ok(())
}

#[cfg(feature = "sessions")]
pub async fn search_turns_hybrid(
    graph: &Graph,
    query_text: &str,
    embedding: Option<&[f32]>,
    limit: usize,
    namespaces: &[String],
) -> Result<Vec<(Turn, f64)>> {
    let ns: Vec<&str> = namespaces.iter().map(String::as_str).collect();
    let filter_ns = !ns.is_empty();

    let mut results: HashMap<String, (Turn, f64)> = HashMap::new();
    let escaped = query_text.replace('\\', "\\\\").replace('"', "\\\"");
    let ft_query = format!("{escaped}~");

    let ft_cypher = if filter_ns {
        "CALL db.index.fulltext.queryNodes('turn_fulltext', $query)
         YIELD node, score
         WHERE node.namespace IN $namespaces
         RETURN node.turn_id AS turn_id, node.session_id AS session_id,
                node.namespace AS namespace, node.role AS role, node.text AS text,
                node.model AS model, node.timestamp AS timestamp,
                node.is_sidechain AS is_sidechain, score
         LIMIT $limit"
    } else {
        "CALL db.index.fulltext.queryNodes('turn_fulltext', $query)
         YIELD node, score
         RETURN node.turn_id AS turn_id, node.session_id AS session_id,
                node.namespace AS namespace, node.role AS role, node.text AS text,
                node.model AS model, node.timestamp AS timestamp,
                node.is_sidechain AS is_sidechain, score
         LIMIT $limit"
    };

    if let Ok(mut result) = graph
        .execute(
            query(ft_cypher)
                .param("query", ft_query.as_str())
                .param("namespaces", ns.clone())
                .param("limit", (limit * 2) as i64),
        )
        .await
    {
        while let Ok(Some(row)) = result.next().await {
            let turn_id: String = row.get("turn_id").unwrap_or_default();
            let score: f64 = row.get("score").unwrap_or(0.0);
            let turn = Turn {
                turn_id: turn_id.clone(),
                session_id: row.get("session_id").unwrap_or_default(),
                namespace: row.get("namespace").unwrap_or_default(),
                role: row.get("role").unwrap_or_default(),
                text: row.get("text").unwrap_or_default(),
                model: row.get::<String>("model").ok().filter(|s| !s.is_empty()),
                timestamp: row.get("timestamp").unwrap_or_default(),
                is_sidechain: row.get("is_sidechain").unwrap_or(false),
                ..Turn::default()
            };
            results.insert(turn_id, (turn, score));
        }
    }

    if let Some(emb) = embedding {
        let embedding_vec: Vec<f64> = emb.iter().map(|&x| f64::from(x)).collect();

        let vec_cypher = if filter_ns {
            "CALL db.index.vector.queryNodes('turn_embedding', $limit, $embedding)
             YIELD node, score
             WHERE node.namespace IN $namespaces
             RETURN node.turn_id AS turn_id, node.session_id AS session_id,
                    node.namespace AS namespace, node.role AS role, node.text AS text,
                    node.model AS model, node.timestamp AS timestamp,
                    node.is_sidechain AS is_sidechain, score"
        } else {
            "CALL db.index.vector.queryNodes('turn_embedding', $limit, $embedding)
             YIELD node, score
             RETURN node.turn_id AS turn_id, node.session_id AS session_id,
                    node.namespace AS namespace, node.role AS role, node.text AS text,
                    node.model AS model, node.timestamp AS timestamp,
                    node.is_sidechain AS is_sidechain, score"
        };

        if let Ok(mut result) = graph
            .execute(
                query(vec_cypher)
                    .param("embedding", embedding_vec)
                    .param("namespaces", ns)
                    .param("limit", (limit * 2) as i64),
            )
            .await
        {
            let ft_count = results.len();
            while let Ok(Some(row)) = result.next().await {
                let turn_id: String = row.get("turn_id").unwrap_or_default();
                let vec_score: f64 = row.get("score").unwrap_or(0.0);

                if let Some(entry) = results.get_mut(&turn_id) {
                    entry.1 = (entry.1 + vec_score) / 2.0;
                } else {
                    let turn = Turn {
                        turn_id: turn_id.clone(),
                        session_id: row.get("session_id").unwrap_or_default(),
                        namespace: row.get("namespace").unwrap_or_default(),
                        role: row.get("role").unwrap_or_default(),
                        text: row.get("text").unwrap_or_default(),
                        model: row.get::<String>("model").ok().filter(|s| !s.is_empty()),
                        timestamp: row.get("timestamp").unwrap_or_default(),
                        is_sidechain: row.get("is_sidechain").unwrap_or(false),
                        ..Turn::default()
                    };
                    let boost = if ft_count > 0 {
                        vec_score * 0.8
                    } else {
                        vec_score
                    };
                    results.insert(turn_id, (turn, boost));
                }
            }
        }
    }

    let mut out: Vec<(Turn, f64)> = results.into_values().collect();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    out.truncate(limit);
    Ok(out)
}

#[cfg(feature = "sessions")]
pub async fn search_reflections_hybrid(
    graph: &Graph,
    query_text: &str,
    embedding: Option<&[f32]>,
    limit: usize,
    namespaces: &[String],
) -> Result<Vec<(Reflection, f64)>> {
    let ns: Vec<&str> = namespaces.iter().map(String::as_str).collect();
    let filter_ns = !ns.is_empty();

    let mut results: HashMap<String, (Reflection, f64)> = HashMap::new();
    let escaped = query_text.replace('\\', "\\\\").replace('"', "\\\"");
    let ft_query = format!("{escaped}~");

    let ft_cypher = if filter_ns {
        "CALL db.index.fulltext.queryNodes('reflection_fulltext', $query)
         YIELD node, score
         WHERE node.namespace IN $namespaces
         RETURN node.reflection_id AS reflection_id, node.turn_id AS turn_id,
                node.session_id AS session_id, node.namespace AS namespace,
                node.text AS text, node.timestamp AS timestamp, score
         LIMIT $limit"
    } else {
        "CALL db.index.fulltext.queryNodes('reflection_fulltext', $query)
         YIELD node, score
         RETURN node.reflection_id AS reflection_id, node.turn_id AS turn_id,
                node.session_id AS session_id, node.namespace AS namespace,
                node.text AS text, node.timestamp AS timestamp, score
         LIMIT $limit"
    };

    if let Ok(mut result) = graph
        .execute(
            query(ft_cypher)
                .param("query", ft_query.as_str())
                .param("namespaces", ns.clone())
                .param("limit", (limit * 2) as i64),
        )
        .await
    {
        while let Ok(Some(row)) = result.next().await {
            let rid: String = row.get("reflection_id").unwrap_or_default();
            let score: f64 = row.get("score").unwrap_or(0.0);
            let r = Reflection {
                reflection_id: rid.clone(),
                turn_id: row.get("turn_id").unwrap_or_default(),
                session_id: row.get("session_id").unwrap_or_default(),
                namespace: row.get("namespace").unwrap_or_default(),
                text: row.get("text").unwrap_or_default(),
                signature: None,
                timestamp: row.get("timestamp").unwrap_or_default(),
            };
            results.insert(rid, (r, score));
        }
    }

    if let Some(emb) = embedding {
        let embedding_vec: Vec<f64> = emb.iter().map(|&x| f64::from(x)).collect();

        let vec_cypher = if filter_ns {
            "CALL db.index.vector.queryNodes('reflection_embedding', $limit, $embedding)
             YIELD node, score
             WHERE node.namespace IN $namespaces
             RETURN node.reflection_id AS reflection_id, node.turn_id AS turn_id,
                    node.session_id AS session_id, node.namespace AS namespace,
                    node.text AS text, node.timestamp AS timestamp, score"
        } else {
            "CALL db.index.vector.queryNodes('reflection_embedding', $limit, $embedding)
             YIELD node, score
             RETURN node.reflection_id AS reflection_id, node.turn_id AS turn_id,
                    node.session_id AS session_id, node.namespace AS namespace,
                    node.text AS text, node.timestamp AS timestamp, score"
        };

        if let Ok(mut result) = graph
            .execute(
                query(vec_cypher)
                    .param("embedding", embedding_vec)
                    .param("namespaces", ns)
                    .param("limit", (limit * 2) as i64),
            )
            .await
        {
            let ft_count = results.len();
            while let Ok(Some(row)) = result.next().await {
                let rid: String = row.get("reflection_id").unwrap_or_default();
                let vec_score: f64 = row.get("score").unwrap_or(0.0);

                if let Some(entry) = results.get_mut(&rid) {
                    entry.1 = (entry.1 + vec_score) / 2.0;
                } else {
                    let r = Reflection {
                        reflection_id: rid.clone(),
                        turn_id: row.get("turn_id").unwrap_or_default(),
                        session_id: row.get("session_id").unwrap_or_default(),
                        namespace: row.get("namespace").unwrap_or_default(),
                        text: row.get("text").unwrap_or_default(),
                        signature: None,
                        timestamp: row.get("timestamp").unwrap_or_default(),
                    };
                    let boost = if ft_count > 0 {
                        vec_score * 0.8
                    } else {
                        vec_score
                    };
                    results.insert(rid, (r, boost));
                }
            }
        }
    }

    let mut out: Vec<(Reflection, f64)> = results.into_values().collect();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    out.truncate(limit);
    Ok(out)
}

#[cfg(feature = "sessions")]
pub async fn ensure_turn_indexes(graph: &Graph) -> Result<()> {
    graph
        .run(query(
            "CREATE CONSTRAINT turn_id_unique IF NOT EXISTS
         FOR (t:Turn) REQUIRE t.turn_id IS UNIQUE",
        ))
        .await?;

    graph
        .run(query(
            "CREATE CONSTRAINT reflection_id_unique IF NOT EXISTS
         FOR (r:Reflection) REQUIRE r.reflection_id IS UNIQUE",
        ))
        .await?;

    graph
        .run(query(
            "CREATE CONSTRAINT file_path_unique IF NOT EXISTS
         FOR (f:File) REQUIRE f.path IS UNIQUE",
        ))
        .await?;

    graph
        .run(query(
            "CREATE CONSTRAINT command_cmd_unique IF NOT EXISTS
         FOR (c:Command) REQUIRE c.cmd IS UNIQUE",
        ))
        .await?;

    graph
        .run(query(
            "CREATE CONSTRAINT toolcall_id_unique IF NOT EXISTS
         FOR (tc:ToolCall) REQUIRE tc.tool_call_id IS UNIQUE",
        ))
        .await?;

    graph
        .run(query(
            "CREATE VECTOR INDEX turn_embedding IF NOT EXISTS
         FOR (t:Turn)
         ON t.embedding
         OPTIONS {indexConfig: {
           `vector.dimensions`: 768,
           `vector.similarity_function`: 'cosine'
         }}",
        ))
        .await?;

    graph
        .run(query(
            "CREATE VECTOR INDEX reflection_embedding IF NOT EXISTS
         FOR (r:Reflection)
         ON r.embedding
         OPTIONS {indexConfig: {
           `vector.dimensions`: 768,
           `vector.similarity_function`: 'cosine'
         }}",
        ))
        .await?;

    graph
        .run(query(
            "CREATE FULLTEXT INDEX turn_fulltext IF NOT EXISTS
         FOR (t:Turn) ON EACH [t.text]",
        ))
        .await?;

    graph
        .run(query(
            "CREATE FULLTEXT INDEX reflection_fulltext IF NOT EXISTS
         FOR (r:Reflection) ON EACH [r.text]",
        ))
        .await?;

    graph
        .run(query(
            "CREATE FULLTEXT INDEX command_fulltext IF NOT EXISTS
         FOR (c:Command) ON EACH [c.cmd]",
        ))
        .await?;

    graph
        .run(query(
            "CREATE INDEX turn_timestamp IF NOT EXISTS
         FOR (t:Turn) ON (t.timestamp)",
        ))
        .await?;

    graph
        .run(query(
            "CREATE INDEX turn_session_id IF NOT EXISTS
         FOR (t:Turn) ON (t.session_id)",
        ))
        .await?;

    Ok(())
}

#[cfg(feature = "sessions")]
pub async fn get_unenriched_sessions(
    graph: &Graph,
    namespaces: &[String],
    limit: usize,
) -> Result<Vec<String>> {
    let ns: Vec<&str> = namespaces.iter().map(String::as_str).collect();
    let filter_ns = !ns.is_empty();

    let cypher = if filter_ns {
        "MATCH (s:Session)
         WHERE s.namespace IN $namespaces
           AND s.deep_indexed_at IS NOT NULL
           AND (s.enriched_at IS NULL OR s.enriched_at < s.deep_indexed_at)
         RETURN s.session_id AS session_id
         ORDER BY s.deep_indexed_at DESC
         LIMIT $limit"
    } else {
        "MATCH (s:Session)
         WHERE s.deep_indexed_at IS NOT NULL
           AND (s.enriched_at IS NULL OR s.enriched_at < s.deep_indexed_at)
         RETURN s.session_id AS session_id
         ORDER BY s.deep_indexed_at DESC
         LIMIT $limit"
    };

    let mut result = graph
        .execute(
            query(cypher)
                .param("namespaces", ns)
                .param("limit", limit as i64),
        )
        .await?;

    let mut ids = Vec::new();
    while let Some(row) = result.next().await? {
        if let Ok(id) = row.get::<String>("session_id") {
            ids.push(id);
        }
    }
    Ok(ids)
}

#[cfg(feature = "sessions")]
pub async fn get_session_text_for_enrichment(
    graph: &Graph,
    session_id: &str,
    include_reflections: bool,
    max_chars: usize,
) -> Result<String> {
    let mut buf = String::new();

    let mut turn_result = graph
        .execute(
            query(
                "MATCH (t:Turn {session_id: $session_id})
             WHERE t.text IS NOT NULL AND t.text <> ''
             RETURN t.role AS role, t.text AS text, t.timestamp AS ts
             ORDER BY t.timestamp ASC",
            )
            .param("session_id", session_id),
        )
        .await?;

    while let Some(row) = turn_result.next().await? {
        if buf.len() >= max_chars {
            break;
        }
        let role: String = row.get("role").unwrap_or_default();
        let text: String = row.get("text").unwrap_or_default();
        if !buf.is_empty() {
            buf.push_str("\n---\n");
        }
        buf.push_str(&format!("[{role}]\n{text}"));
    }

    if include_reflections && buf.len() < max_chars {
        let mut refl_result = graph
            .execute(
                query(
                    "MATCH (r:Reflection {session_id: $session_id})
                 WHERE r.text IS NOT NULL AND r.text <> ''
                 RETURN r.text AS text
                 ORDER BY r.timestamp ASC",
                )
                .param("session_id", session_id),
            )
            .await?;

        while let Some(row) = refl_result.next().await? {
            if buf.len() >= max_chars {
                break;
            }
            let text: String = row.get("text").unwrap_or_default();
            if !buf.is_empty() {
                buf.push_str("\n---\n");
            }
            buf.push_str(&format!("[thinking]\n{text}"));
        }
    }

    if buf.len() > max_chars {
        let mut end = max_chars;
        while !buf.is_char_boundary(end) && end > 0 {
            end -= 1;
        }
        buf.truncate(end);
    }

    Ok(buf)
}

#[cfg(feature = "sessions")]
pub async fn link_concept_to_session(
    graph: &Graph,
    concept_name: &str,
    namespace: &str,
    session_id: &str,
    count: i64,
) -> Result<()> {
    graph
        .run(
            query(
                "MATCH (c:Concept {name: $name, namespace: $namespace})
             MATCH (s:Session {session_id: $session_id})
             MERGE (c)-[r:MENTIONED_IN]->(s)
             ON CREATE SET r.count = $count, r.first_mentioned_at = datetime()
             ON MATCH SET r.count = $count, r.last_mentioned_at = datetime()",
            )
            .param("name", concept_name)
            .param("namespace", namespace)
            .param("session_id", session_id)
            .param("count", count),
        )
        .await?;
    Ok(())
}

pub async fn get_sessions_for_concept(
    graph: &Graph,
    concept_name: &str,
    namespaces: &[String],
    limit: usize,
) -> Result<Vec<(Session, i64)>> {
    let ns: Vec<&str> = namespaces.iter().map(String::as_str).collect();
    let filter_ns = !ns.is_empty();

    let cypher = if filter_ns {
        "MATCH (c:Concept {name: $name})-[r:MENTIONED_IN]->(s:Session)
         WHERE c.namespace IN $namespaces
         RETURN s.session_id AS session_id, s.slug AS slug, s.cwd AS cwd,
                s.namespace AS namespace, s.first_prompt AS first_prompt,
                s.summary AS summary, s.git_branch AS git_branch,
                s.created_at AS created_at, s.ended_at AS ended_at,
                s.message_count AS message_count, s.is_sidechain AS is_sidechain,
                r.count AS mention_count
         ORDER BY s.created_at DESC
         LIMIT $limit"
    } else {
        "MATCH (c:Concept {name: $name})-[r:MENTIONED_IN]->(s:Session)
         RETURN s.session_id AS session_id, s.slug AS slug, s.cwd AS cwd,
                s.namespace AS namespace, s.first_prompt AS first_prompt,
                s.summary AS summary, s.git_branch AS git_branch,
                s.created_at AS created_at, s.ended_at AS ended_at,
                s.message_count AS message_count, s.is_sidechain AS is_sidechain,
                r.count AS mention_count
         ORDER BY s.created_at DESC
         LIMIT $limit"
    };

    let mut result = graph
        .execute(
            query(cypher)
                .param("name", concept_name)
                .param("namespaces", ns)
                .param("limit", limit as i64),
        )
        .await?;

    let mut out = Vec::new();
    while let Some(row) = result.next().await? {
        let session = Session {
            session_id: row.get("session_id").unwrap_or_default(),
            slug: row.get::<String>("slug").ok().filter(|s| !s.is_empty()),
            cwd: row.get("cwd").unwrap_or_default(),
            namespace: row.get("namespace").unwrap_or_default(),
            first_prompt: row.get("first_prompt").unwrap_or_default(),
            summary: row.get::<String>("summary").ok().filter(|s| !s.is_empty()),
            git_branch: row
                .get::<String>("git_branch")
                .ok()
                .filter(|s| !s.is_empty()),
            created_at: row.get("created_at").unwrap_or_default(),
            ended_at: row.get::<String>("ended_at").ok().filter(|s| !s.is_empty()),
            message_count: row.get("message_count").ok(),
            is_sidechain: row.get("is_sidechain").unwrap_or(false),
        };
        let count: i64 = row.get("mention_count").unwrap_or(1);
        out.push((session, count));
    }
    Ok(out)
}

#[cfg(feature = "sessions")]
pub async fn mark_session_enriched(
    graph: &Graph,
    session_id: &str,
    concept_count: i64,
) -> Result<()> {
    graph
        .run(
            query(
                "MATCH (s:Session {session_id: $session_id})
             SET s.enriched_at = datetime(),
                 s.concept_mention_count = $concept_count",
            )
            .param("session_id", session_id)
            .param("concept_count", concept_count),
        )
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(name: &str) -> SearchResult {
        SearchResult {
            name: name.to_string(),
            namespace: "personal".to_string(),
            description: None,
            similarity: 0.0,
        }
    }

    // Regression for issue #18: a verbatim name match must score at the top of
    // the [0,1] range, not in the RRF noise band (~0.0164).
    #[test]
    fn exact_name_match_scores_one() {
        let query = "Shopify content_for blocks needs block_order";
        let keyword = vec![result("Figma MCP tool-call quota"), result(query)];
        // Embedding ranks the exact node well below semantic noise.
        let vector = vec![
            result("Figma MCP tool-call quota"),
            result("c0 durable fact criteria"),
            result(query),
        ];

        let fused = reciprocal_rank_fusion(query, &keyword, &vector, 0.4, 60.0);

        assert_eq!(fused[0].name, query);
        assert!(
            (fused[0].similarity - 1.0).abs() < 1e-6,
            "exact match should score 1.0, got {}",
            fused[0].similarity
        );
    }

    #[test]
    fn rank_one_in_both_normalizes_to_one() {
        let keyword = vec![result("alpha"), result("beta")];
        let vector = vec![result("alpha"), result("gamma")];

        let fused = reciprocal_rank_fusion("unrelated query", &keyword, &vector, 0.4, 60.0);

        let alpha = fused.iter().find(|r| r.name == "alpha").unwrap();
        assert!(
            (alpha.similarity - 1.0).abs() < 1e-6,
            "rank-1 in both lists should normalize to 1.0, got {}",
            alpha.similarity
        );
    }

    #[test]
    fn single_list_hits_reflect_alpha_weighting() {
        // "kw" only appears in keyword results, "vec" only in vector results.
        let keyword = vec![result("kw")];
        let vector = vec![result("vec")];

        let fused = reciprocal_rank_fusion("unrelated query", &keyword, &vector, 0.4, 60.0);

        let kw = fused.iter().find(|r| r.name == "kw").unwrap();
        let vec = fused.iter().find(|r| r.name == "vec").unwrap();

        // keyword-only-#1 -> alpha, vector-only-#1 -> (1 - alpha).
        assert!((kw.similarity - 0.4).abs() < 1e-6, "got {}", kw.similarity);
        assert!(
            (vec.similarity - 0.6).abs() < 1e-6,
            "got {}",
            vec.similarity
        );
        // And the noise floor is well-separated from a perfect hit.
        assert!(vec.similarity < 1.0 && kw.similarity < 1.0);
    }

    #[test]
    fn scores_are_clamped_to_one() {
        let fused = reciprocal_rank_fusion("q", &[result("alpha")], &[result("alpha")], 0.4, 60.0);
        assert!(fused.iter().all(|r| r.similarity <= 1.0));
    }
}
