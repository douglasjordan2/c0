use anyhow::Result;
use neo4rs::Graph;
use serde::Serialize;

use crate::graph;

const LABELS: &[&str] = &[
    "Concept",
    "KnowledgePatch",
    "SemanticTrigger",
    "LiveSource",
    "Event",
];

#[derive(Debug, Serialize)]
pub struct ExportMetadata {
    pub exported_at: String,
    pub namespace_filter: Option<String>,
    pub node_count: usize,
    pub relationship_count: usize,
}

#[derive(Debug, Serialize)]
pub struct GraphExport {
    pub metadata: ExportMetadata,
    pub nodes: Vec<serde_json::Value>,
    pub relationships: Vec<serde_json::Value>,
}

pub async fn export_graph(
    graph_conn: &Graph,
    namespace_filter: Option<&str>,
    no_embeddings: bool,
) -> Result<GraphExport> {
    let ns_vec: Option<Vec<String>> =
        namespace_filter.map(|ns| vec![ns.to_string(), "global".to_string()]);
    let ns_slice = ns_vec.as_deref();

    let mut all_nodes = Vec::new();
    for label in LABELS {
        let nodes =
            graph::export_nodes_by_label(graph_conn, label, ns_slice, no_embeddings).await?;
        all_nodes.extend(nodes);
    }

    let relationships = graph::export_all_relationships(graph_conn, ns_slice).await?;

    let metadata = ExportMetadata {
        exported_at: chrono::Utc::now().to_rfc3339(),
        namespace_filter: namespace_filter.map(String::from),
        node_count: all_nodes.len(),
        relationship_count: relationships.len(),
    };

    Ok(GraphExport {
        metadata,
        nodes: all_nodes,
        relationships,
    })
}

pub fn format_json(export: &GraphExport) -> Result<String> {
    Ok(serde_json::to_string_pretty(export)?)
}

fn escape_cypher(s: &str) -> String {
    s.replace('\\', "\\\\").replace('\'', "\\'")
}

fn props_to_cypher(props: &serde_json::Value) -> String {
    let obj = match props.as_object() {
        Some(o) => o,
        None => return "{}".to_string(),
    };

    let parts: Vec<String> = obj
        .iter()
        .filter(|(k, _)| k.as_str() != "embedding")
        .map(|(k, v)| {
            let val_str = match v {
                serde_json::Value::String(s) => format!("'{}'", escape_cypher(s)),
                serde_json::Value::Number(n) => n.to_string(),
                serde_json::Value::Bool(b) => b.to_string(),
                serde_json::Value::Null => "null".to_string(),
                _ => format!("'{}'", escape_cypher(&v.to_string())),
            };
            format!("{k}: {val_str}")
        })
        .collect();

    format!("{{{}}}", parts.join(", "))
}

pub fn format_cypher(export: &GraphExport) -> String {
    let mut lines = Vec::new();
    lines.push(format!(
        "// c0 graph export - {}",
        export.metadata.exported_at
    ));
    lines.push(format!(
        "// {} nodes, {} relationships",
        export.metadata.node_count, export.metadata.relationship_count
    ));
    if let Some(ref ns) = export.metadata.namespace_filter {
        lines.push(format!("// namespace filter: {ns}"));
    }
    lines.push(String::new());

    for node in &export.nodes {
        let labels = node
            .get("labels")
            .and_then(|l| l.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .collect::<Vec<_>>()
                    .join(":")
            })
            .unwrap_or_else(|| "Node".to_string());

        let props = node
            .get("properties")
            .cloned()
            .unwrap_or(serde_json::json!({}));

        lines.push(format!("CREATE (:{labels} {});", props_to_cypher(&props)));
    }

    if !export.relationships.is_empty() {
        lines.push(String::new());
    }

    for rel in &export.relationships {
        let start = rel
            .get("start_name")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        let end = rel.get("end_name").and_then(|v| v.as_str()).unwrap_or("?");
        let rel_type = rel
            .get("rel_type")
            .and_then(|v| v.as_str())
            .unwrap_or("RELATED_TO");

        let start_label = rel
            .get("start_labels")
            .and_then(|l| l.as_array())
            .and_then(|arr| arr.first())
            .and_then(|v| v.as_str())
            .unwrap_or("Node");

        let end_label = rel
            .get("end_labels")
            .and_then(|l| l.as_array())
            .and_then(|arr| arr.first())
            .and_then(|v| v.as_str())
            .unwrap_or("Node");

        lines.push(format!(
            "MATCH (a:{start_label} {{name: '{}'}}), (b:{end_label} {{name: '{}'}}) CREATE (a)-[:{rel_type}]->(b);",
            escape_cypher(start),
            escape_cypher(end),
        ));
    }

    lines.join("\n")
}
