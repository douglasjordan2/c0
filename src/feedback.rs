use anyhow::Result;
use serde_json::{Value, json};
use std::fs;
use std::path::PathBuf;

use crate::reflector;

const LAST_WALK_FILE: &str = "last-walk.json";
const MAX_WALK_NODES: usize = 20;

pub struct WalkRecord<'a> {
    pub query: &'a str,
    pub namespace: &'a str,
    pub entry: &'a str,
    pub resolved_via: &'a str,
    pub patches: usize,
    pub nodes: &'a [(String, f32)],
}

fn claude_session() -> Option<String> {
    std::env::var("CLAUDE_CODE_SESSION_ID")
        .ok()
        .filter(|s| !s.trim().is_empty())
}

fn is_safe_component(key: &str) -> bool {
    !key.is_empty() && key != "." && key != ".." && !key.contains(['/', '\\'])
}

fn last_walk_path(c0_session: Option<&str>) -> Option<PathBuf> {
    let key = claude_session().or_else(|| c0_session.map(String::from))?;
    if !is_safe_component(&key) {
        return None;
    }
    Some(
        dirs::home_dir()?
            .join(".c0/sessions")
            .join(key)
            .join(LAST_WALK_FILE),
    )
}

fn walk_json(record: &WalkRecord, timestamp: &str) -> Value {
    let nodes: Vec<Value> = record
        .nodes
        .iter()
        .take(MAX_WALK_NODES)
        .map(|(name, score)| json!({ "name": name, "score": score }))
        .collect();
    json!({
        "timestamp": timestamp,
        "query": record.query,
        "namespace": record.namespace,
        "entry": record.entry,
        "resolved_via": record.resolved_via,
        "dead_end": record.nodes.is_empty(),
        "patches": record.patches,
        "node_count": record.nodes.len(),
        "nodes": nodes,
    })
}

pub fn record_walk(c0_session: Option<&str>, record: &WalkRecord) {
    let Some(path) = last_walk_path(c0_session) else {
        return;
    };
    if let Some(parent) = path.parent()
        && fs::create_dir_all(parent).is_err()
    {
        return;
    }
    let value = walk_json(record, &chrono::Utc::now().to_rfc3339());
    let _ = fs::write(path, value.to_string());
}

fn load_last_walk(c0_session: Option<&str>) -> Option<Value> {
    let path = last_walk_path(c0_session)?;
    serde_json::from_str(&fs::read_to_string(path).ok()?).ok()
}

fn build_entry(
    message: &str,
    proposed: Option<&str>,
    namespace: &str,
    c0_session: Option<&str>,
    claude_session: Option<&str>,
    last_walk: Option<&Value>,
    timestamp: &str,
) -> Value {
    json!({
        "timestamp": timestamp,
        "session": c0_session,
        "claude_session": claude_session,
        "namespace": namespace,
        "cwd": std::env::current_dir().ok().map(|d| d.display().to_string()),
        "c0_version": env!("CARGO_PKG_VERSION"),
        "c0_sha": option_env!("C0_GIT_SHA").unwrap_or("unknown"),
        "message": message,
        "proposed": proposed,
        "last_walk": last_walk,
    })
}

fn field<'a>(entry: &'a Value, key: &str) -> Option<&'a str> {
    entry.get(key).and_then(Value::as_str)
}

fn same_report(existing: &str, entry: &Value) -> bool {
    let Ok(existing) = serde_json::from_str::<Value>(existing) else {
        return false;
    };
    let normalized = |v: &Value| field(v, "message").map(|m| m.trim().to_lowercase());
    normalized(&existing) == normalized(entry)
        && field(&existing, "session") == field(entry, "session")
        && field(&existing, "claude_session") == field(entry, "claude_session")
}

pub fn submit(
    message: &str,
    proposed: Option<&str>,
    namespace: &str,
    c0_session: Option<&str>,
) -> Result<()> {
    let message = message.trim();
    if message.is_empty() {
        anyhow::bail!("feedback message is empty");
    }
    let proposed = proposed.map(str::trim).filter(|p| !p.is_empty());
    let claude = claude_session();
    let last_walk = load_last_walk(c0_session);
    let attached = last_walk
        .as_ref()
        .and_then(|w| field(w, "query"))
        .map(String::from);

    let entry = build_entry(
        message,
        proposed,
        namespace,
        c0_session,
        claude.as_deref(),
        last_walk.as_ref(),
        &chrono::Utc::now().to_rfc3339(),
    );

    let added = reflector::append_feedback(&entry.to_string(), |line| same_report(line, &entry))?;
    if !added {
        println!("Already recorded this feedback for this session.");
        return Ok(());
    }

    match attached {
        Some(query) => println!("Feedback recorded (attached last walk: '{query}')."),
        None => println!("Feedback recorded (no walk context for this session)."),
    }
    Ok(())
}

fn describe_walk(walk: &Value) -> Option<String> {
    let query = field(walk, "query")?;
    let entry = field(walk, "entry").unwrap_or("?");
    let via = field(walk, "resolved_via").unwrap_or("?");
    if walk.get("dead_end").and_then(Value::as_bool) == Some(true) {
        return Some(format!("'{query}' -> dead end"));
    }
    let names: Vec<&str> = walk
        .get("nodes")
        .and_then(Value::as_array)
        .map(|nodes| nodes.iter().filter_map(|n| field(n, "name")).collect())
        .unwrap_or_default();
    Some(format!(
        "'{query}' -> '{entry}' via {via}: {}",
        names.join(", ")
    ))
}

pub fn list(as_json: bool) -> Result<()> {
    let content = reflector::read_feedback()?;
    let lines: Vec<&str> = content.lines().filter(|l| !l.trim().is_empty()).collect();

    if as_json {
        for line in &lines {
            println!("{line}");
        }
        return Ok(());
    }

    if lines.is_empty() {
        println!("No feedback recorded.");
        return Ok(());
    }

    println!("{} feedback report(s)", lines.len());
    println!("═══════════════════════════════════════");
    for line in lines {
        let Ok(entry) = serde_json::from_str::<Value>(line) else {
            println!("(unparseable entry)");
            continue;
        };
        let when: String = field(&entry, "timestamp")
            .unwrap_or("?")
            .chars()
            .take(16)
            .collect();
        println!(
            "[{}] {} (c0 {})",
            when.replace('T', " "),
            field(&entry, "namespace").unwrap_or("?"),
            field(&entry, "c0_sha").unwrap_or("unknown"),
        );
        println!("  {}", field(&entry, "message").unwrap_or(""));
        if let Some(proposed) = field(&entry, "proposed") {
            println!("  proposed: {proposed}");
        }
        if let Some(walk) = entry.get("last_walk").and_then(describe_walk) {
            println!("  last walk: {walk}");
        }
        if let Some(session) = field(&entry, "claude_session") {
            println!("  resume: claude --resume {session}");
        }
        println!();
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn entry(message: &str, session: &str) -> Value {
        build_entry(
            message,
            None,
            "c0",
            Some(session),
            Some("claude-1"),
            None,
            "2026-11-18T00:00:00Z",
        )
    }

    #[test]
    fn same_report_matches_on_message_and_session_only() {
        let first = entry("Walk landed on the wrong concept", "s1");
        let line = first.to_string();

        assert!(same_report(
            &line,
            &entry("  walk landed on the WRONG concept ", "s1")
        ));
        assert!(!same_report(
            &line,
            &entry("Walk landed on the wrong concept", "s2")
        ));
        assert!(!same_report(&line, &entry("Different complaint", "s1")));
        assert!(!same_report("not json", &first));
    }

    #[test]
    fn walk_json_caps_nodes_but_keeps_the_true_count() {
        let nodes: Vec<(String, f32)> = (0..MAX_WALK_NODES + 5)
            .map(|i| (format!("n{i}"), 0.5))
            .collect();
        let value = walk_json(
            &WalkRecord {
                query: "slack",
                namespace: "c0",
                entry: "slack",
                resolved_via: "exact",
                patches: 2,
                nodes: &nodes,
            },
            "2026-11-18T00:00:00Z",
        );

        assert_eq!(value["nodes"].as_array().unwrap().len(), MAX_WALK_NODES);
        assert_eq!(value["node_count"], MAX_WALK_NODES + 5);
        assert_eq!(value["dead_end"], false);
        assert_eq!(value["resolved_via"], "exact");
    }

    #[test]
    fn walk_json_marks_an_empty_walk_as_a_dead_end() {
        let value = walk_json(
            &WalkRecord {
                query: "nope",
                namespace: "c0",
                entry: "nope",
                resolved_via: "exact",
                patches: 0,
                nodes: &[],
            },
            "2026-11-18T00:00:00Z",
        );
        assert_eq!(value["dead_end"], true);
    }

    #[test]
    fn describe_walk_summarizes_entry_tier_and_nodes() {
        let walk = json!({
            "query": "slack",
            "entry": "slack",
            "resolved_via": "exact",
            "dead_end": false,
            "nodes": [{ "name": "wl-copy", "score": 0.46 }, { "name": "slack-oauth", "score": 0.6 }],
        });
        assert_eq!(
            describe_walk(&walk).unwrap(),
            "'slack' -> 'slack' via exact: wl-copy, slack-oauth"
        );

        let dead = json!({ "query": "nope", "dead_end": true });
        assert_eq!(describe_walk(&dead).unwrap(), "'nope' -> dead end");
    }

    #[test]
    fn session_keys_cannot_escape_the_sessions_dir() {
        assert!(is_safe_component("0873b5f3-b670"));
        assert!(!is_safe_component("../etc"));
        assert!(!is_safe_component(".."));
        assert!(!is_safe_component(""));
    }
}
