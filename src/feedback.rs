use anyhow::Result;
use serde_json::{Value, json};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use crate::reflector;

const LAST_WALK_FILE: &str = "last-walk.json";
const MAX_WALK_NODES: usize = 20;
pub(crate) const PROCESSING_FILE: &str = "feedback.processing.jsonl";
const RESOLVED_FILE: &str = "feedback.resolved.jsonl";
pub const OUTCOMES: [&str; 7] = [
    "pr",
    "issue",
    "review",
    "fixed",
    "duplicate",
    "wontfix",
    "not-reproducible",
];

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
        "id": short_hash(&format!("{timestamp}\n{message}")),
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

fn short_hash(text: &str) -> String {
    let mut hash: u32 = 0x811c_9dc5;
    for byte in text.bytes() {
        hash ^= u32::from(byte);
        hash = hash.wrapping_mul(0x0100_0193);
    }
    format!("{hash:08x}")
}

fn ensure_id(line: &str, taken: &mut HashSet<String>) -> String {
    let Ok(mut entry) = serde_json::from_str::<Value>(line) else {
        return line.to_string();
    };
    let existing = field(&entry, "id").map(String::from);
    let base = existing.clone().unwrap_or_else(|| short_hash(line));
    let mut id = base.clone();
    let mut n = 2;
    while !taken.insert(id.clone()) {
        id = format!("{base}-{n}");
        n += 1;
    }
    if existing.as_deref() == Some(id.as_str()) {
        return line.to_string();
    }
    if let Some(map) = entry.as_object_mut() {
        map.insert("id".to_string(), Value::String(id));
    }
    entry.to_string()
}

fn claim_in(dir: &Path) -> Result<Vec<String>> {
    let _lock = reflector::lock_dir(dir)?;
    let inbox = dir.join(reflector::FEEDBACK_FILE);
    let processing = dir.join(PROCESSING_FILE);

    let claimed = reflector::read_or_empty(&processing)?;
    let mut taken: HashSet<String> = reflector::non_empty_lines(&claimed)
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .filter_map(|e| field(&e, "id").map(String::from))
        .collect();

    let incoming = reflector::read_or_empty(&inbox)?;
    let incoming: Vec<String> = reflector::non_empty_lines(&incoming)
        .map(|l| ensure_id(l, &mut taken))
        .collect();
    if !incoming.is_empty() {
        reflector::append_unlocked(&processing, &incoming)?;
        fs::write(&inbox, "")?;
    }

    Ok(
        reflector::non_empty_lines(&reflector::read_or_empty(&processing)?)
            .map(String::from)
            .collect(),
    )
}

pub fn claim() -> Result<()> {
    let lines = claim_in(&reflector::reflector_dir())?;
    if lines.is_empty() {
        eprintln!("No open feedback.");
    }
    for line in &lines {
        println!("{line}");
    }
    Ok(())
}

pub struct Resolution<'a> {
    pub outcome: &'a str,
    pub reference: Option<&'a str>,
    pub note: Option<&'a str>,
}

fn resolve_in(
    dir: &Path,
    ids: &[String],
    resolution: &Resolution,
    timestamp: &str,
) -> Result<usize> {
    if !OUTCOMES.contains(&resolution.outcome) {
        anyhow::bail!(
            "unknown outcome '{}' (use one of: {})",
            resolution.outcome,
            OUTCOMES.join(", ")
        );
    }
    let _lock = reflector::lock_dir(dir)?;
    let processing = dir.join(PROCESSING_FILE);
    let current = reflector::read_or_empty(&processing)?;

    let wanted: HashSet<&str> = ids.iter().map(String::as_str).collect();
    let mut found: HashSet<String> = HashSet::new();
    let mut kept: Vec<&str> = Vec::new();
    let mut resolved: Vec<String> = Vec::new();

    for line in reflector::non_empty_lines(&current) {
        let entry = serde_json::from_str::<Value>(line).ok();
        let id = entry
            .as_ref()
            .and_then(|e| field(e, "id"))
            .map(String::from);
        match (entry, id) {
            (Some(mut entry), Some(id)) if wanted.contains(id.as_str()) => {
                if let Some(map) = entry.as_object_mut() {
                    map.insert(
                        "resolution".to_string(),
                        json!({
                            "outcome": resolution.outcome,
                            "ref": resolution.reference,
                            "note": resolution.note,
                            "resolved_at": timestamp,
                        }),
                    );
                }
                found.insert(id);
                resolved.push(entry.to_string());
            }
            _ => kept.push(line),
        }
    }

    let mut missing: Vec<&str> = wanted
        .iter()
        .copied()
        .filter(|id| !found.contains(*id))
        .collect();
    if !missing.is_empty() {
        missing.sort_unstable();
        anyhow::bail!(
            "no claimed feedback with id(s): {} (run `c0 feedback claim` first); nothing resolved",
            missing.join(", ")
        );
    }

    reflector::append_unlocked(&dir.join(RESOLVED_FILE), &resolved)?;
    reflector::write_atomic(&processing, &reflector::join_lines(&kept))?;
    Ok(resolved.len())
}

pub fn resolve(ids: &[String], resolution: &Resolution) -> Result<()> {
    let count = resolve_in(
        &reflector::reflector_dir(),
        ids,
        resolution,
        &chrono::Utc::now().to_rfc3339(),
    )?;
    println!("Resolved {count} report(s) as {}.", resolution.outcome);
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

pub fn list(as_json: bool, resolved: bool) -> Result<()> {
    let dir = reflector::reflector_dir();
    let content = if resolved {
        reflector::read_or_empty(&dir.join(RESOLVED_FILE))?
    } else {
        let mut open = reflector::read_or_empty(&dir.join(PROCESSING_FILE))?;
        open.push_str(&reflector::read_or_empty(
            &dir.join(reflector::FEEDBACK_FILE),
        )?);
        open
    };
    let lines: Vec<&str> = reflector::non_empty_lines(&content).collect();

    if as_json {
        for line in &lines {
            println!("{line}");
        }
        return Ok(());
    }

    if lines.is_empty() {
        println!(
            "{}",
            if resolved {
                "No resolved feedback."
            } else {
                "No open feedback."
            }
        );
        return Ok(());
    }

    println!(
        "{} {} feedback report(s)",
        lines.len(),
        if resolved { "resolved" } else { "open" }
    );
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
            "{} [{}] {} (c0 {})",
            field(&entry, "id").unwrap_or("--------"),
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
        if let Some(resolution) = entry.get("resolution") {
            println!(
                "  resolved: {} {}",
                field(resolution, "outcome").unwrap_or("?"),
                field(resolution, "ref").unwrap_or("")
            );
            if let Some(note) = field(resolution, "note") {
                println!("  note: {note}");
            }
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

    fn scratch(name: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("c0-feedback-{name}-{}-{nanos}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn ids_of(lines: &[String]) -> Vec<String> {
        lines
            .iter()
            .map(|l| {
                let v: Value = serde_json::from_str(l).unwrap();
                v["id"].as_str().unwrap().to_string()
            })
            .collect()
    }

    #[test]
    fn claim_assigns_stable_ids_and_keeps_earlier_claims() {
        let dir = scratch("claim");
        let inbox = dir.join(reflector::FEEDBACK_FILE);
        fs::write(&inbox, "{\"message\":\"a\"}\n{\"message\":\"a\"}\n").unwrap();

        let first = claim_in(&dir).unwrap();
        let first_ids = ids_of(&first);
        assert_eq!(first.len(), 2);
        assert_ne!(first_ids[0], first_ids[1]);
        assert_eq!(fs::read_to_string(&inbox).unwrap(), "");

        fs::write(&inbox, "{\"id\":\"keepme\",\"message\":\"b\"}\n").unwrap();
        let second = claim_in(&dir).unwrap();
        let second_ids = ids_of(&second);
        assert_eq!(second_ids[..2], first_ids[..]);
        assert_eq!(second_ids[2], "keepme");
    }

    #[test]
    fn resolve_moves_only_the_named_reports_and_records_the_outcome() {
        let dir = scratch("resolve");
        fs::write(
            dir.join(reflector::FEEDBACK_FILE),
            "{\"id\":\"one\",\"message\":\"a\"}\n{\"id\":\"two\",\"message\":\"b\"}\n",
        )
        .unwrap();
        claim_in(&dir).unwrap();

        let resolution = Resolution {
            outcome: "pr",
            reference: Some("https://example.test/pr/1"),
            note: None,
        };
        let n = resolve_in(
            &dir,
            &["one".to_string()],
            &resolution,
            "2026-11-23T00:00:00Z",
        )
        .unwrap();
        assert_eq!(n, 1);

        let left = fs::read_to_string(dir.join(PROCESSING_FILE)).unwrap();
        assert_eq!(left, "{\"id\":\"two\",\"message\":\"b\"}\n");

        let done: Value =
            serde_json::from_str(fs::read_to_string(dir.join(RESOLVED_FILE)).unwrap().trim())
                .unwrap();
        assert_eq!(done["id"], "one");
        assert_eq!(done["resolution"]["outcome"], "pr");
        assert_eq!(done["resolution"]["ref"], "https://example.test/pr/1");
    }

    #[test]
    fn resolve_rejects_unknown_ids_and_outcomes_without_changing_anything() {
        let dir = scratch("resolve-bad");
        fs::write(
            dir.join(reflector::FEEDBACK_FILE),
            "{\"id\":\"one\",\"message\":\"a\"}\n",
        )
        .unwrap();
        claim_in(&dir).unwrap();
        let before = fs::read_to_string(dir.join(PROCESSING_FILE)).unwrap();

        let ok = Resolution {
            outcome: "wontfix",
            reference: None,
            note: None,
        };
        let ids = ["one".to_string(), "ghost".to_string()];
        assert!(resolve_in(&dir, &ids, &ok, "t").is_err());

        let bad = Resolution {
            outcome: "merged-it-myself",
            reference: None,
            note: None,
        };
        assert!(resolve_in(&dir, &["one".to_string()], &bad, "t").is_err());

        assert_eq!(
            fs::read_to_string(dir.join(PROCESSING_FILE)).unwrap(),
            before
        );
        assert!(!dir.join(RESOLVED_FILE).exists());
    }

    #[test]
    fn session_keys_cannot_escape_the_sessions_dir() {
        assert!(is_safe_component("0873b5f3-b670"));
        assert!(!is_safe_component("../etc"));
        assert!(!is_safe_component(".."));
        assert!(!is_safe_component(""));
    }
}
