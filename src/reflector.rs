use anyhow::Result;
use std::collections::HashSet;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use crate::claude::{LlmClient, log_usage};
use crate::config::SemanticConfig;

const DEFAULT_HTTP_TIMEOUT_SECS: u64 = 60;
const DEFAULT_CONNECT_TIMEOUT_SECS: u64 = 10;

fn reflector_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".c0/reflector")
}

fn ensure_dir() -> Result<PathBuf> {
    let dir = reflector_dir();
    fs::create_dir_all(&dir)?;

    for file in &[
        "inbox.jsonl",
        "proposed.jsonl",
        "review.jsonl",
        "pending-commits.jsonl",
    ] {
        let path = dir.join(file);
        if !path.exists() {
            fs::write(&path, "")?;
        }
    }

    Ok(dir)
}

pub fn status() -> Result<()> {
    let dir = reflector_dir();

    println!("C0 Reflector Status");
    println!("═══════════════════════════════════════");
    println!("Directory: {}", dir.display());

    let inbox_path = dir.join("inbox.jsonl");
    let inbox_count = if inbox_path.exists() {
        fs::read_to_string(&inbox_path)
            .map(|c| c.lines().filter(|l| !l.is_empty()).count())
            .unwrap_or(0)
    } else {
        0
    };

    let proposed_path = dir.join("proposed.jsonl");
    let proposed_count = if proposed_path.exists() {
        fs::read_to_string(&proposed_path)
            .map(|c| c.lines().filter(|l| !l.is_empty()).count())
            .unwrap_or(0)
    } else {
        0
    };

    let review_path = dir.join("review.jsonl");
    let review_count = if review_path.exists() {
        fs::read_to_string(&review_path)
            .map(|c| c.lines().filter(|l| !l.is_empty()).count())
            .unwrap_or(0)
    } else {
        0
    };

    let pending_path = dir.join("pending-commits.jsonl");
    let pending_count = if pending_path.exists() {
        fs::read_to_string(&pending_path)
            .map(|c| c.lines().filter(|l| !l.is_empty()).count())
            .unwrap_or(0)
    } else {
        0
    };

    println!("Inbox: {inbox_count} dead end(s)");
    println!("Proposed: {proposed_count} learning(s)");
    println!("Review queue: {review_count} item(s)");
    println!("Pending commits: {pending_count} concept(s)");
    println!();
    if pending_count > 0 {
        println!("Run 'c0 reflector apply' to commit pending concepts.");
    } else {
        println!("Ghost auto-commits high-confidence learnings.");
    }

    Ok(())
}

pub fn inbox() -> Result<()> {
    let dir = ensure_dir()?;
    let inbox_path = dir.join("inbox.jsonl");

    let content = fs::read_to_string(&inbox_path)?;
    let lines: Vec<&str> = content.lines().filter(|l| !l.is_empty()).collect();

    if lines.is_empty() {
        println!("Inbox is empty. No dead ends queued for reflection.");
    } else {
        println!("Dead ends queued for reflection ({}):", lines.len());
        println!("═══════════════════════════════════════");
        for line in lines.iter().rev().take(20) {
            if let Ok(entry) = serde_json::from_str::<serde_json::Value>(line) {
                let timestamp = entry
                    .get("timestamp")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?");
                let command = entry.get("command").and_then(|v| v.as_str()).unwrap_or("?");
                let query = entry.get("query").and_then(|v| v.as_str()).unwrap_or("?");
                let session = entry.get("session").and_then(|v| v.as_str()).unwrap_or("?");

                let ts_short = timestamp.split('T').next().unwrap_or(timestamp);
                println!("  [{ts_short}] {command} {query} ({session})");
            } else {
                println!("  {line}");
            }
        }
        if lines.len() > 20 {
            println!("  ... and {} more", lines.len() - 20);
        }
    }

    Ok(())
}

pub fn proposed() -> Result<()> {
    let dir = reflector_dir();
    let proposed_path = dir.join("proposed.jsonl");

    if !proposed_path.exists() {
        println!("No proposed learnings found.");
        return Ok(());
    }

    let content = fs::read_to_string(&proposed_path)?;
    let lines: Vec<&str> = content.lines().filter(|l| !l.is_empty()).collect();

    if lines.is_empty() {
        println!("No proposed learnings. Reflector hasn't suggested anything yet.");
    } else {
        println!("Proposed learnings ({}):", lines.len());
        println!("═══════════════════════════════════════");
        for (i, line) in lines.iter().enumerate() {
            if let Ok(entry) = serde_json::from_str::<serde_json::Value>(line) {
                let concept = entry.get("concept").and_then(|v| v.as_str()).unwrap_or("?");
                let reason = entry.get("reason").and_then(|v| v.as_str()).unwrap_or("?");

                println!("{}. {}", i + 1, concept);
                println!("   Reason: {reason}");

                if let Some(relations) = entry.get("relations").and_then(|v| v.as_array()) {
                    let rels: Vec<&str> = relations.iter().filter_map(|v| v.as_str()).collect();
                    if !rels.is_empty() {
                        println!("   Relations: {}", rels.join(", "));
                    }
                }
                println!();
            } else {
                println!("{}. {}", i + 1, line);
            }
        }
        println!("To commit: c0 add concept <name>");
    }

    Ok(())
}

pub fn clear() -> Result<()> {
    let dir = reflector_dir();

    let inbox_path = dir.join("inbox.jsonl");
    let proposed_path = dir.join("proposed.jsonl");

    let mut cleared = 0;

    if inbox_path.exists() {
        let count = fs::read_to_string(&inbox_path)
            .map(|c| c.lines().filter(|l| !l.is_empty()).count())
            .unwrap_or(0);
        if count > 0 {
            fs::write(&inbox_path, "")?;
            println!("Cleared {count} dead end(s) from inbox");
            cleared += count;
        }
    }

    if proposed_path.exists() {
        let count = fs::read_to_string(&proposed_path)
            .map(|c| c.lines().filter(|l| !l.is_empty()).count())
            .unwrap_or(0);
        if count > 0 {
            fs::write(&proposed_path, "")?;
            println!("Cleared {count} proposed learning(s)");
            cleared += count;
        }
    }

    if cleared == 0 {
        println!("Nothing to clear.");
    }

    Ok(())
}

pub fn review() -> Result<()> {
    let dir = ensure_dir()?;
    let review_path = dir.join("review.jsonl");

    let content = fs::read_to_string(&review_path)?;
    let lines: Vec<&str> = content.lines().filter(|l| !l.is_empty()).collect();

    if lines.is_empty() {
        println!("Review queue is empty. No uncertain items from ghost.");
        return Ok(());
    }

    println!("Items queued for human review ({}):", lines.len());
    println!("═══════════════════════════════════════");
    println!();

    let mut remaining_lines: Vec<String> = Vec::new();
    let mut processed = 0;

    for (i, line) in lines.iter().enumerate() {
        if let Ok(entry) = serde_json::from_str::<serde_json::Value>(line) {
            let query = entry.get("query").and_then(|v| v.as_str()).unwrap_or("?");
            let reason = entry
                .get("reason")
                .and_then(|v| v.as_str())
                .unwrap_or("Ghost was uncertain");
            let timestamp = entry
                .get("timestamp")
                .and_then(|v| v.as_str())
                .unwrap_or("?");

            println!("{}. \"{}\"", i + 1, query);
            println!("   Reason: {reason}");
            println!("   From: {timestamp}");
            println!();
            print!("   Action? [c]ommit / [s]kip / [q]uit: ");
            io::stdout().flush()?;

            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            let choice = input.trim().to_lowercase();

            match choice.as_str() {
                "c" | "commit" => {
                    println!("   → Run: c0 add concept {query}");
                    processed += 1;
                }
                "s" | "skip" => {
                    println!("   → Skipped (keeping in queue)");
                    remaining_lines.push(line.to_string());
                }
                "q" | "quit" => {
                    for remaining in &lines[i..] {
                        remaining_lines.push(remaining.to_string());
                    }
                    break;
                }
                _ => {
                    println!("   → Unknown choice, keeping in queue");
                    remaining_lines.push(line.to_string());
                }
            }
            println!();
        } else {
            remaining_lines.push(line.to_string());
        }
    }

    let new_content = remaining_lines.join("\n");
    let new_content = if new_content.is_empty() {
        String::new()
    } else {
        format!("{new_content}\n")
    };
    fs::write(&review_path, new_content)?;

    println!("═══════════════════════════════════════");
    println!(
        "Processed {} item(s). {} remaining in queue.",
        processed,
        remaining_lines.len()
    );

    Ok(())
}

pub fn apply() -> Result<()> {
    let dir = ensure_dir()?;
    let pending_path = dir.join("pending-commits.jsonl");

    let content = fs::read_to_string(&pending_path)?;
    let lines: Vec<&str> = content.lines().filter(|l| !l.is_empty()).collect();

    if lines.is_empty() {
        println!("No pending commits from ghost.");
        return Ok(());
    }

    println!("Applying {} pending commit(s) from ghost:", lines.len());
    println!("═══════════════════════════════════════");

    let mut applied = 0;
    let mut failed: Vec<String> = Vec::new();

    for line in &lines {
        if let Ok(entry) = serde_json::from_str::<serde_json::Value>(line) {
            let concept = entry.get("concept").and_then(|v| v.as_str()).unwrap_or("");
            let reason = entry.get("reason").and_then(|v| v.as_str()).unwrap_or("");

            if concept.is_empty() {
                continue;
            }

            println!("  Adding: {concept}");
            if !reason.is_empty() {
                println!("  Reason: {reason}");
            }

            // Pass --force so the fuzzy-duplicate guard in `add concept` can't
            // silently skip (exit 0 without creating) the concept the classifier
            // already decided to COMMIT. Pass the reason as the description so the
            // new node gets an embedding and is actually retrievable.
            let mut args: Vec<&str> = vec!["add", "concept", concept, "--force"];
            if !reason.is_empty() {
                args.push("-d");
                args.push(reason);
            }
            let result = Command::new("c0").args(&args).output();

            match result {
                Ok(output) if output.status.success() => {
                    println!("  → Created");
                    applied += 1;
                }
                Ok(output) => {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    if stdout.contains("already exists") || stderr.contains("already exists") {
                        println!("  → Already exists");
                        applied += 1;
                    } else {
                        println!("  → Failed: {}", stderr.trim());
                        failed.push(line.to_string());
                    }
                }
                Err(e) => {
                    println!("  → Error: {e}");
                    failed.push(line.to_string());
                }
            }
            println!();
        }
    }

    let new_content = if failed.is_empty() {
        String::new()
    } else {
        format!("{}\n", failed.join("\n"))
    };
    fs::write(&pending_path, new_content)?;

    println!("═══════════════════════════════════════");
    println!(
        "Applied {} concept(s). {} failed (kept in queue).",
        applied,
        failed.len()
    );

    Ok(())
}

/// Parse a watch interval like `30s`, `5m`, `1h`. A bare number means seconds.
fn parse_interval(spec: &str) -> Result<Duration> {
    let spec = spec.trim();
    let (digits, mult) = match spec.chars().last() {
        Some('s') => (&spec[..spec.len() - 1], 1),
        Some('m') => (&spec[..spec.len() - 1], 60),
        Some('h') => (&spec[..spec.len() - 1], 3600),
        _ => (spec, 1),
    };

    let n: u64 = digits
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid interval '{spec}' (use e.g. 30s, 5m, 1h)"))?;
    if n == 0 {
        anyhow::bail!("interval must be greater than zero");
    }

    Ok(Duration::from_secs(n.saturating_mul(mult)))
}

/// Watch mode: classify the inbox, optionally apply commits, sleep, repeat.
///
/// Runs until interrupted (Ctrl-C). For an unattended setup, prefer a cron
/// entry or a systemd unit — the OS handles restarts and logging.
pub async fn run(interval: &str, apply_commits: bool) -> Result<()> {
    let period = parse_interval(interval)?;

    println!("c0 reflector watch — ticking every {interval} (Ctrl-C to stop)");
    println!(
        "Auto-apply: {}",
        if apply_commits {
            "on"
        } else {
            "off (COMMIT decisions stay in the review queue)"
        }
    );
    println!("═══════════════════════════════════════");

    loop {
        if let Err(e) = process().await {
            eprintln!("process failed: {e}");
        }
        if apply_commits && let Err(e) = apply() {
            eprintln!("apply failed: {e}");
        }
        tokio::time::sleep(period).await;
    }
}

const CLASSIFY_PROMPT: &str = r#"You are a knowledge system curator. Classify this dead-end query (a search that found nothing in the knowledge graph).

Query: {query}
Namespace: {namespace}
Context: {context}

Decide:
- COMMIT: General, reusable concept worth adding (technologies, patterns, APIs, methodologies)
- DISCARD: Noise, typo, test query, session-specific path, or too vague
- QUEUE: Uncertain, might need human context

DISCARD if: contains "test", "example", "foo", "bar", "tmp", local paths, or looks like a typo.
COMMIT if: technology, library, pattern, or concept useful across multiple projects.
Use the context to understand WHY the assistant was searching for this - it helps distinguish real concepts from noise.

Reply with ONLY valid JSON:
{"decision": "COMMIT", "reason": "brief reason"}
or
{"decision": "DISCARD", "reason": "brief reason"}
or
{"decision": "QUEUE", "reason": "brief reason"}"#;

#[derive(serde::Deserialize)]
struct ClassifyResponse {
    decision: String,
    reason: String,
}

#[derive(serde::Deserialize)]
struct OllamaResponse {
    response: String,
}

async fn classify_query_ollama(
    client: &reqwest::Client,
    host: &str,
    model: &str,
    query: &str,
    namespace: &str,
    context: Option<&str>,
) -> Result<ClassifyResponse> {
    let prompt = CLASSIFY_PROMPT
        .replace("{query}", query)
        .replace("{namespace}", namespace)
        .replace("{context}", context.unwrap_or("(none provided)"));

    let resp = client
        .post(format!("{host}/api/generate"))
        .json(&serde_json::json!({
            "model": model,
            "prompt": prompt,
            "stream": false,
            "format": "json"
        }))
        .timeout(std::time::Duration::from_secs(60))
        .send()
        .await?;

    let ollama_resp: OllamaResponse = resp.json().await?;
    let classification: ClassifyResponse = serde_json::from_str(&ollama_resp.response)?;

    Ok(classification)
}

const REFLECTOR_SESSION_FILE: &str = ".c0/reflector-session.txt";

fn get_or_create_session_file() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(REFLECTOR_SESSION_FILE)
}

fn get_reflector_session_id() -> Option<String> {
    let session_file = get_or_create_session_file();
    if session_file.exists() {
        fs::read_to_string(&session_file)
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    } else {
        None
    }
}

fn save_reflector_session_id(session_id: &str) {
    let session_file = get_or_create_session_file();
    let _ = fs::write(&session_file, session_id);
}

const CLASSIFY_SCHEMA: &str = r#"{"type":"object","properties":{"decision":{"type":"string","enum":["COMMIT","DISCARD","QUEUE"]},"reason":{"type":"string"}},"required":["decision","reason"],"additionalProperties":false}"#;

async fn classify_query_llm(
    llm: &LlmClient,
    query: &str,
    namespace: &str,
    context: Option<&str>,
) -> Result<(ClassifyResponse, Option<f64>)> {
    let prompt = CLASSIFY_PROMPT
        .replace("{query}", query)
        .replace("{namespace}", namespace)
        .replace("{context}", context.unwrap_or("(none provided)"));

    let session_id = get_reflector_session_id();

    let system_context = "You are the c0 knowledge curator. You help classify dead-end queries from a knowledge graph system, deciding which should become new concepts (COMMIT), which are noise (DISCARD), and which need human review (QUEUE). Build context over time about the domain and project patterns.";
    let full_prompt = format!("{system_context}\n\n{prompt}");

    let response = if let Some(ref sid) = session_id {
        match llm
            .generate_resume(&prompt, sid, Some(CLASSIFY_SCHEMA))
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("No conversation found") || msg.contains("session ID") {
                    // The persisted Claude conversation is gone (e.g. it aged out
                    // of claude's history). Drop the stale id and retry once from a
                    // fresh session rather than hard-failing every classification.
                    let _ = fs::remove_file(get_or_create_session_file());
                    llm.generate(&full_prompt, Some(CLASSIFY_SCHEMA)).await?
                } else {
                    return Err(e);
                }
            }
        }
    } else {
        llm.generate(&full_prompt, Some(CLASSIFY_SCHEMA)).await?
    };

    if let Some(ref new_sid) = response.session_id {
        save_reflector_session_id(new_sid);
    }

    let result_str = response.result.trim();
    let result_str = if result_str.starts_with("```") {
        result_str
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim()
    } else {
        result_str
    };

    let classification: ClassifyResponse = serde_json::from_str(result_str).map_err(|e| {
        anyhow::anyhow!("Failed to parse LLM response: {e}\nResponse: {result_str}")
    })?;

    Ok((classification, response.total_cost_usd))
}

async fn classify_query(
    config: &SemanticConfig,
    client: &reqwest::Client,
    query: &str,
    namespace: &str,
    context: Option<&str>,
) -> Result<(ClassifyResponse, Option<f64>)> {
    let provider = config.claude.provider_for("classification");
    if provider == "ollama" {
        let model = config.model_for("classification", "ollama");
        let result = classify_query_ollama(
            client,
            &config.ollama_host,
            &model,
            query,
            namespace,
            context,
        )
        .await?;
        Ok((result, None))
    } else {
        let llm = LlmClient::for_task(config, "classification", config.claude.timeout_secs);
        classify_query_llm(&llm, query, namespace, context).await
    }
}

pub async fn process() -> Result<()> {
    let dir = ensure_dir()?;
    let inbox_path = dir.join("inbox.jsonl");
    let pending_path = dir.join("pending-commits.jsonl");
    let review_path = dir.join("review.jsonl");
    let log_path = dir.join("process.log");

    if !inbox_path.exists() {
        println!("No inbox.jsonl found.");
        return Ok(());
    }

    let content = fs::read_to_string(&inbox_path)?;
    let lines: Vec<&str> = content.lines().filter(|l| !l.is_empty()).collect();

    if lines.is_empty() {
        println!("Inbox is empty.");
        return Ok(());
    }

    let mut seen_queries: HashSet<String> = HashSet::new();
    let mut unique_entries: Vec<serde_json::Value> = Vec::new();

    for line in &lines {
        if let Ok(entry) = serde_json::from_str::<serde_json::Value>(line)
            && let Some(query) = entry.get("query").and_then(|v| v.as_str())
            && !seen_queries.contains(query)
        {
            seen_queries.insert(query.to_string());
            unique_entries.push(entry);
        }
    }

    println!(
        "Processing {} unique entries (from {} total)",
        unique_entries.len(),
        lines.len()
    );
    println!("═══════════════════════════════════════");

    let config = SemanticConfig::load();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(DEFAULT_HTTP_TIMEOUT_SECS))
        .connect_timeout(Duration::from_secs(DEFAULT_CONNECT_TIMEOUT_SECS))
        .build()?;

    let mut commits: Vec<serde_json::Value> = Vec::new();
    let mut queued: Vec<serde_json::Value> = Vec::new();
    let mut discarded = 0;

    for entry in &unique_entries {
        let query = entry.get("query").and_then(|v| v.as_str()).unwrap_or("");
        let session = entry
            .get("session")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let namespace = entry
            .get("namespace")
            .and_then(|v| v.as_str())
            .unwrap_or("global");
        let context = entry.get("context").and_then(|v| v.as_str());

        print!("  {query} ... ");
        io::stdout().flush()?;

        match classify_query(&config, &client, query, namespace, context).await {
            Ok((result, cost)) => {
                if let Some(cost_usd) = cost {
                    let provider = config.claude.provider_for("classification");
                    let model = config.model_for("classification", &provider);
                    log_usage("classification", &model, cost_usd);
                }
                let decision = result.decision.to_uppercase();
                let reason = result.reason;

                let now = chrono::Utc::now().to_rfc3339();
                let record = serde_json::json!({
                    "query": query,
                    "concept": query.replace(' ', "-").to_lowercase(),
                    "namespace": namespace,
                    "context": context,
                    "reason": reason,
                    "original_session": session,
                    "processed_at": now,
                });

                match decision.as_str() {
                    "COMMIT" => {
                        println!("COMMIT - {reason}");
                        commits.push(record);
                    }
                    "DISCARD" => {
                        println!("DISCARD - {reason}");
                        discarded += 1;
                    }
                    _ => {
                        println!("QUEUE - {reason}");
                        queued.push(record);
                    }
                }
            }
            Err(e) => {
                println!("ERROR - {e}");
                let now = chrono::Utc::now().to_rfc3339();
                queued.push(serde_json::json!({
                    "query": query,
                    "reason": format!("Classification error: {}", e),
                    "original_session": session,
                    "processed_at": now,
                }));
            }
        }
    }

    if !commits.is_empty() {
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&pending_path)?;
        for record in &commits {
            writeln!(file, "{}", serde_json::to_string(record)?)?;
        }
    }

    if !queued.is_empty() {
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&review_path)?;
        for record in &queued {
            writeln!(file, "{}", serde_json::to_string(record)?)?;
        }
    }

    let now = chrono::Utc::now().to_rfc3339();
    let log_entry = serde_json::json!({
        "timestamp": now,
        "processed": unique_entries.len(),
        "commits": commits.len(),
        "queued": queued.len(),
        "discarded": discarded,
    });
    let mut log_file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)?;
    writeln!(log_file, "{}", serde_json::to_string(&log_entry)?)?;

    fs::write(&inbox_path, "")?;

    println!("═══════════════════════════════════════");
    println!(
        "Done: {} commits, {} queued, {} discarded",
        commits.len(),
        queued.len(),
        discarded
    );
    println!("Inbox cleared.");
    println!();
    if !commits.is_empty() {
        println!("Run 'c0 reflector apply' to commit pending concepts.");
    }

    Ok(())
}

pub async fn notify() -> Result<()> {
    let dir = reflector_dir();

    let pending_path = dir.join("pending-commits.jsonl");
    let pending_count = if pending_path.exists() {
        fs::read_to_string(&pending_path)
            .map(|c| c.lines().filter(|l| !l.is_empty()).count())
            .unwrap_or(0)
    } else {
        0
    };

    let review_path = dir.join("review.jsonl");
    let review_count = if review_path.exists() {
        fs::read_to_string(&review_path)
            .map(|c| c.lines().filter(|l| !l.is_empty()).count())
            .unwrap_or(0)
    } else {
        0
    };

    let state_path = dir.join("notify-state.json");

    if pending_count == 0 && review_count == 0 {
        println!("Nothing to review.");
        let _ = fs::remove_file(&state_path);
        return Ok(());
    }

    let (last_pending, last_review) = if state_path.exists() {
        fs::read_to_string(&state_path)
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .map(|v| {
                (
                    v.get("pending").and_then(|x| x.as_u64()).unwrap_or(0) as usize,
                    v.get("review").and_then(|x| x.as_u64()).unwrap_or(0) as usize,
                )
            })
            .unwrap_or((0, 0))
    } else {
        (0, 0)
    };

    if pending_count <= last_pending && review_count <= last_review {
        println!(
            "Queue unchanged or shrunk since last notify ({pending_count} pending, {review_count} review) - skipping."
        );
        let _ = fs::write(
            &state_path,
            serde_json::to_string(&serde_json::json!({
                "pending": pending_count,
                "review": review_count,
                "notified_at": chrono::Utc::now().to_rfc3339(),
            }))?,
        );
        return Ok(());
    }

    let mut parts = Vec::new();
    if pending_count > 0 {
        parts.push(format!("{pending_count} pending commit(s)"));
    }
    if review_count > 0 {
        parts.push(format!("{review_count} queued for review"));
    }
    let body = parts.join(", ");

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .connect_timeout(Duration::from_secs(5))
        .build()?;
    let result = client
        .post("http://127.0.0.1:3002/api/notify")
        .json(&serde_json::json!({
            "title": "Run `c0reflect`",
            "body": body,
            "url": "/",
            "tag": "c0-reflector",
            "source": "c0",
            "requireInteraction": true,
            "actions": []
        }))
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await;

    match result {
        Ok(resp) if resp.status().is_success() => {
            println!("Push notification sent: {body}");
            let _ = fs::write(
                &state_path,
                serde_json::to_string(&serde_json::json!({
                    "pending": pending_count,
                    "review": review_count,
                    "notified_at": chrono::Utc::now().to_rfc3339(),
                }))?,
            );
        }
        Ok(resp) => {
            println!("Push notification failed ({}): {}", resp.status(), body);
        }
        Err(e) => {
            println!("Push notification error: {e} - {body}");
        }
    }

    Ok(())
}
