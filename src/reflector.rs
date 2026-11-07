use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use crate::claude::{LlmClient, log_usage};
use crate::config::SemanticConfig;

const DEFAULT_HTTP_TIMEOUT_SECS: u64 = 60;
const DEFAULT_CONNECT_TIMEOUT_SECS: u64 = 10;

const INBOX_FILE: &str = "inbox.jsonl";
const CLAIMED_FILE: &str = "inbox.processing.jsonl";
const REVIEW_FILE: &str = "review.jsonl";
const PENDING_FILE: &str = "pending-commits.jsonl";
const LOCK_FILE: &str = "reflector.lock";

fn reflector_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".c0/reflector")
}

fn lock_dir(dir: &Path) -> Result<fs::File> {
    fs::create_dir_all(dir)?;
    let lock = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(dir.join(LOCK_FILE))?;
    lock.lock()?;
    Ok(lock)
}

fn read_or_empty(path: &Path) -> Result<String> {
    match fs::read_to_string(path) {
        Ok(content) => Ok(content),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(String::new()),
        Err(e) => Err(e.into()),
    }
}

fn non_empty_lines(content: &str) -> impl Iterator<Item = &str> {
    content.lines().filter(|l| !l.trim().is_empty())
}

fn count_lines(path: &Path) -> usize {
    read_or_empty(path)
        .map(|c| non_empty_lines(&c).count())
        .unwrap_or(0)
}

fn join_lines<S: AsRef<str>>(lines: &[S]) -> String {
    let mut out = String::new();
    for line in lines {
        out.push_str(line.as_ref().trim_end_matches('\n'));
        out.push('\n');
    }
    out
}

fn append_unlocked<S: AsRef<str>>(path: &Path, lines: &[S]) -> Result<()> {
    if lines.is_empty() {
        return Ok(());
    }
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    file.write_all(join_lines(lines).as_bytes())?;
    Ok(())
}

fn write_atomic(path: &Path, content: &str) -> Result<()> {
    let tmp = path.with_extension("jsonl.tmp");
    fs::write(&tmp, content)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

fn append_lines(dir: &Path, file: &str, lines: &[String]) -> Result<()> {
    let _lock = lock_dir(dir)?;
    append_unlocked(&dir.join(file), lines)
}

fn remove_lines(dir: &Path, file: &str, handled: &[String]) -> Result<usize> {
    let _lock = lock_dir(dir)?;
    let path = dir.join(file);
    let current = read_or_empty(&path)?;

    let mut to_remove: HashMap<&str, usize> = HashMap::new();
    for line in handled {
        *to_remove.entry(line.as_str()).or_insert(0) += 1;
    }

    let kept: Vec<&str> = non_empty_lines(&current)
        .filter(|line| match to_remove.get_mut(line) {
            Some(n) if *n > 0 => {
                *n -= 1;
                false
            }
            _ => true,
        })
        .collect();

    write_atomic(&path, &join_lines(&kept))?;
    Ok(kept.len())
}

fn with_inbox_in<T>(dir: &Path, f: impl FnOnce(&str, &mut fs::File) -> T) -> Result<T> {
    let _lock = lock_dir(dir)?;
    let mut existing = read_or_empty(&dir.join(INBOX_FILE))?;
    existing.push_str(&read_or_empty(&dir.join(CLAIMED_FILE))?);
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join(INBOX_FILE))?;
    Ok(f(&existing, &mut file))
}

pub fn with_inbox<T>(f: impl FnOnce(&str, &mut fs::File) -> T) -> Result<T> {
    with_inbox_in(&reflector_dir(), f)
}

fn claim_inbox_in(dir: &Path) -> Result<Vec<String>> {
    let _lock = lock_dir(dir)?;
    let inbox = dir.join(INBOX_FILE);
    let claimed = dir.join(CLAIMED_FILE);

    let incoming = read_or_empty(&inbox)?;
    let incoming: Vec<&str> = non_empty_lines(&incoming).collect();
    if !incoming.is_empty() {
        append_unlocked(&claimed, &incoming)?;
        fs::write(&inbox, "")?;
    }

    Ok(non_empty_lines(&read_or_empty(&claimed)?)
        .map(String::from)
        .collect())
}

fn release_claim_in(dir: &Path) -> Result<()> {
    let _lock = lock_dir(dir)?;
    match fs::remove_file(dir.join(CLAIMED_FILE)) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.into()),
    }
}

pub fn claim() -> Result<()> {
    let lines = claim_inbox_in(&ensure_dir()?)?;
    if lines.is_empty() {
        eprintln!("Inbox is empty.");
    }
    for line in &lines {
        println!("{line}");
    }
    Ok(())
}

pub fn release() -> Result<()> {
    release_claim_in(&ensure_dir()?)?;
    eprintln!("Released claimed batch.");
    Ok(())
}

pub fn enqueue_review() -> Result<()> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;

    let mut lines: Vec<String> = Vec::new();
    for (i, line) in non_empty_lines(&input).enumerate() {
        let value: serde_json::Value = serde_json::from_str(line).map_err(|e| {
            anyhow::anyhow!("line {} is not valid JSON ({e}); nothing queued", i + 1)
        })?;
        lines.push(serde_json::to_string(&value)?);
    }

    append_lines(&ensure_dir()?, REVIEW_FILE, &lines)?;
    println!("Queued {} item(s) for review.", lines.len());
    Ok(())
}

fn ensure_dir() -> Result<PathBuf> {
    let dir = reflector_dir();
    fs::create_dir_all(&dir)?;

    for file in &[INBOX_FILE, "proposed.jsonl", REVIEW_FILE, PENDING_FILE] {
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

    let inbox_count = count_lines(&dir.join(INBOX_FILE));
    let claimed_count = count_lines(&dir.join(CLAIMED_FILE));
    let proposed_count = count_lines(&dir.join("proposed.jsonl"));
    let review_count = count_lines(&dir.join(REVIEW_FILE));
    let pending_count = count_lines(&dir.join(PENDING_FILE));

    println!("Inbox: {inbox_count} dead end(s)");
    if claimed_count > 0 {
        println!("Claimed (being processed or interrupted): {claimed_count} dead end(s)");
    }
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
    let _lock = lock_dir(&dir)?;

    let inbox_path = dir.join(INBOX_FILE);
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
    let content = read_or_empty(&dir.join(REVIEW_FILE))?;
    let lines: Vec<&str> = non_empty_lines(&content).collect();

    if lines.is_empty() {
        println!("Review queue is empty. No uncertain items from ghost.");
        return Ok(());
    }

    println!("Items queued for human review ({}):", lines.len());
    println!("═══════════════════════════════════════");
    println!();

    let mut handled: Vec<String> = Vec::new();

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
                    handled.push(line.to_string());
                }
                "s" | "skip" => {
                    println!("   → Skipped (keeping in queue)");
                }
                "q" | "quit" => {
                    break;
                }
                _ => {
                    println!("   → Unknown choice, keeping in queue");
                }
            }
            println!();
        }
    }

    let remaining = remove_lines(&dir, REVIEW_FILE, &handled)?;

    println!("═══════════════════════════════════════");
    println!(
        "Processed {} item(s). {} remaining in queue.",
        handled.len(),
        remaining
    );

    Ok(())
}

pub fn apply() -> Result<()> {
    let dir = ensure_dir()?;
    let content = read_or_empty(&dir.join(PENDING_FILE))?;
    let lines: Vec<&str> = non_empty_lines(&content).collect();

    if lines.is_empty() {
        println!("No pending commits from ghost.");
        return Ok(());
    }

    println!("Applying {} pending commit(s) from ghost:", lines.len());
    println!("═══════════════════════════════════════");

    let mut applied: Vec<String> = Vec::new();

    for line in &lines {
        let Ok(entry) = serde_json::from_str::<serde_json::Value>(line) else {
            println!("  Skipping malformed line (kept in queue): {line}");
            println!();
            continue;
        };
        let concept = entry.get("concept").and_then(|v| v.as_str()).unwrap_or("");
        let reason = entry.get("reason").and_then(|v| v.as_str()).unwrap_or("");

        if concept.is_empty() {
            println!("  Skipping entry with no concept (kept in queue): {line}");
            println!();
            continue;
        }

        println!("  Adding: {concept}");
        if !reason.is_empty() {
            println!("  Reason: {reason}");
        }

        // Pass --force so the fuzzy-duplicate guard in `add concept` can't
        // silently skip the concept the classifier already decided to COMMIT.
        // --quiet suppresses suggestion noise. --source reflector stamps
        // provenance so the reflector's output can be audited (otherwise it
        // is indistinguishable from manual adds). Pass the reason as the
        // description so the new node gets an embedding and is retrievable.
        let mut args: Vec<&str> = vec![
            "add",
            "concept",
            concept,
            "--force",
            "--quiet",
            "--source",
            "reflector",
        ];
        if !reason.is_empty() {
            args.push("-d");
            args.push(reason);
        }
        // Invoke this same binary by absolute path rather than relying on `c0`
        // being on $PATH — under systemd the daemon's $PATH often lacks the
        // user's cargo bin dir, which silently breaks every commit.
        let exe = std::env::current_exe().unwrap_or_else(|_| "c0".into());
        let result = Command::new(exe).args(&args).output();

        match result {
            Ok(output) if output.status.success() => {
                println!("  → Created");
                applied.push(line.to_string());
            }
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                let stdout = String::from_utf8_lossy(&output.stdout);
                if stdout.contains("already exists") || stderr.contains("already exists") {
                    println!("  → Already exists");
                    applied.push(line.to_string());
                } else {
                    println!("  → Failed: {}", stderr.trim());
                }
            }
            Err(e) => {
                println!("  → Error: {e}");
            }
        }
        println!();
    }

    let applied_count = applied.len();
    let remaining = remove_lines(&dir, PENDING_FILE, &applied)?;

    println!("═══════════════════════════════════════");
    println!("Applied {applied_count} concept(s). {remaining} kept in queue.");

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

    let system_context = "You are the c0 knowledge curator. You help classify dead-end queries from a knowledge graph system, deciding which should become new concepts (COMMIT), which are noise (DISCARD), and which need human review (QUEUE).";
    let full_prompt = format!("{system_context}\n\n{prompt}");

    // Classification is stateless: each dead end is independent, so we do NOT
    // resume a persisted Claude session. The old `--resume` path replayed an
    // ever-growing conversation on every call, so per-classification latency
    // climbed with each processed dead end until it crossed the 120s CLI
    // timeout — silently dropping real concepts into the review queue as false
    // QUEUEs. A fresh `generate` keeps every call O(1) and removes the
    // stale-session failure mode entirely.
    //
    // Any single failure (timeout, dropped connection, a transient CLI error)
    // gets exactly one fresh retry before we give up, so one hiccup no longer
    // demotes a classifiable concept to the review queue.
    let response = match llm.generate(&full_prompt, Some(CLASSIFY_SCHEMA)).await {
        Ok(resp) => resp,
        Err(first_err) => {
            eprintln!("reflector: classification attempt failed ({first_err}); retrying once");
            llm.generate(&full_prompt, Some(CLASSIFY_SCHEMA))
                .await
                .map_err(|retry_err| {
                    anyhow::anyhow!(
                        "classification failed after retry: {retry_err} \
                         (first attempt: {first_err})"
                    )
                })?
        }
    };

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
    let log_path = dir.join("process.log");

    let lines = claim_inbox_in(&dir)?;

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

    let commit_lines = commits
        .iter()
        .map(serde_json::to_string)
        .collect::<Result<Vec<_>, _>>()?;
    append_lines(&dir, PENDING_FILE, &commit_lines)?;

    let queued_lines = queued
        .iter()
        .map(serde_json::to_string)
        .collect::<Result<Vec<_>, _>>()?;
    append_lines(&dir, REVIEW_FILE, &queued_lines)?;

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

    release_claim_in(&dir)?;

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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    fn scratch(name: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "c0-reflector-{name}-{}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn claim_moves_inbox_into_batch_and_keeps_interrupted_leftovers() {
        let dir = scratch("claim");
        fs::write(dir.join(CLAIMED_FILE), "{\"query\":\"left\"}\n").unwrap();
        fs::write(dir.join(INBOX_FILE), "{\"query\":\"new\"}\n\n").unwrap();

        let batch = claim_inbox_in(&dir).unwrap();

        assert_eq!(batch, vec!["{\"query\":\"left\"}", "{\"query\":\"new\"}"]);
        assert_eq!(fs::read_to_string(dir.join(INBOX_FILE)).unwrap(), "");

        release_claim_in(&dir).unwrap();
        assert!(!dir.join(CLAIMED_FILE).exists());
        release_claim_in(&dir).unwrap();

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn remove_lines_preserves_entries_appended_after_the_snapshot() {
        let dir = scratch("remove");
        fs::write(dir.join(REVIEW_FILE), "a\nb\n").unwrap();

        let snapshot = read_or_empty(&dir.join(REVIEW_FILE)).unwrap();
        let handled: Vec<String> = non_empty_lines(&snapshot)
            .filter(|l| *l == "a")
            .map(String::from)
            .collect();
        append_lines(&dir, REVIEW_FILE, &["c".to_string()]).unwrap();

        let remaining = remove_lines(&dir, REVIEW_FILE, &handled).unwrap();

        assert_eq!(remaining, 2);
        assert_eq!(fs::read_to_string(dir.join(REVIEW_FILE)).unwrap(), "b\nc\n");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn remove_lines_drops_one_occurrence_per_handled_entry() {
        let dir = scratch("dupes");
        fs::write(dir.join(PENDING_FILE), "x\nx\ny\n").unwrap();

        let remaining = remove_lines(&dir, PENDING_FILE, &["x".to_string()]).unwrap();

        assert_eq!(remaining, 2);
        assert_eq!(
            fs::read_to_string(dir.join(PENDING_FILE)).unwrap(),
            "x\ny\n"
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn with_inbox_sees_claimed_entries_for_dedup() {
        let dir = scratch("dedup");
        fs::write(dir.join(CLAIMED_FILE), "{\"query\":\"busy\"}\n").unwrap();

        let seen = with_inbox_in(&dir, |existing, _| existing.contains("busy")).unwrap();

        assert!(seen);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn concurrent_appends_are_never_lost_across_claims() {
        let dir = scratch("race");
        let writers = 4;
        let per_writer = 250;
        let done = Arc::new(AtomicBool::new(false));

        let handles: Vec<_> = (0..writers)
            .map(|w| {
                let dir = dir.clone();
                std::thread::spawn(move || {
                    for i in 0..per_writer {
                        with_inbox_in(&dir, |_, f| writeln!(f, "{{\"w\":{w},\"i\":{i}}}").unwrap())
                            .unwrap();
                    }
                })
            })
            .collect();

        let claimer = {
            let dir = dir.clone();
            let done = Arc::clone(&done);
            std::thread::spawn(move || {
                let mut seen: Vec<String> = Vec::new();
                loop {
                    let finished = done.load(Ordering::SeqCst);
                    seen.extend(claim_inbox_in(&dir).unwrap());
                    release_claim_in(&dir).unwrap();
                    if finished {
                        return seen;
                    }
                }
            })
        };

        for h in handles {
            h.join().unwrap();
        }
        done.store(true, Ordering::SeqCst);
        let seen = claimer.join().unwrap();

        let unique: HashSet<&String> = seen.iter().collect();
        assert_eq!(seen.len(), writers * per_writer);
        assert_eq!(unique.len(), writers * per_writer);
        fs::remove_dir_all(&dir).unwrap();
    }
}
