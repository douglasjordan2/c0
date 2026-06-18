use anyhow::Result;
use std::collections::HashMap;
use std::path::PathBuf;

use crate::claude::{ExtractedConcept, LlmClient};
use crate::config;
use crate::embeddings;
use crate::graph::{
    self, BashCall, EnrichmentInputs, EnrichmentTurn, FileTouch, Reflection, Session,
    SessionAggregates, ToolCallRecord, ToolResultBackfill, Turn,
};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct SessionsState {
    dirs: HashMap<String, DirState>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct DirState {
    index_mtime: Option<u64>,
    sessions: HashMap<String, u64>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct SessionsIndex {
    entries: Vec<SessionsIndexEntry>,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct SessionsIndexEntry {
    session_id: String,
    #[serde(default)]
    first_prompt: Option<String>,
    #[serde(default)]
    summary: Option<String>,
    #[serde(default)]
    message_count: Option<i64>,
    #[serde(default)]
    created: Option<String>,
    #[serde(default)]
    modified: Option<String>,
    #[serde(default)]
    git_branch: Option<String>,
    #[serde(default)]
    project_path: Option<String>,
    #[serde(default)]
    is_sidechain: Option<bool>,
    #[serde(default)]
    file_mtime: Option<u64>,
}

fn get_state_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".c0/sessions-state.json")
}

fn load_state() -> SessionsState {
    let path = get_state_path();
    if let Ok(content) = std::fs::read_to_string(&path) {
        serde_json::from_str(&content).unwrap_or_else(|_| SessionsState {
            dirs: HashMap::new(),
        })
    } else {
        SessionsState {
            dirs: HashMap::new(),
        }
    }
}

fn save_state(state: &SessionsState) -> Result<()> {
    let path = get_state_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let content = serde_json::to_string_pretty(state)?;
    std::fs::write(path, content)?;
    Ok(())
}

fn get_projects_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".claude/projects")
}

fn derive_namespace(dir_name: &str) -> String {
    let decoded = dir_name.replace('-', "/");
    let path = PathBuf::from(&decoded);

    let c0_config = path.join(".c0/config.toml");
    if let Ok(content) = std::fs::read_to_string(&c0_config) {
        if let Ok(config) = content.parse::<toml::Value>() {
            if let Some(ns) = config.get("namespace").and_then(|v| v.as_str()) {
                return ns.to_string();
            }
        }
    }

    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("/home"));
    if path == home {
        return "global".to_string();
    }

    path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("global")
        .to_string()
}

fn file_mtime_ms(path: &PathBuf) -> Option<u64> {
    std::fs::metadata(path)
        .ok()
        .and_then(|m| m.modified().ok())
        .map(|t| {
            t.duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64
        })
}

fn parse_first_prompt_from_jsonl(path: &PathBuf) -> Option<String> {
    let file = std::fs::File::open(path).ok()?;
    let reader = std::io::BufReader::new(file);
    use std::io::BufRead;

    for line in reader.lines().take(20) {
        let line = line.ok()?;
        if let Ok(obj) = serde_json::from_str::<serde_json::Value>(&line) {
            if obj.get("type").and_then(|t| t.as_str()) == Some("user") {
                if let Some(msg) = obj.get("message") {
                    if let Some(content) = msg.get("content").and_then(|c| c.as_str()) {
                        let truncated = if content.len() > 500 {
                            format!("{}...", &content[..500])
                        } else {
                            content.to_string()
                        };
                        return Some(truncated);
                    }
                    if let Some(content_arr) = msg.get("content").and_then(|c| c.as_array()) {
                        for block in content_arr {
                            if block.get("type").and_then(|t| t.as_str()) == Some("text") {
                                if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                                    let truncated = if text.len() > 500 {
                                        format!("{}...", &text[..500])
                                    } else {
                                        text.to_string()
                                    };
                                    return Some(truncated);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    None
}

fn parse_session_metadata_from_jsonl(
    path: &PathBuf,
) -> Option<(String, Option<String>, Option<bool>)> {
    let file = std::fs::File::open(path).ok()?;
    let reader = std::io::BufReader::new(file);
    use std::io::BufRead;

    let mut cwd = None;
    let mut git_branch = None;
    let mut is_sidechain = None;

    for line in reader.lines().take(10) {
        let line = line.ok()?;
        if let Ok(obj) = serde_json::from_str::<serde_json::Value>(&line) {
            if cwd.is_none() {
                if let Some(c) = obj.get("cwd").and_then(|v| v.as_str()) {
                    cwd = Some(c.to_string());
                }
            }
            if git_branch.is_none() {
                if let Some(b) = obj.get("gitBranch").and_then(|v| v.as_str()) {
                    if !b.is_empty() {
                        git_branch = Some(b.to_string());
                    }
                }
            }
            if is_sidechain.is_none() {
                if let Some(sc) = obj.get("isSidechain").and_then(|v| v.as_bool()) {
                    is_sidechain = Some(sc);
                }
            }
            if cwd.is_some() && git_branch.is_some() {
                break;
            }
        }
    }

    cwd.map(|c| (c, git_branch, is_sidechain))
}

pub async fn index_sessions() -> Result<()> {
    let projects_dir = get_projects_dir();
    if !projects_dir.exists() {
        println!("No Claude projects directory found.");
        return Ok(());
    }

    let semantic_config = config::SemanticConfig::load();
    let ollama_client = embeddings::OllamaClient::from_config(&semantic_config);

    let graph_conn = graph::connect().await?;
    let mut state = load_state();
    let mut indexed = 0u32;
    let mut skipped = 0u32;

    let entries: Vec<_> = std::fs::read_dir(&projects_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|ft| ft.is_dir()).unwrap_or(false))
        .collect();

    for entry in &entries {
        let dir_name = entry.file_name().to_string_lossy().to_string();
        let dir_path = entry.path();
        let namespace = derive_namespace(&dir_name);

        let index_path = dir_path.join("sessions-index.json");
        let dir_state = state
            .dirs
            .entry(dir_name.clone())
            .or_insert_with(|| DirState {
                index_mtime: None,
                sessions: HashMap::new(),
            });

        let mut indexed_ids: std::collections::HashSet<String> = std::collections::HashSet::new();

        if index_path.exists() {
            let current_mtime = file_mtime_ms(&index_path);
            let index_unchanged = dir_state.index_mtime == current_mtime && current_mtime.is_some();

            if let Ok(content) = std::fs::read_to_string(&index_path) {
                if let Ok(index) = serde_json::from_str::<SessionsIndex>(&content) {
                    for entry in &index.entries {
                        indexed_ids.insert(entry.session_id.clone());

                        let stored_mtime = dir_state.sessions.get(&entry.session_id).copied();
                        let entry_mtime = entry.file_mtime;

                        if stored_mtime == entry_mtime && entry_mtime.is_some() {
                            skipped += 1;
                            continue;
                        }

                        if index_unchanged {
                            skipped += 1;
                            continue;
                        }

                        let first_prompt = entry
                            .first_prompt
                            .as_deref()
                            .unwrap_or("No prompt")
                            .to_string();
                        let cwd = entry.project_path.as_deref().unwrap_or("").to_string();

                        let session = Session {
                            session_id: entry.session_id.clone(),
                            slug: None,
                            cwd,
                            namespace: namespace.clone(),
                            first_prompt: first_prompt.clone(),
                            summary: entry.summary.clone(),
                            git_branch: entry.git_branch.clone().filter(|b| !b.is_empty()),
                            created_at: entry.created.clone().unwrap_or_default(),
                            ended_at: entry.modified.clone(),
                            message_count: entry.message_count,
                            is_sidechain: entry.is_sidechain.unwrap_or(false),
                        };

                        let embed_text = format!(
                            "{} {}",
                            first_prompt,
                            entry.summary.as_deref().unwrap_or("")
                        );

                        let embedding = if let Some(ref client) = ollama_client {
                            client.embed(&embed_text).await.ok()
                        } else {
                            None
                        };

                        graph::add_session(&graph_conn, &session, embedding.as_deref()).await?;

                        if let Some(mtime) = entry_mtime {
                            dir_state.sessions.insert(entry.session_id.clone(), mtime);
                        }
                        indexed += 1;
                    }

                    dir_state.index_mtime = current_mtime;
                }
            }
        }

        let jsonl_files: Vec<_> = std::fs::read_dir(&dir_path)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|ext| ext.to_str()) == Some("jsonl"))
            .collect();

        for jsonl_entry in &jsonl_files {
            let jsonl_path = jsonl_entry.path();
            let session_id = jsonl_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();

            if session_id.is_empty() || indexed_ids.contains(&session_id) {
                continue;
            }

            let current_mtime = file_mtime_ms(&jsonl_path);
            let stored_mtime = dir_state.sessions.get(&session_id).copied();

            if stored_mtime == current_mtime && current_mtime.is_some() {
                skipped += 1;
                continue;
            }

            let first_prompt = parse_first_prompt_from_jsonl(&jsonl_path)
                .unwrap_or_else(|| "No prompt".to_string());

            let (cwd, git_branch, is_sidechain) = parse_session_metadata_from_jsonl(&jsonl_path)
                .unwrap_or_else(|| (String::new(), None, None));

            let created_at = std::fs::metadata(&jsonl_path)
                .ok()
                .and_then(|m| m.created().ok())
                .map(|t| {
                    let dt: chrono::DateTime<chrono::Utc> = t.into();
                    dt.to_rfc3339()
                })
                .unwrap_or_default();

            let session = Session {
                session_id: session_id.clone(),
                slug: None,
                cwd,
                namespace: namespace.clone(),
                first_prompt: first_prompt.clone(),
                summary: None,
                git_branch,
                created_at,
                ended_at: None,
                message_count: None,
                is_sidechain: is_sidechain.unwrap_or(false),
            };

            let embedding = if let Some(ref client) = ollama_client {
                client.embed(&first_prompt).await.ok()
            } else {
                None
            };

            graph::add_session(&graph_conn, &session, embedding.as_deref()).await?;

            if let Some(mtime) = current_mtime {
                dir_state.sessions.insert(session_id, mtime);
            }
            indexed += 1;
        }
    }

    save_state(&state)?;

    println!(
        "Indexed {indexed} sessions ({skipped} unchanged, {} dirs scanned)",
        entries.len()
    );
    Ok(())
}

pub async fn list_sessions(namespaces: &[String], limit: usize) -> Result<()> {
    let graph_conn = graph::connect().await?;
    let sessions = graph::get_sessions(&graph_conn, namespaces, limit).await?;

    if sessions.is_empty() {
        println!("No sessions indexed.");
        println!("\nRun 'c0 sessions index' to index sessions.");
        return Ok(());
    }

    println!(
        "{:<38} {:<12} {:<16} {}",
        "SESSION ID", "NAMESPACE", "CREATED", "FIRST PROMPT"
    );
    println!("{}", "-".repeat(100));

    for s in &sessions {
        let date = if s.created_at.len() >= 10 {
            &s.created_at[..10]
        } else {
            &s.created_at
        };
        let prompt = if s.first_prompt.len() > 40 {
            format!("{}...", &s.first_prompt[..40])
        } else {
            s.first_prompt.clone()
        };
        let display_name = s
            .summary
            .as_deref()
            .filter(|s| !s.is_empty())
            .unwrap_or(&prompt);
        let truncated = if display_name.len() > 50 {
            format!("{}...", &display_name[..50])
        } else {
            display_name.to_string()
        };

        println!(
            "{:<38} {:<12} {:<16} {}",
            s.session_id, s.namespace, date, truncated
        );
    }

    println!("\n{} sessions shown", sessions.len());
    Ok(())
}

pub async fn search_sessions(query: &str, namespaces: &[String], limit: usize) -> Result<()> {
    let semantic_config = config::SemanticConfig::load();
    let ollama_client = embeddings::OllamaClient::from_config(&semantic_config);

    let embedding = if let Some(ref client) = ollama_client {
        client.embed(query).await.ok()
    } else {
        None
    };

    let graph_conn = graph::connect().await?;
    let results =
        graph::search_sessions_hybrid(&graph_conn, query, embedding.as_deref(), limit, namespaces)
            .await?;

    if results.is_empty() {
        println!("No sessions found matching: \"{query}\"");
        return Ok(());
    }

    println!("Sessions matching: \"{query}\"\n");
    println!(
        "{:<6} {:<38} {:<12} {}",
        "SCORE", "SESSION ID", "NAMESPACE", "SUMMARY"
    );
    println!("{}", "-".repeat(100));

    for (s, score) in &results {
        let display = s
            .summary
            .as_deref()
            .filter(|v| !v.is_empty())
            .unwrap_or(&s.first_prompt);
        let truncated = if display.len() > 50 {
            format!("{}...", &display[..50])
        } else {
            display.to_string()
        };

        println!(
            "{:<6.2} {:<38} {:<12} {}",
            score, s.session_id, s.namespace, truncated
        );
    }

    println!("\n{} results", results.len());
    Ok(())
}

pub async fn resume_session(query: &str, namespaces: &[String]) -> Result<()> {
    let semantic_config = config::SemanticConfig::load();
    let ollama_client = embeddings::OllamaClient::from_config(&semantic_config);

    let embedding = if let Some(ref client) = ollama_client {
        client.embed(query).await.ok()
    } else {
        None
    };

    let graph_conn = graph::connect().await?;
    let results =
        graph::search_sessions_hybrid(&graph_conn, query, embedding.as_deref(), 1, namespaces)
            .await?;

    if let Some((session, score)) = results.first() {
        let display = session
            .summary
            .as_deref()
            .filter(|v| !v.is_empty())
            .unwrap_or(&session.first_prompt);
        let truncated = if display.len() > 80 {
            format!("{}...", &display[..80])
        } else {
            display.to_string()
        };

        println!("Best match ({score:.2}): {truncated}");
        if let Some(ref branch) = session.git_branch {
            println!("Branch: {branch}");
        }
        println!("\nclaude --resume {}", session.session_id);
    } else {
        println!("No sessions found matching: \"{query}\"");
    }

    Ok(())
}

const TURN_EMBED_MIN_CHARS: usize = 50;
const REFLECTION_EMBED_MIN_CHARS: usize = 50;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
struct TurnsState {
    dirs: HashMap<String, TurnsDirState>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
struct TurnsDirState {
    sessions: HashMap<String, u64>,
}

fn get_turns_state_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".c0/turns-state.json")
}

fn load_turns_state() -> TurnsState {
    let path = get_turns_state_path();
    std::fs::read_to_string(&path)
        .ok()
        .and_then(|c| serde_json::from_str(&c).ok())
        .unwrap_or_default()
}

fn save_turns_state(state: &TurnsState) -> Result<()> {
    let path = get_turns_state_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let content = serde_json::to_string_pretty(state)?;
    std::fs::write(path, content)?;
    Ok(())
}

#[derive(Debug, Clone, Default)]
pub struct ExtractStats {
    pub sessions_extracted: u32,
    pub sessions_skipped: u32,
    pub turns_emitted: u32,
    pub reflections_emitted: u32,
    pub toolcalls_emitted: u32,
    pub embed_calls: u32,
    pub errors: u32,
}

#[derive(Debug, Clone, Default)]
struct ParsedSession {
    session_id: String,
    namespace: String,
    cwd: String,
    git_branch: Option<String>,
    is_sidechain_overall: bool,
    first_prompt: String,
    summary: Option<String>,
    slug: Option<String>,
    created_at: String,
    ended_at: Option<String>,
    turns: Vec<ParsedTurn>,
    backfills: Vec<ToolResultBackfill>,
}

#[derive(Debug, Clone, Default)]
struct ParsedTurn {
    turn_id: String,
    role: String,
    text: String,
    model: Option<String>,
    timestamp: String,
    parent_turn_id: Option<String>,
    is_sidechain: bool,
    git_branch: Option<String>,
    cwd: Option<String>,
    input_tokens: Option<i64>,
    output_tokens: Option<i64>,
    cache_creation_tokens: Option<i64>,
    cache_read_tokens: Option<i64>,
    tool_use_count: i64,
    tool_use_names: Vec<String>,
    text_chars: i64,
    thinking_chars: i64,
    reflections: Vec<ParsedReflection>,
    toolcalls: Vec<ParsedToolCall>,
}

#[derive(Debug, Clone)]
struct ParsedReflection {
    reflection_id: String,
    text: String,
    signature: Option<String>,
}

#[derive(Debug, Clone)]
struct ParsedToolCall {
    tool_call_id: String,
    name: String,
    input_json: String,
    file_touches: Vec<FileTouch>,
    bash: Option<BashCall>,
}

fn ext_string(v: &serde_json::Value, key: &str) -> Option<String> {
    v.get(key)
        .and_then(|x| x.as_str())
        .map(std::string::ToString::to_string)
        .filter(|s| !s.is_empty())
}

fn ext_bool(v: &serde_json::Value, key: &str) -> Option<bool> {
    v.get(key).and_then(serde_json::Value::as_bool)
}

fn ext_i64(v: &serde_json::Value, key: &str) -> Option<i64> {
    v.get(key).and_then(serde_json::Value::as_i64)
}

fn touches_for_tool(name: &str, input: &serde_json::Value) -> Vec<FileTouch> {
    let action = match name {
        "Read" => "read",
        "Edit" | "MultiEdit" | "NotebookEdit" => "edit",
        "Write" => "write",
        "Grep" => "grep",
        "Glob" => "glob",
        _ => return Vec::new(),
    };

    let mut paths = Vec::new();
    if let Some(p) = input.get("file_path").and_then(|v| v.as_str())
        && !p.is_empty()
    {
        paths.push(p.to_string());
    }
    if let Some(p) = input.get("path").and_then(|v| v.as_str())
        && !p.is_empty()
        && !paths.iter().any(|q| q == p)
    {
        paths.push(p.to_string());
    }
    if let Some(p) = input.get("notebook_path").and_then(|v| v.as_str())
        && !p.is_empty()
    {
        paths.push(p.to_string());
    }

    paths
        .into_iter()
        .map(|p| FileTouch {
            path: p,
            action: action.to_string(),
        })
        .collect()
}

fn bash_call_for_tool(name: &str, input: &serde_json::Value) -> Option<BashCall> {
    if name != "Bash" {
        return None;
    }
    let cmd = input
        .get("command")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if cmd.is_empty() {
        return None;
    }
    let description = input
        .get("description")
        .and_then(|v| v.as_str())
        .map(std::string::ToString::to_string)
        .filter(|s| !s.is_empty());
    Some(BashCall { cmd, description })
}

fn extract_tool_result_text(content: &serde_json::Value) -> String {
    if let Some(s) = content.as_str() {
        return s.to_string();
    }
    if let Some(arr) = content.as_array() {
        let mut out = String::new();
        for block in arr {
            if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(t);
            }
        }
        return out;
    }
    String::new()
}

fn parse_jsonl_file(
    path: &PathBuf,
    namespace: &str,
    skip_sidechains: bool,
) -> Result<ParsedSession> {
    use std::io::BufRead;

    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);

    let session_id = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();

    let mut parsed = ParsedSession {
        session_id: session_id.clone(),
        namespace: namespace.to_string(),
        ..ParsedSession::default()
    };

    let created_at = std::fs::metadata(path)
        .ok()
        .and_then(|m| m.created().ok())
        .map(|t| {
            let dt: chrono::DateTime<chrono::Utc> = t.into();
            dt.to_rfc3339()
        })
        .unwrap_or_default();
    parsed.created_at = created_at;

    let mut last_timestamp: Option<String> = None;
    let mut first_user_prompt_set = false;

    for line in reader.lines() {
        let Ok(line) = line else { continue };
        if line.is_empty() {
            continue;
        }
        let Ok(obj) = serde_json::from_str::<serde_json::Value>(&line) else {
            continue;
        };

        let line_type = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");

        if parsed.cwd.is_empty()
            && let Some(c) = ext_string(&obj, "cwd")
        {
            parsed.cwd = c;
        }
        if parsed.git_branch.is_none() {
            parsed.git_branch = ext_string(&obj, "gitBranch");
        }

        if let Some(ts) = ext_string(&obj, "timestamp") {
            last_timestamp = Some(ts);
        }

        if line_type == "ai-title" {
            if let Some(t) = ext_string(&obj, "title") {
                parsed.slug = Some(t);
            }
            continue;
        }
        if line_type == "summary" {
            if let Some(s) = ext_string(&obj, "summary") {
                parsed.summary = Some(s);
            }
            continue;
        }

        if line_type != "user" && line_type != "assistant" {
            continue;
        }

        let is_sidechain = ext_bool(&obj, "isSidechain").unwrap_or(false);
        if is_sidechain {
            parsed.is_sidechain_overall = true;
            if skip_sidechains {
                continue;
            }
        }

        let Some(turn_id) = ext_string(&obj, "uuid") else {
            continue;
        };

        let parent_turn_id = ext_string(&obj, "parentUuid");
        let timestamp = ext_string(&obj, "timestamp").unwrap_or_default();
        let cwd = ext_string(&obj, "cwd");
        let git_branch = ext_string(&obj, "gitBranch");

        let msg = obj.get("message");
        let role = msg
            .and_then(|m| m.get("role"))
            .and_then(|r| r.as_str())
            .unwrap_or(line_type)
            .to_string();
        let model = msg.and_then(|m| ext_string(m, "model"));

        let usage = msg.and_then(|m| m.get("usage"));
        let input_tokens = usage.and_then(|u| ext_i64(u, "input_tokens"));
        let output_tokens = usage.and_then(|u| ext_i64(u, "output_tokens"));
        let cache_creation_tokens = usage.and_then(|u| ext_i64(u, "cache_creation_input_tokens"));
        let cache_read_tokens = usage.and_then(|u| ext_i64(u, "cache_read_input_tokens"));

        let mut turn = ParsedTurn {
            turn_id: turn_id.clone(),
            role,
            timestamp,
            parent_turn_id,
            is_sidechain,
            git_branch,
            cwd,
            model,
            input_tokens,
            output_tokens,
            cache_creation_tokens,
            cache_read_tokens,
            ..ParsedTurn::default()
        };

        let content = msg.and_then(|m| m.get("content"));

        if let Some(c) = content.and_then(|c| c.as_str()) {
            turn.text = c.to_string();
            turn.text_chars = c.len() as i64;
        } else if let Some(blocks) = content.and_then(|c| c.as_array()) {
            let mut text_buf = String::new();
            let mut reflection_idx = 0u32;
            let mut tool_names_seen: Vec<String> = Vec::new();

            for block in blocks {
                let block_type = block.get("type").and_then(|v| v.as_str()).unwrap_or("");
                match block_type {
                    "text" => {
                        if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
                            if !text_buf.is_empty() {
                                text_buf.push('\n');
                            }
                            text_buf.push_str(t);
                        }
                    }
                    "thinking" => {
                        if let Some(t) = block.get("thinking").and_then(|v| v.as_str()) {
                            if t.is_empty() {
                                continue;
                            }
                            let signature = ext_string(block, "signature");
                            turn.thinking_chars += t.len() as i64;
                            turn.reflections.push(ParsedReflection {
                                reflection_id: format!("{turn_id}#r{reflection_idx}"),
                                text: t.to_string(),
                                signature,
                            });
                            reflection_idx += 1;
                        }
                    }
                    "tool_use" => {
                        let id = ext_string(block, "id").unwrap_or_default();
                        if id.is_empty() {
                            continue;
                        }
                        let name = ext_string(block, "name").unwrap_or_default();
                        let input = block
                            .get("input")
                            .cloned()
                            .unwrap_or(serde_json::Value::Null);
                        let input_json = serde_json::to_string(&input).unwrap_or_default();
                        let file_touches = touches_for_tool(&name, &input);
                        let bash = bash_call_for_tool(&name, &input);

                        turn.tool_use_count += 1;
                        if !tool_names_seen.iter().any(|n| n == &name) {
                            tool_names_seen.push(name.clone());
                        }

                        turn.toolcalls.push(ParsedToolCall {
                            tool_call_id: id,
                            name,
                            input_json,
                            file_touches,
                            bash,
                        });
                    }
                    "tool_result" => {
                        let id = ext_string(block, "tool_use_id").unwrap_or_default();
                        if id.is_empty() {
                            continue;
                        }
                        let is_error = ext_bool(block, "is_error").unwrap_or(false);
                        let error_text = if is_error {
                            let raw = block
                                .get("content")
                                .map(extract_tool_result_text)
                                .unwrap_or_default();
                            if raw.is_empty() { None } else { Some(raw) }
                        } else {
                            None
                        };
                        parsed.backfills.push(ToolResultBackfill {
                            tool_call_id: id,
                            is_error,
                            error_text,
                        });
                    }
                    _ => {}
                }
            }

            turn.text_chars = text_buf.len() as i64;
            turn.text = text_buf;
            turn.tool_use_names = tool_names_seen;
        }

        if !first_user_prompt_set && turn.role == "user" && !turn.text.is_empty() {
            let snippet = if turn.text.len() > 4096 {
                &turn.text[..4096]
            } else {
                &turn.text
            };
            parsed.first_prompt = snippet.to_string();
            first_user_prompt_set = true;
        }

        parsed.turns.push(turn);
    }

    parsed.ended_at = last_timestamp;

    if parsed.first_prompt.is_empty() {
        parsed.first_prompt = "No prompt".to_string();
    }

    Ok(parsed)
}

async fn embed_text_opt(
    client: Option<&embeddings::OllamaClient>,
    text: &str,
    min_chars: usize,
) -> (Option<Vec<f32>>, u32) {
    if text.len() < min_chars {
        return (None, 0);
    }
    let Some(c) = client else {
        return (None, 0);
    };
    match c.embed(text).await {
        Ok(emb) => (Some(emb), 1),
        Err(_) => (None, 0),
    }
}

async fn write_parsed_session(
    graph_conn: &neo4rs::Graph,
    parsed: ParsedSession,
    ollama: Option<&embeddings::OllamaClient>,
    stats: &mut ExtractStats,
) -> Result<()> {
    let session = Session {
        session_id: parsed.session_id.clone(),
        slug: parsed.slug.clone(),
        cwd: parsed.cwd.clone(),
        namespace: parsed.namespace.clone(),
        first_prompt: parsed.first_prompt.clone(),
        summary: parsed.summary.clone(),
        git_branch: parsed.git_branch.clone(),
        created_at: parsed.created_at.clone(),
        ended_at: parsed.ended_at.clone(),
        message_count: Some(parsed.turns.len() as i64),
        is_sidechain: parsed.is_sidechain_overall,
    };

    let session_embed_text = format!(
        "{} {}",
        parsed.first_prompt,
        parsed.summary.as_deref().unwrap_or("")
    );
    let (session_emb, n) = embed_text_opt(ollama, &session_embed_text, TURN_EMBED_MIN_CHARS).await;
    stats.embed_calls += n;
    graph::add_session(graph_conn, &session, session_emb.as_deref()).await?;

    graph::delete_session_turns(graph_conn, &parsed.session_id).await?;

    let mut agg = SessionAggregates::default();

    for pturn in &parsed.turns {
        let (turn_emb, n) = embed_text_opt(ollama, &pturn.text, TURN_EMBED_MIN_CHARS).await;
        stats.embed_calls += n;

        let turn = Turn {
            turn_id: pturn.turn_id.clone(),
            session_id: parsed.session_id.clone(),
            namespace: parsed.namespace.clone(),
            role: pturn.role.clone(),
            text: pturn.text.clone(),
            model: pturn.model.clone(),
            timestamp: pturn.timestamp.clone(),
            parent_turn_id: pturn.parent_turn_id.clone(),
            is_sidechain: pturn.is_sidechain,
            git_branch: pturn.git_branch.clone(),
            cwd: pturn.cwd.clone(),
            input_tokens: pturn.input_tokens,
            output_tokens: pturn.output_tokens,
            cache_creation_tokens: pturn.cache_creation_tokens,
            cache_read_tokens: pturn.cache_read_tokens,
            tool_use_count: pturn.tool_use_count,
            tool_use_names: pturn.tool_use_names.clone(),
        };
        graph::add_turn(graph_conn, &turn, turn_emb.as_deref()).await?;
        stats.turns_emitted += 1;

        agg.total_turns += 1;
        agg.total_text_chars += pturn.text_chars;
        agg.total_thinking_chars += pturn.thinking_chars;
        agg.total_input_tokens += pturn.input_tokens.unwrap_or(0);
        agg.total_output_tokens += pturn.output_tokens.unwrap_or(0);
        agg.total_tool_calls += pturn.tool_use_count;

        for refl in &pturn.reflections {
            let (refl_emb, n) =
                embed_text_opt(ollama, &refl.text, REFLECTION_EMBED_MIN_CHARS).await;
            stats.embed_calls += n;
            let r = Reflection {
                reflection_id: refl.reflection_id.clone(),
                turn_id: pturn.turn_id.clone(),
                session_id: parsed.session_id.clone(),
                namespace: parsed.namespace.clone(),
                text: refl.text.clone(),
                signature: refl.signature.clone(),
                timestamp: pturn.timestamp.clone(),
            };
            graph::add_reflection(graph_conn, &r, refl_emb.as_deref()).await?;
            stats.reflections_emitted += 1;
        }

        for tc in &pturn.toolcalls {
            let record = ToolCallRecord {
                tool_call_id: tc.tool_call_id.clone(),
                turn_id: pturn.turn_id.clone(),
                session_id: parsed.session_id.clone(),
                namespace: parsed.namespace.clone(),
                name: tc.name.clone(),
                input_json: tc.input_json.clone(),
                timestamp: pturn.timestamp.clone(),
            };
            graph::add_toolcall(graph_conn, &record, &tc.file_touches, tc.bash.as_ref()).await?;
            stats.toolcalls_emitted += 1;
        }
    }

    for bf in &parsed.backfills {
        graph::backfill_toolcall_result(graph_conn, bf).await?;
    }

    graph::build_reply_chain(graph_conn, &parsed.session_id).await?;
    graph::update_session_aggregates(graph_conn, &parsed.session_id, &agg).await?;

    Ok(())
}

fn find_jsonl_for_session(session_id: &str) -> Option<(PathBuf, String)> {
    let projects_dir = get_projects_dir();
    let entries = std::fs::read_dir(&projects_dir).ok()?;
    for entry in entries.flatten() {
        if !entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
            continue;
        }
        let dir_name = entry.file_name().to_string_lossy().to_string();
        let candidate = entry.path().join(format!("{session_id}.jsonl"));
        if candidate.exists() {
            return Some((candidate, dir_name));
        }
    }
    None
}

pub async fn extract_session(
    session_id: &str,
    force: bool,
    skip_sidechains: bool,
) -> Result<ExtractStats> {
    let Some((path, dir_name)) = find_jsonl_for_session(session_id) else {
        anyhow::bail!("No JSONL file found for session {session_id} under ~/.claude/projects/");
    };

    let namespace = derive_namespace(&dir_name);
    let semantic_config = config::SemanticConfig::load();
    let ollama = embeddings::OllamaClient::from_config(&semantic_config);
    let graph_conn = graph::connect().await?;
    graph::ensure_turn_indexes(&graph_conn).await?;

    let mut state = load_turns_state();
    let dir_state = state.dirs.entry(dir_name.clone()).or_default();

    let current_mtime = file_mtime_ms(&path);
    let stored_mtime = dir_state.sessions.get(session_id).copied();

    let mut stats = ExtractStats::default();

    if !force && stored_mtime == current_mtime && current_mtime.is_some() {
        stats.sessions_skipped += 1;
        println!("Skipped {session_id} (unchanged since last extract). Use --force to rebuild.");
        return Ok(stats);
    }

    println!("Extracting session {session_id} (namespace: {namespace})");
    let parsed = parse_jsonl_file(&path, &namespace, skip_sidechains)?;
    println!(
        "  parsed {} turn(s), {} reflection(s), {} toolcall(s)",
        parsed.turns.len(),
        parsed
            .turns
            .iter()
            .map(|t| t.reflections.len())
            .sum::<usize>(),
        parsed
            .turns
            .iter()
            .map(|t| t.toolcalls.len())
            .sum::<usize>(),
    );

    write_parsed_session(&graph_conn, parsed, ollama.as_ref(), &mut stats).await?;
    stats.sessions_extracted += 1;

    if let Some(m) = current_mtime {
        dir_state.sessions.insert(session_id.to_string(), m);
    }
    save_turns_state(&state)?;

    println!(
        "  ✓ {} turns, {} reflections, {} toolcalls, {} embeds",
        stats.turns_emitted, stats.reflections_emitted, stats.toolcalls_emitted, stats.embed_calls
    );
    Ok(stats)
}

pub async fn extract_all(force: bool, skip_sidechains: bool) -> Result<ExtractStats> {
    let projects_dir = get_projects_dir();
    if !projects_dir.exists() {
        println!("No Claude projects directory found.");
        return Ok(ExtractStats::default());
    }

    let semantic_config = config::SemanticConfig::load();
    let ollama = embeddings::OllamaClient::from_config(&semantic_config);
    let graph_conn = graph::connect().await?;
    graph::ensure_turn_indexes(&graph_conn).await?;

    let mut state = load_turns_state();
    let mut stats = ExtractStats::default();

    let dirs: Vec<_> = std::fs::read_dir(&projects_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|ft| ft.is_dir()).unwrap_or(false))
        .collect();

    let total_dirs = dirs.len();
    println!("Scanning {total_dirs} project directories under ~/.claude/projects/");

    use std::io::Write;
    for entry in &dirs {
        let dir_name = entry.file_name().to_string_lossy().to_string();
        let namespace = derive_namespace(&dir_name);
        let dir_state = state.dirs.entry(dir_name.clone()).or_default();

        let jsonl_files: Vec<_> = match std::fs::read_dir(entry.path()) {
            Ok(rd) => rd
                .filter_map(std::result::Result::ok)
                .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("jsonl"))
                .collect(),
            Err(_) => continue,
        };

        for f in &jsonl_files {
            let path = f.path();
            let session_id = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();
            if session_id.is_empty() {
                continue;
            }

            let current_mtime = file_mtime_ms(&path);
            let stored_mtime = dir_state.sessions.get(&session_id).copied();

            if !force && stored_mtime == current_mtime && current_mtime.is_some() {
                stats.sessions_skipped += 1;
                continue;
            }

            match parse_jsonl_file(&path, &namespace, skip_sidechains) {
                Ok(parsed) => {
                    let n_turns = parsed.turns.len();
                    let n_refl: usize = parsed.turns.iter().map(|t| t.reflections.len()).sum();
                    let n_tc: usize = parsed.turns.iter().map(|t| t.toolcalls.len()).sum();
                    print!(
                        "  [{namespace}] {session_id} → {n_turns} turn(s), {n_refl} refl, {n_tc} tc ... "
                    );
                    std::io::stdout().flush().ok();

                    match write_parsed_session(&graph_conn, parsed, ollama.as_ref(), &mut stats)
                        .await
                    {
                        Ok(()) => {
                            stats.sessions_extracted += 1;
                            if let Some(m) = current_mtime {
                                dir_state.sessions.insert(session_id, m);
                            }
                            println!("✓");
                        }
                        Err(e) => {
                            stats.errors += 1;
                            println!("✗ {e}");
                        }
                    }
                }
                Err(e) => {
                    stats.errors += 1;
                    eprintln!("  parse error for {session_id}: {e}");
                }
            }
        }
    }

    save_turns_state(&state)?;

    println!();
    println!(
        "Extracted {} session(s) ({} skipped, {} errors)",
        stats.sessions_extracted, stats.sessions_skipped, stats.errors
    );
    println!(
        "Emitted {} turn(s), {} reflection(s), {} toolcall(s), {} embedding call(s)",
        stats.turns_emitted, stats.reflections_emitted, stats.toolcalls_emitted, stats.embed_calls
    );
    Ok(stats)
}

pub async fn search_turns(
    query: &str,
    namespaces: &[String],
    limit: usize,
    reflections: bool,
) -> Result<()> {
    let semantic_config = config::SemanticConfig::load();
    let ollama_client = embeddings::OllamaClient::from_config(&semantic_config);
    let embedding = if let Some(ref client) = ollama_client {
        client.embed(query).await.ok()
    } else {
        None
    };

    let graph_conn = graph::connect().await?;

    if reflections {
        let results = graph::search_reflections_hybrid(
            &graph_conn,
            query,
            embedding.as_deref(),
            limit,
            namespaces,
        )
        .await?;

        if results.is_empty() {
            println!("No reflections found matching: \"{query}\"");
            return Ok(());
        }

        println!("Reflections matching: \"{query}\"\n");
        for (r, score) in &results {
            let snippet = if r.text.len() > 240 {
                format!("{}...", &r.text[..240])
            } else {
                r.text.clone()
            };
            println!(
                "[{score:.2}] {} (session {})",
                r.namespace,
                &r.session_id[..8.min(r.session_id.len())]
            );
            println!("  {snippet}");
            println!();
        }
        println!("{} result(s)", results.len());
    } else {
        let results =
            graph::search_turns_hybrid(&graph_conn, query, embedding.as_deref(), limit, namespaces)
                .await?;

        if results.is_empty() {
            println!("No turns found matching: \"{query}\"");
            return Ok(());
        }

        println!("Turns matching: \"{query}\"\n");
        for (t, score) in &results {
            let snippet = if t.text.len() > 240 {
                format!("{}...", &t.text[..240])
            } else {
                t.text.clone()
            };
            let date = if t.timestamp.len() >= 10 {
                &t.timestamp[..10]
            } else {
                &t.timestamp
            };
            println!(
                "[{score:.2}] {} {} {} (session {})",
                t.role,
                date,
                t.namespace,
                &t.session_id[..8.min(t.session_id.len())]
            );
            println!("  {snippet}");
            println!();
        }
        println!("{} result(s)", results.len());
    }

    Ok(())
}

pub async fn list_session_costs(namespaces: &[String], limit: usize) -> Result<()> {
    let graph_conn = graph::connect().await?;
    let ns: Vec<&str> = namespaces.iter().map(String::as_str).collect();
    let filter_ns = !ns.is_empty();

    let cypher = if filter_ns {
        "MATCH (s:Session)
         WHERE s.namespace IN $namespaces AND s.deep_indexed_at IS NOT NULL
         RETURN s.session_id AS sid, s.namespace AS ns,
                coalesce(s.total_input_tokens, 0) AS in_tokens,
                coalesce(s.total_output_tokens, 0) AS out_tokens,
                coalesce(s.total_turns, 0) AS turns,
                s.first_prompt AS first_prompt, s.created_at AS created_at,
                s.summary AS summary
         ORDER BY (coalesce(s.total_input_tokens,0) + coalesce(s.total_output_tokens,0)) DESC
         LIMIT $limit"
    } else {
        "MATCH (s:Session)
         WHERE s.deep_indexed_at IS NOT NULL
         RETURN s.session_id AS sid, s.namespace AS ns,
                coalesce(s.total_input_tokens, 0) AS in_tokens,
                coalesce(s.total_output_tokens, 0) AS out_tokens,
                coalesce(s.total_turns, 0) AS turns,
                s.first_prompt AS first_prompt, s.created_at AS created_at,
                s.summary AS summary
         ORDER BY (coalesce(s.total_input_tokens,0) + coalesce(s.total_output_tokens,0)) DESC
         LIMIT $limit"
    };

    let mut result = graph_conn
        .execute(
            neo4rs::query(cypher)
                .param("namespaces", ns)
                .param("limit", limit as i64),
        )
        .await?;

    println!(
        "{:<10} {:<14} {:>10} {:>10} {:>6} {}",
        "DATE", "NAMESPACE", "IN_TOK", "OUT_TOK", "TURNS", "TITLE"
    );
    println!("{}", "-".repeat(100));

    let mut total_in: i64 = 0;
    let mut total_out: i64 = 0;
    let mut count = 0u32;
    while let Some(row) = result.next().await? {
        let sid: String = row.get("sid").unwrap_or_default();
        let _ = sid;
        let ns: String = row.get("ns").unwrap_or_default();
        let in_tok: i64 = row.get("in_tokens").unwrap_or(0);
        let out_tok: i64 = row.get("out_tokens").unwrap_or(0);
        let turns: i64 = row.get("turns").unwrap_or(0);
        let first_prompt: String = row.get("first_prompt").unwrap_or_default();
        let created_at: String = row.get("created_at").unwrap_or_default();
        let summary: String = row.get("summary").unwrap_or_default();

        let date = if created_at.len() >= 10 {
            &created_at[..10]
        } else {
            &created_at
        };
        let title_src = if !summary.is_empty() {
            &summary
        } else {
            &first_prompt
        };
        let title = if title_src.len() > 50 {
            format!("{}...", &title_src[..50])
        } else {
            title_src.to_string()
        };

        println!(
            "{:<10} {:<14} {:>10} {:>10} {:>6} {}",
            date, ns, in_tok, out_tok, turns, title
        );
        total_in += in_tok;
        total_out += out_tok;
        count += 1;
    }

    println!("{}", "-".repeat(100));
    println!(
        "Top {count} session(s) — totals: {total_in} input + {total_out} output tokens = {} combined",
        total_in + total_out
    );
    Ok(())
}

const ENRICHMENT_TEXT_BUDGET: usize = 8_000;
const DEFAULT_MAX_CONCEPTS_PER_SESSION: usize = 8;
const ENRICHMENT_OLLAMA_TIMEOUT_SECS: u64 = 600;
/// Fraction of the text budget reserved for the tool/file signal block.
const ENRICHMENT_SIGNAL_DIV: usize = 4;

fn enrichment_text_budget() -> usize {
    std::env::var("C0_ENRICH_TEXT_BUDGET")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(ENRICHMENT_TEXT_BUDGET)
}

/// When set (`C0_ENRICH_FULL=1`), extract over the whole session via map-reduce
/// instead of a single salience-selected window.
fn enrichment_full() -> bool {
    std::env::var("C0_ENRICH_FULL")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

fn enrichment_max_concepts() -> usize {
    std::env::var("C0_ENRICH_MAX_CONCEPTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MAX_CONCEPTS_PER_SESSION)
}

fn enrichment_ollama_timeout_secs() -> u64 {
    std::env::var("C0_ENRICH_TIMEOUT_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(ENRICHMENT_OLLAMA_TIMEOUT_SECS)
}

#[derive(Debug, Clone, Default)]
pub struct EnrichStats {
    pub sessions_enriched: u32,
    pub sessions_skipped: u32,
    pub concepts_created: u32,
    pub concepts_linked: u32,
    pub errors: u32,
}

#[derive(serde::Deserialize)]
struct OllamaGenerateResponse {
    response: String,
}

async fn extract_session_concepts_ollama(
    host: &str,
    model: &str,
    text: &str,
    max_concepts: usize,
) -> Result<Vec<ExtractedConcept>> {
    let prompt = format!(
        "You are a knowledge curator. Extract up to {max_concepts} distinct technology concepts, libraries, frameworks, patterns, or methodologies discussed in this Claude Code session text.\n\n\
         Each concept: name (lowercase kebab-case, alphanumeric + hyphens, 3-60 chars), description (one sentence, max 200 chars).\n\
         Skip generic terms (database, api, code, app, stuff). Skip pronouns and one-off names.\n\n\
         Return ONLY valid JSON: {{\"concepts\":[{{\"name\":\"...\",\"description\":\"...\"}}]}}\n\n\
         Session text:\n{text}"
    );

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(
            enrichment_ollama_timeout_secs(),
        ))
        .connect_timeout(std::time::Duration::from_secs(10))
        .build()?;

    let resp = client
        .post(format!("{host}/api/generate"))
        .json(&serde_json::json!({
            "model": model,
            "prompt": prompt,
            "stream": false,
            "format": "json"
        }))
        .send()
        .await
        .map_err(|e| {
            anyhow::anyhow!(
                "Ollama POST to {host} failed: {e:#} (source: {:?})",
                std::error::Error::source(&e)
            )
        })?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("Ollama returned {status}: {body}");
    }

    let parsed: OllamaGenerateResponse = resp
        .json()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to parse Ollama JSON: {e:#}"))?;
    parse_ollama_concepts(&parsed.response, max_concepts)
}

fn parse_ollama_concepts(raw: &str, max: usize) -> Result<Vec<ExtractedConcept>> {
    #[derive(serde::Deserialize)]
    struct R {
        #[serde(default)]
        concepts: Vec<ExtractedConcept>,
    }

    let cleaned = raw
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    let parsed: R = serde_json::from_str(cleaned)
        .map_err(|e| anyhow::anyhow!("Failed to parse Ollama concepts: {e}\nRaw: {cleaned}"))?;

    let bad = [
        "type", "system", "hook", "subtype", "empty", "none", "database", "api", "code", "app",
        "stuff", "things", "service", "domain", "generic", "term", "concept",
    ];

    let out: Vec<ExtractedConcept> = parsed
        .concepts
        .into_iter()
        .filter_map(|c| {
            let name = c.name.trim().to_lowercase().replace(' ', "-");
            if name.len() < 3 || name.len() > 60 {
                return None;
            }
            if !name.chars().all(|ch| ch.is_alphanumeric() || ch == '-') {
                return None;
            }
            if bad.iter().any(|p| name == *p) {
                return None;
            }
            let desc = c.description.trim().to_string();
            if desc.is_empty() {
                return None;
            }
            let description = if desc.len() > 220 {
                let mut end = 220;
                while !desc.is_char_boundary(end) && end > 0 {
                    end -= 1;
                }
                format!("{}...", &desc[..end])
            } else {
                desc
            };
            Some(ExtractedConcept { name, description })
        })
        .take(max)
        .collect();
    Ok(out)
}

async fn extract_session_concepts(
    text: &str,
    max_concepts: usize,
) -> Result<Vec<ExtractedConcept>> {
    let semantic_config = config::SemanticConfig::load();

    let client = LlmClient::for_task(
        &semantic_config,
        "enrichment",
        semantic_config.claude.timeout_secs,
    );

    if client.provider_name() == "ollama" {
        let model_override = std::env::var("C0_ENRICH_MODEL").ok();
        let model = model_override.unwrap_or_else(|| client.model.clone());
        return extract_session_concepts_ollama(
            &semantic_config.ollama_host,
            &model,
            text,
            max_concepts,
        )
        .await;
    }

    client.extract_session_concepts(text, max_concepts).await
}

/// Collapse a string to a single capped line for the signal block.
fn truncate_for_signal(s: &str, max: usize) -> String {
    let one_line = s.replace('\n', " ");
    if one_line.len() <= max {
        return one_line;
    }
    let mut end = max;
    while !one_line.is_char_boundary(end) && end > 0 {
        end -= 1;
    }
    format!("{}…", &one_line[..end])
}

/// Build the tool/file signal block: touched files and run commands carry dense
/// library/framework/tool names that prose often omits.
fn build_signal_block(files: &[String], commands: &[String], cap: usize) -> String {
    if cap == 0 || (files.is_empty() && commands.is_empty()) {
        return String::new();
    }
    let mut block = String::new();
    if !files.is_empty() {
        block.push_str("[files touched]\n");
        for path in files {
            let line = format!("- {path}\n");
            if block.len() + line.len() > cap {
                break;
            }
            block.push_str(&line);
        }
    }
    if !commands.is_empty() && block.len() < cap {
        block.push_str("[commands run]\n");
        for cmd in commands {
            let line = format!("- {}\n", truncate_for_signal(cmd, 120));
            if block.len() + line.len() > cap {
                break;
            }
            block.push_str(&line);
        }
    }
    block
}

/// Pick the most concept-bearing turns to fill `budget`, returning their indices
/// in chronological order. Ranks by similarity to the session centroid (most
/// representative turns), boosting turns that ran tools and the opening turn
/// (which usually states the task). Falls back to chronological fill when no
/// turn embeddings exist.
fn select_salient_turn_indices(turns: &[EnrichmentTurn], budget: usize) -> Vec<usize> {
    // Separator + role-label overhead per turn ("\n---\n[role]\n").
    const PER_TURN_OVERHEAD: usize = 12;

    let dim = turns
        .iter()
        .find_map(|t| t.embedding.as_ref().map(Vec::len))
        .unwrap_or(0);

    // No embeddings: replicate the original chronological-fill behaviour.
    if dim == 0 {
        let mut used = 0usize;
        let mut chosen = Vec::new();
        for (i, t) in turns.iter().enumerate() {
            let cost = t.text.len() + PER_TURN_OVERHEAD;
            if used + cost > budget && !chosen.is_empty() {
                break;
            }
            chosen.push(i);
            used += cost;
        }
        return chosen;
    }

    // Centroid over embedded turns.
    let mut centroid = vec![0f32; dim];
    let mut n = 0f32;
    for t in turns {
        if let Some(emb) = &t.embedding
            && emb.len() == dim
        {
            for (c, v) in centroid.iter_mut().zip(emb) {
                *c += v;
            }
            n += 1.0;
        }
    }
    if n > 0.0 {
        for c in &mut centroid {
            *c /= n;
        }
    }

    let mut scored: Vec<(usize, f32)> = turns
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let mut score = match &t.embedding {
                Some(emb) if emb.len() == dim => embeddings::cosine_similarity(emb, &centroid),
                // Short turns are unembedded; keep them eligible but low-priority.
                _ => 0.3,
            };
            if t.tool_use_count > 0 {
                score += 0.15;
            }
            if i == 0 && t.role.eq_ignore_ascii_case("user") {
                score += 0.25;
            }
            (i, score)
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut used = 0usize;
    let mut chosen = Vec::new();
    for (i, _) in scored {
        let cost = turns[i].text.len() + PER_TURN_OVERHEAD;
        // Always keep at least one turn; otherwise pack within budget.
        if used + cost > budget && !chosen.is_empty() {
            continue;
        }
        chosen.push(i);
        used += cost;
    }
    chosen.sort_unstable();
    chosen
}

/// Assemble the single-window enrichment text: summary seed + tool/file signal +
/// salience-selected turns, with reflections appended if room remains.
fn build_enrichment_text(inputs: &EnrichmentInputs, budget: usize) -> String {
    let signal = build_signal_block(
        &inputs.files,
        &inputs.commands,
        budget / ENRICHMENT_SIGNAL_DIV,
    );

    let mut sections: Vec<String> = Vec::new();
    if let Some(summary) = &inputs.summary {
        sections.push(format!("[summary]\n{}", truncate_for_signal(summary, 400)));
    }
    if !signal.is_empty() {
        sections.push(signal.trim_end().to_string());
    }

    let reserved: usize = sections.iter().map(|s| s.len() + 5).sum();
    let turn_budget = budget.saturating_sub(reserved);

    for i in select_salient_turn_indices(&inputs.turns, turn_budget) {
        let t = &inputs.turns[i];
        sections.push(format!("[{}]\n{}", t.role, t.text));
    }

    let mut text = sections.join("\n---\n");

    if text.len() < budget {
        for refl in &inputs.reflections {
            if text.len() >= budget {
                break;
            }
            text.push_str("\n---\n[thinking]\n");
            text.push_str(refl);
        }
    }

    if text.len() > budget {
        let mut end = budget;
        while !text.is_char_boundary(end) && end > 0 {
            end -= 1;
        }
        text.truncate(end);
    }
    text
}

/// Full-coverage map-reduce: chunk every turn chronologically into budget-sized
/// windows, extract concepts from each, and merge deduped by name.
async fn extract_concepts_full(
    inputs: &EnrichmentInputs,
    budget: usize,
    max_concepts: usize,
) -> Result<Vec<ExtractedConcept>> {
    let signal = build_signal_block(
        &inputs.files,
        &inputs.commands,
        budget / ENRICHMENT_SIGNAL_DIV,
    );

    let mut chunks: Vec<String> = Vec::new();
    let mut cur = String::new();
    for t in &inputs.turns {
        let piece = format!("[{}]\n{}", t.role, t.text);
        if !cur.is_empty() && cur.len() + piece.len() + 5 > budget {
            chunks.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push_str("\n---\n");
        }
        cur.push_str(&piece);
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }

    // Prepend the tool/file signal to the first chunk for shared context.
    if !signal.is_empty() {
        match chunks.first_mut() {
            Some(first) => *first = format!("{signal}\n---\n{first}"),
            None => chunks.push(signal),
        }
    }

    let total = chunks.len();
    let mut merged: Vec<ExtractedConcept> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (i, chunk) in chunks.iter().enumerate() {
        if chunk.trim().is_empty() {
            continue;
        }
        match extract_session_concepts(chunk, max_concepts).await {
            Ok(concepts) => {
                for c in concepts {
                    if seen.insert(c.name.clone()) {
                        merged.push(c);
                    }
                }
            }
            Err(e) => eprintln!("  chunk {}/{total} extraction failed: {e}", i + 1),
        }
    }
    Ok(merged)
}

/// Fetch a session's enrichment inputs and extract concepts, choosing the
/// salience-window or full map-reduce strategy. Returns the concepts plus the
/// character count fed to the model (0 means there was nothing to enrich).
async fn extract_concepts_for_session(
    graph_conn: &neo4rs::Graph,
    session_id: &str,
    budget: usize,
    max_concepts: usize,
) -> Result<(Vec<ExtractedConcept>, usize)> {
    let inputs = graph::get_session_enrichment_inputs(graph_conn, session_id).await?;
    if inputs.turns.is_empty() && inputs.reflections.is_empty() {
        return Ok((Vec::new(), 0));
    }

    if enrichment_full() {
        let total: usize = inputs.turns.iter().map(|t| t.text.len()).sum();
        let concepts = extract_concepts_full(&inputs, budget, max_concepts).await?;
        Ok((concepts, total))
    } else {
        let text = build_enrichment_text(&inputs, budget);
        if text.trim().is_empty() {
            return Ok((Vec::new(), 0));
        }
        let len = text.len();
        let concepts = extract_session_concepts(&text, max_concepts).await?;
        Ok((concepts, len))
    }
}

pub async fn enrich_session(session_id: &str, namespace: &str, force: bool) -> Result<EnrichStats> {
    let graph_conn = graph::connect().await?;
    let semantic_config = config::SemanticConfig::load();
    let ollama = embeddings::OllamaClient::from_config(&semantic_config);

    let mut stats = EnrichStats::default();

    if !force {
        let mut r = graph_conn
            .execute(
                neo4rs::query(
                    "MATCH (s:Session {session_id: $id})
                 RETURN s.enriched_at AS enriched_at, s.deep_indexed_at AS deep_at",
                )
                .param("id", session_id),
            )
            .await?;
        if let Some(row) = r.next().await? {
            let enriched: Option<String> = row.get("enriched_at").ok();
            let deep: Option<String> = row.get("deep_at").ok();
            if let (Some(e), Some(d)) = (enriched.as_ref(), deep.as_ref()) {
                if e >= d {
                    stats.sessions_skipped += 1;
                    println!("Skipped {session_id} (already enriched). Use --force to re-enrich.");
                    return Ok(stats);
                }
            }
        }
    }

    let (concepts, text_len) = match extract_concepts_for_session(
        &graph_conn,
        session_id,
        enrichment_text_budget(),
        enrichment_max_concepts(),
    )
    .await
    {
        Ok(v) => v,
        Err(e) => {
            stats.errors += 1;
            eprintln!("  LLM extraction failed: {e}");
            return Ok(stats);
        }
    };

    if text_len == 0 {
        println!("Session {session_id} has no text to enrich.");
        stats.sessions_skipped += 1;
        return Ok(stats);
    }

    let mode = if enrichment_full() { " [full]" } else { "" };
    println!("Enriching session {session_id} ({text_len} chars of text{mode})");

    println!("  extracted {} concept(s)", concepts.len());

    for concept in &concepts {
        let embedding = if let Some(ref client) = ollama {
            client
                .embed(&format!("{}: {}", concept.name, concept.description))
                .await
                .ok()
        } else {
            None
        };

        if let Err(e) = graph::add_concept(
            &graph_conn,
            &concept.name,
            namespace,
            Some(&concept.description),
            Some("session-enrichment"),
            None,
            embedding.as_deref(),
            None,
        )
        .await
        {
            eprintln!("  failed to add concept {}: {e}", concept.name);
            stats.errors += 1;
            continue;
        }
        stats.concepts_created += 1;

        if let Err(e) =
            graph::link_concept_to_session(&graph_conn, &concept.name, namespace, session_id, 1)
                .await
        {
            eprintln!("  failed to link concept {}: {e}", concept.name);
            stats.errors += 1;
            continue;
        }
        stats.concepts_linked += 1;
        println!("  • {} — {}", concept.name, concept.description);
    }

    graph::mark_session_enriched(&graph_conn, session_id, stats.concepts_linked as i64).await?;
    stats.sessions_enriched += 1;

    println!(
        "  ✓ {} concepts linked, {} errors",
        stats.concepts_linked, stats.errors
    );
    Ok(stats)
}

pub async fn enrich_all(namespaces: &[String], limit: usize, force: bool) -> Result<EnrichStats> {
    let graph_conn = graph::connect().await?;
    let semantic_config = config::SemanticConfig::load();
    let ollama = embeddings::OllamaClient::from_config(&semantic_config);

    let session_ids = if force {
        let mut result = graph_conn
            .execute(
                neo4rs::query(
                    "MATCH (s:Session)
                 WHERE s.deep_indexed_at IS NOT NULL
                   AND ($all_ns OR s.namespace IN $namespaces)
                 RETURN s.session_id AS id, s.namespace AS ns
                 ORDER BY s.deep_indexed_at DESC
                 LIMIT $limit",
                )
                .param(
                    "namespaces",
                    namespaces.iter().map(String::as_str).collect::<Vec<_>>(),
                )
                .param("all_ns", namespaces.is_empty())
                .param("limit", limit as i64),
            )
            .await?;
        let mut out = Vec::new();
        while let Some(row) = result.next().await? {
            let id: String = row.get("id").unwrap_or_default();
            let ns: String = row.get("ns").unwrap_or_default();
            if !id.is_empty() {
                out.push((id, ns));
            }
        }
        out
    } else {
        let ids = graph::get_unenriched_sessions(&graph_conn, namespaces, limit).await?;
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            let mut r = graph_conn
                .execute(
                    neo4rs::query("MATCH (s:Session {session_id: $id}) RETURN s.namespace AS ns")
                        .param("id", id.as_str()),
                )
                .await?;
            if let Some(row) = r.next().await? {
                let ns: String = row.get("ns").unwrap_or_default();
                out.push((id, ns));
            }
        }
        out
    };

    if session_ids.is_empty() {
        println!("No sessions to enrich.");
        return Ok(EnrichStats::default());
    }

    println!("Enriching {} session(s)...", session_ids.len());

    let mut total = EnrichStats::default();

    for (sid, ns) in &session_ids {
        let short = &sid[..8.min(sid.len())];
        let (concepts, text_len) = match extract_concepts_for_session(
            &graph_conn,
            sid,
            enrichment_text_budget(),
            enrichment_max_concepts(),
        )
        .await
        {
            Ok(v) => v,
            Err(e) => {
                println!("  [{ns}] {short} ✗ extract: {e}");
                total.errors += 1;
                continue;
            }
        };

        if text_len == 0 {
            total.sessions_skipped += 1;
            continue;
        }

        let mut linked = 0u32;
        for concept in &concepts {
            let embedding = if let Some(ref client) = ollama {
                client
                    .embed(&format!("{}: {}", concept.name, concept.description))
                    .await
                    .ok()
            } else {
                None
            };
            let added = graph::add_concept(
                &graph_conn,
                &concept.name,
                ns,
                Some(&concept.description),
                Some("session-enrichment"),
                None,
                embedding.as_deref(),
                None,
            )
            .await;
            if added.is_err() {
                total.errors += 1;
                continue;
            }
            total.concepts_created += 1;
            if graph::link_concept_to_session(&graph_conn, &concept.name, ns, sid, 1)
                .await
                .is_ok()
            {
                linked += 1;
                total.concepts_linked += 1;
            }
        }

        graph::mark_session_enriched(&graph_conn, sid, i64::from(linked)).await?;
        total.sessions_enriched += 1;
        println!("  [{ns}] {short} ({text_len} chars) ✓ {linked} concept(s)");
    }

    println!();
    println!(
        "Enriched {} session(s), {} concepts linked, {} errors, {} skipped",
        total.sessions_enriched, total.concepts_linked, total.errors, total.sessions_skipped
    );
    Ok(total)
}

#[cfg(test)]
mod enrichment_tests {
    use super::*;

    fn turn(role: &str, text: &str, tools: i64, emb: Option<Vec<f32>>) -> EnrichmentTurn {
        EnrichmentTurn {
            role: role.to_string(),
            text: text.to_string(),
            tool_use_count: tools,
            embedding: emb,
        }
    }

    #[test]
    fn truncate_for_signal_collapses_and_caps() {
        assert_eq!(truncate_for_signal("a\nb c", 10), "a b c");
        let t = truncate_for_signal("abcdefghij", 5);
        assert!(t.starts_with("abcde") && t.ends_with('…'));
    }

    #[test]
    fn signal_block_lists_files_and_commands_within_cap() {
        let files = vec!["src/main.rs".to_string(), "Cargo.toml".to_string()];
        let cmds = vec!["cargo build".to_string()];
        let block = build_signal_block(&files, &cmds, 200);
        assert!(block.contains("[files touched]"));
        assert!(block.contains("src/main.rs"));
        assert!(block.contains("[commands run]"));
        assert!(block.contains("cargo build"));

        // A zero cap yields nothing.
        assert_eq!(build_signal_block(&files, &cmds, 0), "");
    }

    #[test]
    fn salience_without_embeddings_is_chronological_fill() {
        let turns = vec![
            turn("user", "first", 0, None),
            turn("assistant", "second", 0, None),
            turn("user", "third", 0, None),
        ];
        // Generous budget keeps every turn, in order.
        assert_eq!(select_salient_turn_indices(&turns, 10_000), vec![0, 1, 2]);
    }

    #[test]
    fn salience_keeps_at_least_one_turn_when_over_budget() {
        let turns = vec![turn("user", &"x".repeat(500), 0, None)];
        let picked = select_salient_turn_indices(&turns, 10);
        assert_eq!(picked, vec![0]);
    }

    #[test]
    fn salience_ranks_representative_turns_first() {
        // Centroid sits near [1,0]; the off-axis turn should be dropped first
        // when the budget only fits two of the three turns.
        let turns = vec![
            turn("user", &"a".repeat(100), 0, Some(vec![1.0, 0.0])),
            turn("assistant", &"b".repeat(100), 0, Some(vec![0.9, 0.1])),
            turn("assistant", &"c".repeat(100), 0, Some(vec![0.0, 1.0])),
        ];
        let picked = select_salient_turn_indices(&turns, 230);
        assert!(picked.contains(&0));
        assert!(!picked.contains(&2));
        // Output stays chronological.
        let mut sorted = picked.clone();
        sorted.sort_unstable();
        assert_eq!(picked, sorted);
    }

    #[test]
    fn enrichment_text_includes_summary_and_signal() {
        let inputs = EnrichmentInputs {
            summary: Some("Refactor enrichment selection".to_string()),
            turns: vec![turn("user", "let's optimize the enrichment", 0, None)],
            reflections: vec![],
            files: vec!["src/sessions.rs".to_string()],
            commands: vec!["cargo test".to_string()],
        };
        let text = build_enrichment_text(&inputs, 8_000);
        assert!(text.contains("[summary]"));
        assert!(text.contains("Refactor enrichment selection"));
        assert!(text.contains("src/sessions.rs"));
        assert!(text.contains("let's optimize"));
        assert!(text.len() <= 8_000);
    }
}
