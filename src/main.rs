mod audit;
mod claude;
mod config;
mod embeddings;
mod export;
mod extract;
mod fetch;
mod graph;
mod reflector;
#[cfg(feature = "sessions")]
mod sessions;

use anyhow::Result;
use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use clap::{Parser, Subcommand};
use std::io::Write;
use std::time::{Duration, Instant};
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

fn parse_date_to_datetime(date_str: &str) -> anyhow::Result<DateTime<Utc>> {
    if let Ok(dt) = DateTime::parse_from_rfc3339(date_str) {
        return Ok(dt.with_timezone(&Utc));
    }
    if let Ok(naive_date) = NaiveDate::parse_from_str(date_str, "%Y-%m-%d") {
        let midnight = naive_date
            .and_hms_opt(0, 0, 0)
            .ok_or_else(|| anyhow::anyhow!("Failed to create midnight time"))?;
        return Ok(Utc.from_utc_datetime(&midnight));
    }
    anyhow::bail!("Invalid date format. Use YYYY-MM-DD or ISO 8601 format.")
}

#[derive(Parser)]
#[command(name = "c0")]
#[command(about = "Claude's thinking system - graph-based knowledge traversal")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Add {
        #[command(subcommand)]
        what: AddCommands,
    },
    Relate {
        from: String,
        rel_type: String,
        to: String,
    },
    Walk {
        start: String,
        #[arg(long, default_value = "2")]
        depth: u32,
        #[arg(long, short)]
        context: Option<String>,
        #[arg(long, short)]
        live: bool,
        #[arg(long, help = "Point-in-time query (ISO 8601 date)")]
        as_of: Option<String>,
        #[arg(long, help = "Include expired/invalidated knowledge")]
        include_expired: bool,
    },
    Find {
        pattern: String,
    },
    Link {
        #[command(subcommand)]
        what: LinkCommands,
    },
    List {
        #[command(subcommand)]
        what: ListCommands,
    },
    Trigger {
        #[command(subcommand)]
        action: TriggerCommands,
    },
    Config {
        #[command(subcommand)]
        action: ConfigCommands,
    },
    Init {
        #[arg(long)]
        namespace: String,
        #[arg(long)]
        parent: Option<String>,
        #[arg(long, default_value = "default")]
        r#type: String,
    },
    #[cfg(feature = "sessions")]
    Sessions {
        #[command(subcommand)]
        action: Option<SessionCommands>,
    },
    #[cfg(feature = "sessions")]
    Turns {
        #[command(subcommand)]
        action: TurnsCommands,
    },
    Migrate,
    Status,
    Describe {
        concept: String,
        description: String,
    },
    Reflector {
        #[command(subcommand)]
        action: ReflectorCommands,
    },
    Fetch {
        query: String,
        #[arg(long, short, default_value = "3")]
        limit: usize,
        #[arg(long)]
        all: bool,
        #[arg(long, help = "Bypass cache and fetch fresh content")]
        fresh: bool,
    },
    Cache {
        #[command(subcommand)]
        action: CacheCommands,
    },
    Backfill {
        #[command(subcommand)]
        what: BackfillCommands,
    },
    Extract {
        input: String,
        #[arg(long)]
        index: bool,
        #[arg(long)]
        dry_run: bool,
    },
    Invalidate {
        #[command(subcommand)]
        what: InvalidateCommands,
    },
    Supersede {
        old: String,
        #[arg(long)]
        with: String,
        #[arg(long, help = "When the supersession occurred (ISO 8601 date)")]
        as_of: Option<String>,
    },
    ExtractConcepts {
        prompt: String,
        #[arg(long, default_value = "3")]
        limit: usize,
        #[arg(long, help = "Only return concepts that exist in the graph")]
        known_only: bool,
        #[arg(long, help = "Output JSON format")]
        json: bool,
    },
    InvalidationChain {
        name: String,
    },
    Audit {
        #[command(subcommand)]
        action: AuditCommands,
    },
    Move {
        #[command(subcommand)]
        what: MoveCommands,
    },
    Search {
        query: String,
        #[arg(long, short, default_value = "10")]
        limit: usize,
        #[arg(long, short, default_value = "0.3")]
        threshold: f32,
        #[arg(long)]
        json: bool,
        #[arg(long, help = "Force vector-only search (no hybrid)")]
        vector_only: bool,
        #[arg(long, help = "Force keyword-only search (no embedding)")]
        keyword_only: bool,
    },
    Health {
        #[arg(long)]
        json: bool,
        #[arg(long, help = "Auto-fix: backfill embeddings, clear broken refs")]
        fix: bool,
    },
    Export {
        #[arg(long, short, default_value = "json")]
        format: String,
        #[arg(long, short)]
        namespace: Option<String>,
        #[arg(long, short)]
        output: Option<String>,
        #[arg(long, help = "Exclude embedding vectors to reduce size")]
        no_embeddings: bool,
    },
}

#[derive(Subcommand)]
enum CacheCommands {
    Clear,
}

#[derive(Subcommand)]
enum BackfillCommands {
    Embeddings {
        #[arg(long)]
        dry_run: bool,
    },
    /// Inline file-backed patch content into the graph so patches resolve from
    /// any machine, and recover patches with missing/relative patch_file refs.
    PatchContent {
        #[arg(long)]
        dry_run: bool,
    },
}

#[derive(Subcommand)]
enum AddCommands {
    Concept {
        name: String,
        #[arg(long, short = 'd')]
        description: Option<String>,
        #[arg(long)]
        source: Option<String>,
        #[arg(long)]
        url: Option<String>,
        #[arg(long, short = 'f')]
        force: bool,
        #[arg(long, short = 't')]
        to: Option<String>,
        #[arg(long, help = "When this knowledge became true (ISO 8601 date)")]
        valid_at: Option<String>,
    },
    Patch {
        name: String,
        #[arg(long)]
        corrects: Option<String>,
        #[arg(long)]
        file: Option<String>,
        #[arg(long)]
        content: Option<String>,
        #[arg(long)]
        source: Option<String>,
        #[arg(long)]
        url: Option<String>,
        #[arg(long, short = 't')]
        to: Option<String>,
        #[arg(long, help = "When this knowledge became true (ISO 8601 date)")]
        valid_at: Option<String>,
    },
}

#[derive(Subcommand)]
enum LinkCommands {
    Patch {
        patch: String,
        concept: String,
    },
    Source {
        #[command(subcommand)]
        action: SourceCommands,
    },
}

#[derive(Subcommand)]
enum SourceCommands {
    Add {
        name: String,
        #[arg(long)]
        url: String,
        #[arg(long, short = 't')]
        r#type: Option<String>,
        #[arg(long, short)]
        concept: Option<String>,
    },
    Remove {
        name: String,
    },
    Refresh {
        name: Option<String>,
        #[arg(long)]
        all: bool,
    },
    Fetch {
        name: String,
    },
    Search {
        query: String,
        #[arg(long, short, default_value = "5")]
        limit: usize,
        #[arg(long)]
        fetch: bool,
    },
}

#[derive(Subcommand)]
enum ListCommands {
    Patches,
    Triggers {
        #[arg(long)]
        semantic: bool,
    },
    Sources,
}

#[derive(Subcommand)]
enum TriggerCommands {
    Add {
        pattern: String,
        #[arg(long)]
        semantic: bool,
        #[arg(long, short)]
        description: Option<String>,
        #[arg(long)]
        threshold: Option<f32>,
        #[arg(long)]
        no_enrich: bool,
    },
    Remove {
        pattern: String,
        #[arg(long)]
        semantic: bool,
    },
    Test {
        prompt: String,
    },
    Match {
        prompt: String,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
enum ConfigCommands {
    TestOllama,
    Show,
}

#[derive(Subcommand)]
enum ReflectorCommands {
    Status,
    Inbox,
    Proposed,
    Review,
    Apply,
    Clear,
    Process,
    Notify,
    /// Run the loop continuously: classify the inbox, apply commits, then sleep.
    Run {
        /// How often to tick, e.g. 30s, 5m, 1h (a bare number means seconds).
        #[arg(long, default_value = "1h")]
        interval: String,
        /// Classify only; leave COMMIT decisions in the queue for human review.
        #[arg(long)]
        no_apply: bool,
    },
}

#[cfg(feature = "sessions")]
#[derive(Subcommand)]
enum SessionCommands {
    Index,
    Search {
        query: String,
        #[arg(long, short, default_value = "10")]
        limit: usize,
        #[arg(long, short, help = "Search across all namespaces")]
        all: bool,
    },
    Resume {
        query: String,
        #[arg(long, short, help = "Search across all namespaces")]
        all: bool,
    },
    Extract {
        #[arg(long, help = "Extract a single session by ID (default: extract all)")]
        session: Option<String>,
        #[arg(
            long,
            help = "Re-extract even if the JSONL hasn't changed since last run"
        )]
        force: bool,
        #[arg(long, help = "Skip turns from sidechain (subagent) sessions")]
        skip_sidechains: bool,
    },
    Enrich {
        #[arg(
            long,
            help = "Enrich a single session by ID (default: enrich unenriched sessions)"
        )]
        session: Option<String>,
        #[arg(
            long,
            short,
            default_value = "20",
            help = "Max sessions to enrich in this run"
        )]
        limit: usize,
        #[arg(long, help = "Re-enrich even if already enriched")]
        force: bool,
        #[arg(long, short, help = "Enrich across all namespaces")]
        all: bool,
    },
    Cost {
        #[arg(
            long,
            short,
            default_value = "20",
            help = "Number of top sessions to display"
        )]
        limit: usize,
        #[arg(long, short, help = "Show all namespaces")]
        all: bool,
    },
}

#[cfg(feature = "sessions")]
#[derive(Subcommand)]
enum TurnsCommands {
    Search {
        query: String,
        #[arg(long, short, default_value = "10")]
        limit: usize,
        #[arg(long, help = "Search :Reflection (thinking blocks) instead of :Turn")]
        reflections: bool,
        #[arg(long, short, help = "Search across all namespaces")]
        all: bool,
    },
}

#[derive(Subcommand)]
enum InvalidateCommands {
    Concept {
        name: String,
        #[arg(long, help = "When the knowledge became invalid (ISO 8601 date)")]
        as_of: Option<String>,
        #[arg(long, help = "Concept or event that caused this to become invalid")]
        by: Option<String>,
        #[arg(long, help = "Reason for invalidation")]
        reason: Option<String>,
    },
    Patch {
        name: String,
        #[arg(long, help = "When the patch became invalid (ISO 8601 date)")]
        as_of: Option<String>,
        #[arg(long, help = "Concept or event that caused this to become invalid")]
        by: Option<String>,
        #[arg(long, help = "Reason for invalidation")]
        reason: Option<String>,
    },
}

#[derive(Subcommand)]
enum AuditCommands {
    Staleness {
        #[arg(long, help = "Namespace to audit (defaults to current)")]
        namespace: Option<String>,
        #[arg(long, default_value = "90", help = "Days threshold for staleness")]
        days: u32,
        #[arg(long, help = "Output JSON format")]
        json: bool,
    },
    Namespaces {
        #[arg(long, help = "Show suggested namespace moves")]
        suggest: bool,
        #[arg(long, help = "Output JSON format")]
        json: bool,
    },
    All {
        #[arg(long, help = "Output JSON format")]
        json: bool,
    },
    /// Connect orphaned concepts to their nearest semantic neighbours.
    Enrich {
        #[arg(long, help = "Namespace to enrich (defaults to current)")]
        namespace: Option<String>,
        #[arg(long, short, help = "Enrich every known namespace")]
        all: bool,
        #[arg(
            long,
            default_value = "0.82",
            help = "Min cosine similarity for same-namespace links"
        )]
        same_threshold: f32,
        #[arg(
            long,
            default_value = "0.90",
            help = "Min cosine similarity for a cross-namespace bridge"
        )]
        cross_threshold: f32,
        #[arg(
            long,
            default_value = "2",
            help = "Max same-namespace links added per orphan"
        )]
        max_links: usize,
        #[arg(long, help = "Show proposed edges without writing them")]
        dry_run: bool,
        #[arg(
            long,
            help = "Delete auto-enriched edges; optionally pass a run-id (default: most recent)",
            num_args = 0..=1,
            default_missing_value = ""
        )]
        rollback: Option<String>,
        #[arg(long, help = "Output JSON format")]
        json: bool,
    },
}

#[derive(Subcommand)]
enum MoveCommands {
    Concept {
        name: String,
        #[arg(long, help = "Target namespace")]
        to: String,
        #[arg(long, help = "Also move associated patches")]
        with_patches: bool,
    },
    Prefix {
        prefix: String,
        #[arg(long, help = "Target namespace")]
        to: String,
        #[arg(long, default_value = "global", help = "Source namespace")]
        from: String,
        #[arg(long, help = "Also move associated patches")]
        with_patches: bool,
        #[arg(long, help = "Preview only, don't actually move")]
        dry_run: bool,
    },
}

/// Depth-bounded scan for `patches/` directories under `root`. Descends into
/// `.c0` among dotdirs but skips other hidden dirs and known-heavy trees.
fn discover_patch_dirs(
    root: &std::path::Path,
    max_depth: usize,
    out: &mut Vec<std::path::PathBuf>,
) {
    if max_depth == 0 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = entry.file_name();
        if let Some(s) = name.to_str() {
            if s.starts_with('.') && s != ".c0" {
                continue;
            }
            if matches!(
                s,
                "node_modules" | "target" | "cargo" | "rustup" | "go" | ".git"
            ) {
                continue;
            }
            if s == "patches" && !out.contains(&path) {
                out.push(path.clone());
            }
        }
        discover_patch_dirs(&path, max_depth - 1, out);
    }
}

/// Render a patch's body. A `--file` ref wins (live re-read keeps refresh
/// semantics) but falls back to the inline `content` copy if the file moved or
/// was deleted, instead of leaving the patch as `[Error reading …]`.
fn print_patch_body(patch: &graph::Patch) {
    let body = patch
        .file
        .as_ref()
        .and_then(|f| std::fs::read_to_string(shellexpand::tilde(f).as_ref()).ok())
        .or_else(|| patch.content.clone());
    match body {
        Some(c) => println!("{c}"),
        None => match &patch.file {
            Some(f) => println!(
                "[patch '{}': could not read {f} and no inline content]",
                patch.name
            ),
            None => println!("[patch '{}' has no readable content]", patch.name),
        },
    }
}

fn read_triggers(ctx: &config::NamespaceContext) -> Vec<String> {
    let files = config::get_triggers_files(ctx);
    let mut triggers = Vec::new();

    for file in files {
        if let Ok(content) = std::fs::read_to_string(&file) {
            for line in content.lines() {
                if !line.is_empty() && !line.starts_with('#') {
                    triggers.push(line.to_string());
                }
            }
        }
    }
    triggers
}

fn write_triggers(triggers: &[String], ctx: &config::NamespaceContext) {
    let file = if let Some(ref project_dir) = ctx.project_dir {
        project_dir.join("triggers.txt")
    } else {
        dirs::home_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join(".c0/triggers.txt")
    };

    let header = format!(
        "# c0 Memory Trigger Patterns ({})\n# Updated via: c0 trigger add/remove\n\n",
        ctx.namespace
    );
    let content = format!("{}{}", header, triggers.join("\n"));
    let _ = std::fs::write(file, content);
}

fn build_enrichment_prompt(
    trigger_name: &str,
    base_description: &str,
    namespace: &str,
    related_concepts: &[String],
) -> String {
    let related_str = if related_concepts.is_empty() {
        String::new()
    } else {
        format!(
            "\nRelated concepts in knowledge graph: {}",
            related_concepts.join(", ")
        )
    };

    format!(
        r#"You are expanding a semantic trigger for a knowledge retrieval system.

Input: "{base_description}" (topic: {trigger_name}, project: {namespace}){related_str}

Task: Write a dense, comma-separated list of synonyms, related terms, and common phrasings that should match this trigger. Include technical terms specific to this domain. Max 200 characters.

Example input: "building apps"
Example output: app development, create application, make software, build mobile app, develop web app, software creation

Your output (ONLY the expanded terms, nothing else):"#
    )
}

async fn enrich_with_ollama(
    config: &config::SemanticConfig,
    prompt: &str,
) -> anyhow::Result<String> {
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .build()?;
    let resp = client
        .post(format!("{}/api/generate", config.ollama_host))
        .json(&serde_json::json!({
            "model": config.reflector_model,
            "prompt": prompt,
            "stream": false
        }))
        .timeout(std::time::Duration::from_secs(60))
        .send()
        .await?;

    #[derive(serde::Deserialize)]
    struct OllamaResponse {
        response: String,
    }

    let ollama_resp: OllamaResponse = resp.json().await?;
    Ok(ollama_resp.response.trim().to_string())
}

async fn enrich_with_llm(config: &config::SemanticConfig, prompt: &str) -> anyhow::Result<String> {
    let client = claude::LlmClient::for_task(config, "enrichment", config.claude.timeout_secs);

    let response = client.generate(prompt, None).await?;

    if let Some(cost) = response.total_cost_usd {
        claude::log_usage("enrichment", &client.model, cost);
    }

    Ok(response.result.trim().to_string())
}

async fn enrich_trigger_description(
    config: &config::SemanticConfig,
    trigger_name: &str,
    base_description: &str,
    namespace: &str,
    related_concepts: &[String],
) -> anyhow::Result<String> {
    let prompt =
        build_enrichment_prompt(trigger_name, base_description, namespace, related_concepts);

    let enriched = if config.claude.provider_for("enrichment") == "ollama" {
        enrich_with_ollama(config, &prompt).await?
    } else {
        enrich_with_llm(config, &prompt).await?
    };

    let truncated = if enriched.len() > 250 {
        enriched.chars().take(250).collect()
    } else {
        enriched
    };

    Ok(truncated)
}

fn extract_topic(prompt: &str, pattern: &str) -> String {
    if let Ok(re) = regex::Regex::new(pattern)
        && let Some(m) = re.find(prompt)
    {
        let matched = m.as_str();
        let cleaned = matched
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '-' })
            .collect::<String>()
            .trim_matches('-')
            .to_string();
        if !cleaned.is_empty() && cleaned.len() <= 50 {
            return cleaned;
        }
    }

    let simple = pattern
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == ' ' || *c == '-')
        .collect::<String>()
        .split_whitespace()
        .next()
        .unwrap_or(pattern)
        .to_string();

    if simple.is_empty() || simple.len() > 50 {
        "topic".to_string()
    } else {
        simple
    }
}

fn get_session_id() -> Option<String> {
    std::env::var("C0_SESSION").ok().or_else(|| {
        dirs::home_dir()
            .and_then(|home| std::fs::read_to_string(home.join(".c0/current-session")).ok())
            .map(|s| s.trim().to_string())
    })
}

fn log_dead_end(command: &str, query: &str, namespace: &str, context: Option<&str>) {
    eprintln!("DEAD_END:{command}:{query}");

    let session_id = get_session_id().unwrap_or_else(|| "unknown".to_string());

    if let Some(home) = dirs::home_dir() {
        let log_dir = home.join(".c0/sessions").join(&session_id);
        if std::fs::create_dir_all(&log_dir).is_ok() {
            let log_file = log_dir.join("dead-ends.log");
            let entry = format!(
                "{}:{}:{}\n",
                chrono::Utc::now().to_rfc3339(),
                command,
                query
            );
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(log_file)
            {
                let _ = f.write_all(entry.as_bytes());
            }
        }

        let inbox_dir = home.join(".c0/reflector");
        if std::fs::create_dir_all(&inbox_dir).is_ok() {
            let inbox_file = inbox_dir.join("inbox.jsonl");
            let entry = serde_json::json!({
                "timestamp": chrono::Utc::now().to_rfc3339(),
                "session": session_id,
                "namespace": namespace,
                "command": command,
                "query": query,
                "context": context,
            });
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(inbox_file)
            {
                let _ = writeln!(f, "{entry}");
            }
        }
    }
}

fn cmd_kind(c: &Commands) -> &'static str {
    match c {
        Commands::Add { .. } => "add",
        Commands::Relate { .. } => "relate",
        Commands::Walk { .. } => "walk",
        Commands::Find { .. } => "find",
        Commands::Link { .. } => "link",
        Commands::List { .. } => "list",
        Commands::Trigger { .. } => "trigger",
        Commands::Config { .. } => "config",
        Commands::Init { .. } => "init",
        #[cfg(feature = "sessions")]
        Commands::Sessions { .. } => "sessions",
        #[cfg(feature = "sessions")]
        Commands::Turns { .. } => "turns",
        Commands::Migrate => "migrate",
        Commands::Status => "status",
        Commands::Describe { .. } => "describe",
        Commands::Reflector { .. } => "reflector",
        Commands::Fetch { .. } => "fetch",
        Commands::Cache { .. } => "cache",
        Commands::Backfill { .. } => "backfill",
        Commands::Extract { .. } => "extract",
        Commands::Invalidate { .. } => "invalidate",
        Commands::Supersede { .. } => "supersede",
        Commands::ExtractConcepts { .. } => "extract-concepts",
        Commands::InvalidationChain { .. } => "invalidation-chain",
        Commands::Audit { .. } => "audit",
        Commands::Move { .. } => "move",
        Commands::Search { .. } => "search",
        Commands::Health { .. } => "health",
        Commands::Export { .. } => "export",
    }
}

struct CmdGuard {
    start: Instant,
    cmd: &'static str,
    ns: String,
}

impl Drop for CmdGuard {
    fn drop(&mut self) {
        let latency_ms = self.start.elapsed().as_millis() as u64;
        claude::log_cmd(self.cmd, &self.ns, latency_ms);
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // When output is piped to a reader that closes early (`c0 walk | grep -q`,
    // `| head`), the next `println!` fails with EPIPE and the default hook
    // panics with a noisy "failed printing to stdout: Broken pipe" backtrace.
    // Rust ignores SIGPIPE at startup, and `unsafe_code = "forbid"` rules out
    // restoring SIG_DFL, so swallow that one panic and exit cleanly instead.
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let msg = info
            .payload()
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| info.payload().downcast_ref::<&str>().copied())
            .unwrap_or("");
        if msg.contains("Broken pipe") {
            std::process::exit(0);
        }
        default_hook(info);
    }));

    tracing_subscriber::registry()
        .with(fmt::layer().with_target(false))
        .with(
            EnvFilter::from_default_env()
                .add_directive("c0=info".parse().expect("valid directive")),
        )
        .init();

    let cli = Cli::parse();
    let ctx = config::detect_namespace();

    let _cmd_guard = CmdGuard {
        start: Instant::now(),
        cmd: cmd_kind(&cli.command),
        ns: ctx.namespace.clone(),
    };

    match cli.command {
        Commands::Init {
            namespace,
            parent,
            r#type,
        } => {
            let path = config::init_namespace(&namespace, &r#type, parent.as_deref())?;
            println!(
                "Initialized c0 namespace '{namespace}' at {}",
                path.display()
            );
            if let Some(ref p) = parent {
                println!("Parent namespace: {p}");
            }
            println!("\nCreated:");
            println!("  .c0/config.toml");
            println!("  .c0/patches/");
            println!("  .c0/triggers.txt");
            if r#type == "solution" {
                println!("  .c0/patches/client-overview.md");
                println!("  CLAUDE.md");
                println!("\nSolution project initialized with:");
                println!("  - SA process triggers (discovery, kickoff, tech stack, etc.)");
                println!("  - Client overview template");
                println!("  - CLAUDE.md with session instructions");
            }
            return Ok(());
        }
        #[cfg(feature = "sessions")]
        Commands::Sessions { action } => match action {
            None => {
                sessions::list_sessions(&ctx.namespaces, 20).await?;
                return Ok(());
            }
            Some(SessionCommands::Index) => {
                sessions::index_sessions().await?;
                return Ok(());
            }
            Some(SessionCommands::Search { query, limit, all }) => {
                let ns = if all { vec![] } else { ctx.namespaces.clone() };
                sessions::search_sessions(&query, &ns, limit).await?;
                return Ok(());
            }
            Some(SessionCommands::Resume { query, all }) => {
                let ns = if all { vec![] } else { ctx.namespaces.clone() };
                sessions::resume_session(&query, &ns).await?;
                return Ok(());
            }
            Some(SessionCommands::Extract {
                session,
                force,
                skip_sidechains,
            }) => {
                if let Some(sid) = session {
                    sessions::extract_session(&sid, force, skip_sidechains).await?;
                } else {
                    sessions::extract_all(force, skip_sidechains).await?;
                }
                return Ok(());
            }
            Some(SessionCommands::Cost { limit, all }) => {
                let ns = if all { vec![] } else { ctx.namespaces.clone() };
                sessions::list_session_costs(&ns, limit).await?;
                return Ok(());
            }
            Some(SessionCommands::Enrich {
                session,
                limit,
                force,
                all,
            }) => {
                let ns = if all { vec![] } else { ctx.namespaces.clone() };
                if let Some(sid) = session {
                    let conn = graph::connect().await?;
                    let mut r = conn
                        .execute(
                            neo4rs::query(
                                "MATCH (s:Session {session_id: $id}) RETURN s.namespace AS ns",
                            )
                            .param("id", sid.as_str()),
                        )
                        .await?;
                    let session_ns: String = if let Some(row) = r.next().await? {
                        row.get("ns").unwrap_or_else(|_| ctx.namespace.clone())
                    } else {
                        eprintln!(
                            "Session {sid} not found in graph. Run 'c0 sessions extract --session {sid}' first."
                        );
                        return Ok(());
                    };
                    sessions::enrich_session(&sid, &session_ns, force).await?;
                } else {
                    sessions::enrich_all(&ns, limit, force).await?;
                }
                return Ok(());
            }
        },
        #[cfg(feature = "sessions")]
        Commands::Turns { action } => match action {
            TurnsCommands::Search {
                query,
                limit,
                reflections,
                all,
            } => {
                let ns = if all { vec![] } else { ctx.namespaces.clone() };
                sessions::search_turns(&query, &ns, limit, reflections).await?;
                return Ok(());
            }
        },
        Commands::Status => {
            println!("c0 Namespace Status");
            println!("═══════════════════════════════════════");
            println!("Active namespace: {}", ctx.namespace);
            if !ctx.parent_dirs.is_empty() {
                let parent_names: Vec<String> = ctx
                    .namespaces
                    .iter()
                    .skip(1)
                    .filter(|n| *n != "global")
                    .cloned()
                    .collect();
                if !parent_names.is_empty() {
                    println!("Parent chain: {parent_names:?}");
                }
            }
            if let Some(ref pt) = ctx.project_type {
                println!("Project type: {pt}");
            }
            if let Some(ref dir) = ctx.project_dir {
                println!("Project dir: {}", dir.display());
            } else {
                println!("Project dir: (global only)");
            }
            println!("Searching namespaces: {:?}", ctx.namespaces);
            let patch_dirs = config::get_all_patches_dirs(&ctx);
            if !patch_dirs.is_empty() {
                println!("Patch directories:");
                for dir in &patch_dirs {
                    let count = std::fs::read_dir(dir)
                        .map(|entries| entries.filter_map(|e| e.ok()).count())
                        .unwrap_or(0);
                    println!("  {} ({count} files)", dir.display());
                }
            }
            return Ok(());
        }
        Commands::Reflector { action } => {
            match action {
                ReflectorCommands::Status => reflector::status()?,
                ReflectorCommands::Inbox => reflector::inbox()?,
                ReflectorCommands::Proposed => reflector::proposed()?,
                ReflectorCommands::Review => reflector::review()?,
                ReflectorCommands::Apply => reflector::apply()?,
                ReflectorCommands::Clear => reflector::clear()?,
                ReflectorCommands::Process => reflector::process().await?,
                ReflectorCommands::Notify => reflector::notify().await?,
                ReflectorCommands::Run { interval, no_apply } => {
                    reflector::run(&interval, !no_apply).await?
                }
            }
            return Ok(());
        }
        Commands::Config { action } => {
            match action {
                ConfigCommands::TestOllama => {
                    let sem_config = config::SemanticConfig::load();
                    println!("Testing Ollama connection...");
                    println!("  Host: {}", sem_config.ollama_host);
                    println!("  Model: {}", sem_config.ollama_model);
                    println!("  Timeout: {}ms", sem_config.ollama_timeout_ms);

                    match embeddings::OllamaClient::from_config(&sem_config) {
                        Some(client) => match client.test_connection().await {
                            Ok(()) => {
                                println!("\n✓ Connected successfully!");
                                println!("  Embedding dimension: 768 (nomic-embed-text)");
                            }
                            Err(e) => {
                                println!("\n✗ Connection failed: {e}");
                            }
                        },
                        None => {
                            println!("\n✗ Semantic triggers disabled in config");
                        }
                    }
                }
                ConfigCommands::Show => {
                    let sem_config = config::SemanticConfig::load();
                    println!("c0 Semantic Configuration");
                    println!("═══════════════════════════════════════");
                    println!("Enabled: {}", sem_config.enabled);
                    println!("Ollama host: {}", sem_config.ollama_host);
                    println!("Ollama model: {}", sem_config.ollama_model);
                    println!("Timeout: {}ms", sem_config.ollama_timeout_ms);
                    println!("Default threshold: {:.2}", sem_config.default_threshold);
                    println!("Fallback to regex: {}", sem_config.fallback_to_regex);
                }
            }
            return Ok(());
        }
        Commands::Health { json, fix } => {
            #[derive(serde::Serialize)]
            struct HealthReport {
                neo4j_ok: bool,
                ollama_ok: bool,
                fulltext_index_ok: bool,
                missing_embeddings: Vec<(String, String)>,
                broken_patch_refs: Vec<(String, String, String)>,
                empty_patches: Vec<(String, String)>,
                orphaned_concepts: Vec<(String, String)>,
                namespace_summary: Vec<(String, i64)>,
            }

            let neo4j_ok = tokio::time::timeout(Duration::from_secs(2), async {
                match graph::connect().await {
                    Ok(conn) => graph::ping(&conn).await.is_ok(),
                    Err(_) => false,
                }
            })
            .await
            .unwrap_or(false);

            let sem_config = config::SemanticConfig::load();
            let ollama_ok = if let Some(client) = embeddings::OllamaClient::from_config(&sem_config)
            {
                client.test_connection().await.is_ok()
            } else {
                false
            };

            let mut missing_embeddings: Vec<(String, String)> = Vec::new();
            let mut broken_patch_refs: Vec<(String, String, String)> = Vec::new();
            let mut empty_patches: Vec<(String, String)> = Vec::new();
            let mut orphaned_concepts: Vec<(String, String)> = Vec::new();
            let mut namespace_summary: Vec<(String, i64)> = Vec::new();
            let mut fulltext_index_ok = false;

            if neo4j_ok {
                if let Ok(conn) = graph::connect().await {
                    fulltext_index_ok = graph::check_fulltext_index_exists(&conn)
                        .await
                        .unwrap_or(false);
                    missing_embeddings =
                        graph::get_concepts_without_embeddings(&conn, &ctx.namespaces)
                            .await
                            .unwrap_or_default();

                    let patches_with_files = graph::find_patches_with_files(&conn, &ctx.namespaces)
                        .await
                        .unwrap_or_default();
                    for (name, namespace, file_path) in &patches_with_files {
                        let expanded = shellexpand::tilde(file_path);
                        if !std::path::Path::new(expanded.as_ref()).exists() {
                            broken_patch_refs.push((
                                name.clone(),
                                namespace.clone(),
                                file_path.clone(),
                            ));
                        }
                    }

                    // Empty shells: no inline content AND no file ref at all —
                    // these render blank on every walk. Scanned across all
                    // namespaces (not just the active chain).
                    if let Ok(all_patches) = graph::find_all_patches(&conn).await {
                        for (name, namespace, file, has_content) in &all_patches {
                            if !has_content && file.as_deref().unwrap_or("").is_empty() {
                                empty_patches.push((name.clone(), namespace.clone()));
                            }
                        }
                    }

                    orphaned_concepts = graph::find_orphaned_concepts(&conn, &ctx.namespaces)
                        .await
                        .unwrap_or_default();

                    namespace_summary = graph::count_concepts_by_namespace(&conn)
                        .await
                        .unwrap_or_default();

                    if fix {
                        if !missing_embeddings.is_empty() {
                            if let Some(client) = embeddings::OllamaClient::from_config(&sem_config)
                            {
                                let mut backfilled = 0;
                                for (name, namespace) in &missing_embeddings {
                                    if let Ok(embedding) = client.embed(name).await {
                                        if graph::update_concept_embedding(
                                            &conn, name, namespace, &embedding,
                                        )
                                        .await
                                        .is_ok()
                                        {
                                            backfilled += 1;
                                        }
                                    }
                                }
                                if !json {
                                    println!("Fixed: backfilled {backfilled} embeddings");
                                }
                                missing_embeddings =
                                    graph::get_concepts_without_embeddings(&conn, &ctx.namespaces)
                                        .await
                                        .unwrap_or_default();
                            }
                        }

                        // Do NOT null broken patch_file refs: a relative/tilde
                        // path or a file on another host is recoverable, and
                        // clearing it turns a fixable link into a permanent
                        // empty shell. Point the user at the inlining backfill,
                        // which resolves the file and stores content in-graph.
                        if !broken_patch_refs.is_empty() && !json {
                            println!(
                                "Note: {} patch(es) have unresolved file refs. Run `c0 backfill patch-content` to inline recoverable content (non-destructive).",
                                broken_patch_refs.len()
                            );
                        }
                    }
                }
            }

            let report = HealthReport {
                neo4j_ok,
                ollama_ok,
                fulltext_index_ok,
                missing_embeddings,
                broken_patch_refs,
                empty_patches,
                orphaned_concepts,
                namespace_summary,
            };

            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&report).unwrap_or_default()
                );
            } else {
                println!("c0 Health Check");
                println!("═══════════════════════════════════════");

                let neo4j_icon = if report.neo4j_ok { "✓" } else { "✗" };
                println!(
                    "{neo4j_icon} Neo4j: {}",
                    if report.neo4j_ok {
                        "connected"
                    } else {
                        "unreachable"
                    }
                );

                let ollama_icon = if report.ollama_ok { "✓" } else { "✗" };
                println!(
                    "{ollama_icon} Ollama: {}",
                    if report.ollama_ok {
                        "connected"
                    } else {
                        "unreachable"
                    }
                );

                let ft_icon = if report.fulltext_index_ok {
                    "✓"
                } else {
                    "✗"
                };
                println!(
                    "{ft_icon} Fulltext index: {}",
                    if report.fulltext_index_ok {
                        "present"
                    } else {
                        "missing (run c0 migrate)"
                    }
                );

                if !report.neo4j_ok {
                    println!("\nCannot run graph checks — Neo4j is not available.");
                } else {
                    println!();
                    if report.missing_embeddings.is_empty() {
                        println!("✓ Embeddings: all concepts have embeddings");
                    } else {
                        println!("⚠ Missing embeddings: {}", report.missing_embeddings.len());
                        for (name, ns) in &report.missing_embeddings {
                            println!("    {name} [{ns}]");
                        }
                        if !fix {
                            println!("  Run with --fix to backfill");
                        }
                    }

                    if report.broken_patch_refs.is_empty() {
                        println!("✓ Patch files: all references valid");
                    } else {
                        println!(
                            "⚠ Broken patch file refs: {}",
                            report.broken_patch_refs.len()
                        );
                        for (name, ns, path) in &report.broken_patch_refs {
                            println!("    {name} [{ns}] -> {path}");
                        }
                        println!(
                            "  Run `c0 backfill patch-content` to inline recoverable content (non-destructive)"
                        );
                    }

                    if report.empty_patches.is_empty() {
                        println!("✓ Empty patches: none");
                    } else {
                        println!(
                            "⚠ Empty patches (no content, no file ref): {}",
                            report.empty_patches.len()
                        );
                        for (name, ns) in report.empty_patches.iter().take(15) {
                            println!("    {name} [{ns}]");
                        }
                        if report.empty_patches.len() > 15 {
                            println!("    … and {} more", report.empty_patches.len() - 15);
                        }
                        println!(
                            "  These render blank on walk. Recover via `c0 backfill patch-content`, or remove."
                        );
                    }

                    if report.orphaned_concepts.is_empty() {
                        println!("✓ Orphaned concepts: none");
                    } else {
                        println!(
                            "⚠ Orphaned concepts (no relationships): {}",
                            report.orphaned_concepts.len()
                        );
                        for (name, ns) in report.orphaned_concepts.iter().take(10) {
                            println!("    {name} [{ns}]");
                        }
                        if report.orphaned_concepts.len() > 10 {
                            println!("    ... and {} more", report.orphaned_concepts.len() - 10);
                        }
                    }

                    println!();
                    println!("Namespace summary:");
                    for (ns, count) in &report.namespace_summary {
                        println!("  {ns}: {count} concepts");
                    }
                }
            }
            return Ok(());
        }
        _ => {}
    }

    let graph_conn = graph::connect().await?;

    match cli.command {
        Commands::Add { what } => match what {
            AddCommands::Concept {
                name,
                description,
                source,
                url,
                force,
                to,
                valid_at,
            } => {
                let target_namespace = to.as_ref().unwrap_or(&ctx.namespace);

                if let Some(ref target) = to
                    && !ctx.namespaces.contains(target)
                {
                    anyhow::bail!(
                        "'{}' is not in the namespace chain {:?}",
                        target,
                        ctx.namespaces
                    );
                }

                let embed_text = match &description {
                    Some(desc) => format!("{name}: {desc}"),
                    None => name.clone(),
                };

                let semantic_config = config::SemanticConfig::load();
                let embedding =
                    if let Some(client) = embeddings::OllamaClient::from_config(&semantic_config) {
                        match client.embed(&embed_text).await {
                            Ok(emb) => {
                                println!("Generated embedding for '{name}'");
                                Some(emb)
                            }
                            Err(e) => {
                                eprintln!("Warning: Could not generate embedding: {e}");
                                None
                            }
                        }
                    } else {
                        None
                    };

                graph::ensure_concept_unique_constraint(&graph_conn).await?;
                graph::ensure_concept_embedding_index(&graph_conn).await?;

                if !force && let Some(ref emb) = embedding {
                    let similar = graph::find_similar_concepts(
                        &graph_conn,
                        emb,
                        semantic_config.default_threshold,
                        &ctx.namespaces,
                    )
                    .await
                    .unwrap_or_default();

                    if !similar.is_empty() {
                        println!("⚠️  Similar concepts already exist:");
                        for (similar_name, similarity) in &similar {
                            println!("   - {} ({:.0}% similar)", similar_name, similarity * 100.0);
                        }
                        println!();
                        println!("Consider:");
                        if let Some((best_name, _)) = similar.first() {
                            println!("  c0 relate {name} RELATED_TO {best_name}");
                        }
                        println!("  c0 add concept {name} --force  # to create anyway");
                        return Ok(());
                    }
                }

                let valid_at_dt = valid_at
                    .as_ref()
                    .map(|s| parse_date_to_datetime(s))
                    .transpose()?;

                graph::add_concept(
                    &graph_conn,
                    &name,
                    target_namespace,
                    description.as_deref(),
                    source.as_deref(),
                    url.as_deref(),
                    embedding.as_deref(),
                    valid_at_dt,
                )
                .await?;
                println!("Created concept: {name} [{target_namespace}]");
                if let Some(d) = &description {
                    println!("  description: {d}");
                }
                if let Some(s) = &source {
                    println!("  source: {s}");
                }
                if let Some(u) = &url {
                    println!("  url: {u}");
                }
                if let Some(v) = &valid_at {
                    println!("  valid_at: {v}");
                }
            }
            AddCommands::Patch {
                name,
                corrects,
                file,
                content,
                source,
                url,
                to,
                valid_at,
            } => {
                if file.is_none() && content.is_none() {
                    anyhow::bail!(
                        "a patch needs content. Pass --file <path> or --content <text>.\n       (a patch with neither renders empty on walk)"
                    );
                }
                if let Some(f) = &file {
                    let expanded = shellexpand::tilde(f).to_string();
                    if !std::path::Path::new(&expanded).exists() {
                        anyhow::bail!("--file not found: {f}");
                    }
                }

                let target_namespace = to.as_ref().unwrap_or(&ctx.namespace);

                if let Some(ref target) = to
                    && !ctx.namespaces.contains(target)
                {
                    anyhow::bail!(
                        "'{}' is not in the namespace chain {:?}",
                        target,
                        ctx.namespaces
                    );
                }

                let valid_at_dt = valid_at
                    .as_ref()
                    .map(|s| parse_date_to_datetime(s))
                    .transpose()?;

                graph::add_patch(
                    &graph_conn,
                    &name,
                    corrects.as_deref(),
                    file.as_deref(),
                    content.as_deref(),
                    target_namespace,
                    source.as_deref(),
                    url.as_deref(),
                    valid_at_dt,
                )
                .await?;
                println!("Created patch: {name} [{target_namespace}]");
                if let Some(c) = &corrects {
                    println!("  corrects: {c}");
                }
                if let Some(f) = &file {
                    println!("  file: {f}");
                }
                if let Some(s) = &source {
                    println!("  source: {s}");
                }
                if let Some(u) = &url {
                    println!("  url: {u}");
                }
                if let Some(v) = &valid_at {
                    println!("  valid_at: {v}");
                }

                // A patch with no --corrects has no HAS_PATCH/CORRECTS edge, so
                // `c0 walk` (which traverses those edges only) can never reach
                // it — it would surface in `c0 search` alone. Anchor it to the
                // active namespace's concept so it's reachable from
                // `c0 walk <namespace>`.
                if corrects.is_none() {
                    if graph::concept_exists(&graph_conn, target_namespace, &ctx.namespaces).await?
                    {
                        graph::link_patch(&graph_conn, &name, target_namespace, &ctx.namespaces)
                            .await?;
                        println!("  linked: {target_namespace} -[HAS_PATCH]-> {name}");
                    } else {
                        println!(
                            "  note: no '{target_namespace}' concept exists yet, so this patch \
                             is reachable via `c0 search` but not `c0 walk`. Create the concept \
                             (`c0 add concept {target_namespace}`) then `c0 relate \
                             {target_namespace} HAS_PATCH {name}`, or re-add with --corrects."
                        );
                    }
                }
            }
        },
        Commands::Relate { from, rel_type, to } => {
            graph::relate(&graph_conn, &from, &rel_type, &to, &ctx.namespaces).await?;
            println!("{from} -[{rel_type}]-> {to}");
        }
        Commands::Walk {
            start,
            depth,
            context,
            live,
            as_of,
            include_expired,
        } => {
            let timer = Instant::now();

            let temporal = {
                let mut t = graph::TemporalQuery::default();
                if let Some(ref date_str) = as_of {
                    t.as_of = Some(parse_date_to_datetime(date_str)?);
                }
                if include_expired {
                    t.include_expired = true;
                }
                t
            };

            if as_of.is_some() || include_expired {
                let mode = if include_expired {
                    "all (including expired)".to_string()
                } else if let Some(ref date) = as_of {
                    format!("as-of {date}")
                } else {
                    "default".to_string()
                };
                println!("⏰ Temporal query mode: {mode}");
            }

            let mut concept = start.clone();
            let mut patches =
                graph::get_patches_temporal(&graph_conn, &concept, &ctx.namespaces, &temporal)
                    .await?;
            let mut connected =
                graph::traverse_temporal(&graph_conn, &concept, depth, &ctx.namespaces, &temporal)
                    .await?;

            if patches.is_empty() && connected.is_empty() {
                let ft_matches =
                    graph::search_concepts_fulltext(&graph_conn, &start, 5, &ctx.namespaces)
                        .await?;
                let query_terms: Vec<String> =
                    start.split_whitespace().map(|t| t.to_lowercase()).collect();
                let best = ft_matches.iter().find(|r| {
                    let name_lower = r.name.to_lowercase();
                    let matching = query_terms
                        .iter()
                        .filter(|t| name_lower.contains(t.as_str()))
                        .count();
                    matching > query_terms.len() / 2
                });
                if let Some(best) = best {
                    println!(
                        "(fulltext: '{start}' -> '{}' [score: {:.2}])",
                        best.name, best.similarity
                    );
                    concept = best.name.clone();
                    patches = graph::get_patches_temporal(
                        &graph_conn,
                        &concept,
                        &ctx.namespaces,
                        &temporal,
                    )
                    .await?;
                    connected = graph::traverse_temporal(
                        &graph_conn,
                        &concept,
                        depth,
                        &ctx.namespaces,
                        &temporal,
                    )
                    .await?;
                }
            }

            if patches.is_empty() && connected.is_empty() {
                let use_semantic = start.len() > 10 || start.contains(' ');
                if use_semantic {
                    let semantic_config = config::SemanticConfig::load();
                    if semantic_config.enabled
                        && let Some(client) =
                            embeddings::OllamaClient::from_config(&semantic_config)
                        && let Ok(query_embedding) = client.embed(&start).await
                    {
                        let hybrid_config = graph::HybridSearchConfig::default();
                        let similar = graph::search_hybrid_temporal(
                            &graph_conn,
                            &start,
                            &query_embedding,
                            &ctx.namespaces,
                            &temporal,
                            &hybrid_config,
                        )
                        .await
                        .unwrap_or_default();

                        if let Some((best_name, score)) = similar.first() {
                            println!(
                                "(hybrid: '{}' -> '{}' [rrf: {:.6}])",
                                start, best_name, score
                            );
                            concept = best_name.clone();
                            patches = graph::get_patches_temporal(
                                &graph_conn,
                                &concept,
                                &ctx.namespaces,
                                &temporal,
                            )
                            .await?;
                            connected = graph::traverse_temporal(
                                &graph_conn,
                                &concept,
                                depth,
                                &ctx.namespaces,
                                &temporal,
                            )
                            .await?;
                        }
                    }
                }
            }

            if !patches.is_empty() {
                println!("KNOWLEDGE PATCH:");
                println!("---");
                for patch in patches {
                    if ctx.namespace != "global" && patch.namespace == "global" {
                        println!("[global]");
                    } else if patch.namespace != "global" {
                        println!("[{}]", patch.namespace);
                    }
                    if let Some(ref url) = patch.url {
                        println!("📎 {url}");
                    }
                    print_patch_body(&patch);
                }
                println!("---");
            }

            if connected.is_empty() {
                log_dead_end("walk", &start, &ctx.namespace, context.as_deref());
                println!("No connections from '{concept}'");
            } else {
                let semantic_config = config::SemanticConfig::load();
                let mut ranked_connected: Vec<(String, f32)> = Vec::new();

                if semantic_config.enabled
                    && let Some(client) = embeddings::OllamaClient::from_config(&semantic_config)
                    && let Ok(query_emb) = client.embed(&start).await
                {
                    for name in &connected {
                        if let Ok(Some(concept_emb)) =
                            graph::get_concept_embedding(&graph_conn, name, &ctx.namespaces).await
                        {
                            let sim = embeddings::cosine_similarity(&query_emb, &concept_emb);
                            ranked_connected.push((name.clone(), sim));
                        } else {
                            ranked_connected.push((name.clone(), 0.0));
                        }
                    }
                    ranked_connected
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                }

                if ranked_connected.is_empty() {
                    ranked_connected = connected.iter().map(|n| (n.clone(), 0.0)).collect();
                }

                println!("Walking from '{concept}' (depth {depth}):");
                for (name, score) in &ranked_connected {
                    if *score > 0.0 {
                        println!("  -> {} ({:.0}%)", name, score * 100.0);
                    } else {
                        println!("  -> {name}");
                    }
                }

                const PATCH_DISPLAY_THRESHOLD: f32 = 0.30;

                for (name, score) in &ranked_connected {
                    if *score < PATCH_DISPLAY_THRESHOLD && *score > 0.0 {
                        continue;
                    }
                    let node_patches =
                        graph::get_patches_temporal(&graph_conn, name, &ctx.namespaces, &temporal)
                            .await?;
                    for patch in node_patches {
                        if *score > 0.0 {
                            println!("\n[{} PATCH ({:.0}%)]", name.to_uppercase(), score * 100.0);
                        } else {
                            println!("\n[{} PATCH]", name.to_uppercase());
                        }
                        println!("---");
                        if ctx.namespace != "global" && patch.namespace == "global" {
                            println!("[global]");
                        } else if patch.namespace != "global" {
                            println!("[{}]", patch.namespace);
                        }
                        if let Some(ref url) = patch.url {
                            println!("📎 {url}");
                        }
                        print_patch_body(&patch);
                        println!("---");
                    }
                }
            }

            let linked_sessions =
                graph::get_sessions_for_concept(&graph_conn, &concept, &ctx.namespaces, 5)
                    .await
                    .unwrap_or_default();
            if !linked_sessions.is_empty() {
                println!("\n📝 SESSIONS DISCUSSING '{concept}':");
                for (session, count) in &linked_sessions {
                    let date = if session.created_at.len() >= 10 {
                        &session.created_at[..10]
                    } else {
                        &session.created_at
                    };
                    let display = session
                        .summary
                        .as_deref()
                        .filter(|s| !s.is_empty())
                        .unwrap_or(&session.first_prompt);
                    let snippet = if display.len() > 90 {
                        format!("{}...", &display[..90])
                    } else {
                        display.to_string()
                    };
                    let mention = if *count > 1 {
                        format!(" ×{count}")
                    } else {
                        String::new()
                    };
                    println!(
                        "  [{}] {}{}  {}",
                        date,
                        &session.session_id[..8.min(session.session_id.len())],
                        mention,
                        snippet
                    );
                }
                println!("  (resume any: claude --resume <session-id>)");
            }

            if live {
                let live_sources =
                    graph::get_live_sources_for_concept(&graph_conn, &concept, &ctx.namespaces)
                        .await?;
                if !live_sources.is_empty() {
                    println!("\n📡 LIVE SOURCES:");
                    for source in &live_sources {
                        println!("  Fetching {} ({})...", source.name, source.source_type);
                        match fetch::fetch_url(&source.url, &source.source_type).await {
                            Ok(result) => {
                                println!("---");
                                println!("[LIVE: {} ({})]", source.name, source.source_type);
                                println!("{}", result.content);
                                println!("---");
                            }
                            Err(e) => {
                                println!("  ✗ Failed to fetch {}: {}", source.name, e);
                            }
                        }
                    }
                }
            }
            eprintln!("[c0: {}ms]", timer.elapsed().as_millis());
        }
        Commands::Find { pattern } => {
            let results = graph::find_pattern(&graph_conn, &pattern).await?;
            if results.is_empty() {
                log_dead_end("find", &pattern, &ctx.namespace, None);
                println!("No matches for pattern");
            } else {
                println!("Pattern results:");
                for row in results {
                    println!("  {row}");
                }
            }
        }
        Commands::Link { what } => match what {
            LinkCommands::Patch { patch, concept } => {
                graph::link_patch(&graph_conn, &patch, &concept, &ctx.namespaces).await?;
                println!("{concept} -[HAS_PATCH]-> {patch}");
            }
            LinkCommands::Source { action } => match action {
                SourceCommands::Add {
                    name,
                    url,
                    r#type,
                    concept,
                } => {
                    let source_type =
                        r#type.unwrap_or_else(|| fetch::detect_source_type(&url).to_string());

                    println!("Fetching content from URL to generate embedding...");
                    let content = match fetch::fetch_url(&url, &source_type).await {
                        Ok(result) => {
                            println!("  ✓ Fetched {} bytes", result.content.len());
                            fetch::truncate_for_embedding(&result.content, 4000)
                        }
                        Err(e) => {
                            eprintln!("  ⚠ Could not fetch URL: {e}");
                            eprintln!("  Using name and URL for embedding instead.");
                            format!("{name}: {url}")
                        }
                    };

                    let semantic_config = config::SemanticConfig::load();
                    let embedding = if let Some(client) =
                        embeddings::OllamaClient::from_config(&semantic_config)
                    {
                        match client.embed(&content).await {
                            Ok(emb) => {
                                println!("  ✓ Generated embedding ({} dims)", emb.len());
                                Some(emb)
                            }
                            Err(e) => {
                                eprintln!("  ⚠ Could not generate embedding: {e}");
                                None
                            }
                        }
                    } else {
                        eprintln!("  ⚠ Semantic search not configured");
                        None
                    };

                    graph::ensure_live_source_index(&graph_conn).await?;
                    graph::add_live_source(
                        &graph_conn,
                        &name,
                        &url,
                        &source_type,
                        &ctx.namespace,
                        concept.as_deref(),
                        embedding.as_deref(),
                    )
                    .await?;

                    println!("\nCreated live source: {} [{}]", name, ctx.namespace);
                    println!("  url: {url}");
                    println!("  type: {source_type}");
                    if let Some(c) = &concept {
                        println!("  linked to: {c}");
                    }
                }
                SourceCommands::Remove { name } => {
                    let deleted =
                        graph::remove_live_source(&graph_conn, &name, &ctx.namespace).await?;
                    if deleted {
                        println!("Removed live source: {name}");
                    } else {
                        println!("Live source not found: {name}");
                    }
                }
                SourceCommands::Refresh { name, all } => {
                    let sources = if all {
                        graph::list_live_sources(&graph_conn, &ctx.namespaces).await?
                    } else if let Some(ref n) = name {
                        if let Some(s) =
                            graph::get_live_source(&graph_conn, n, &ctx.namespaces).await?
                        {
                            vec![s]
                        } else {
                            println!("Live source not found: {n}");
                            return Ok(());
                        }
                    } else {
                        println!("Specify a source name or use --all");
                        return Ok(());
                    };

                    if sources.is_empty() {
                        println!("No live sources to refresh");
                        return Ok(());
                    }

                    let semantic_config = config::SemanticConfig::load();
                    let client =
                        if let Some(c) = embeddings::OllamaClient::from_config(&semantic_config) {
                            c
                        } else {
                            eprintln!("Error: Semantic search not configured");
                            return Ok(());
                        };

                    println!("Refreshing {} source(s)...", sources.len());
                    for source in sources {
                        print!("  {} ... ", source.name);
                        match fetch::fetch_url(&source.url, &source.source_type).await {
                            Ok(result) => {
                                let content = fetch::truncate_for_embedding(&result.content, 4000);
                                match client.embed(&content).await {
                                    Ok(embedding) => {
                                        graph::update_live_source_embedding(
                                            &graph_conn,
                                            &source.name,
                                            &source.namespace,
                                            &embedding,
                                        )
                                        .await?;
                                        println!("✓ refreshed");
                                    }
                                    Err(e) => println!("✗ embedding error: {e}"),
                                }
                            }
                            Err(e) => println!("✗ fetch error: {e}"),
                        }
                    }
                }
                SourceCommands::Fetch { name } => {
                    let source = if let Some(s) =
                        graph::get_live_source(&graph_conn, &name, &ctx.namespaces).await?
                    {
                        s
                    } else {
                        println!("Live source not found: {name}");
                        return Ok(());
                    };

                    println!("Fetching: {} ({})", source.url, source.source_type);
                    match fetch::fetch_url(&source.url, &source.source_type).await {
                        Ok(result) => {
                            println!("---");
                            println!("{}", result.content);
                            println!("---");
                        }
                        Err(e) => {
                            eprintln!("Error fetching source: {e}");
                        }
                    }
                }
                SourceCommands::Search {
                    query,
                    limit,
                    fetch: do_fetch,
                } => {
                    let semantic_config = config::SemanticConfig::load();
                    let client =
                        if let Some(c) = embeddings::OllamaClient::from_config(&semantic_config) {
                            c
                        } else {
                            eprintln!("Error: Semantic search not configured");
                            return Ok(());
                        };

                    println!("Searching for: \"{query}\"");
                    let query_embedding = client.embed(&query).await?;

                    let results = graph::find_similar_live_sources(
                        &graph_conn,
                        &query_embedding,
                        semantic_config.query_floor_threshold,
                        &ctx.namespaces,
                        limit,
                    )
                    .await?;

                    if results.is_empty() {
                        println!("No matching sources found");
                        return Ok(());
                    }

                    println!("\nFound {} matching source(s):", results.len());
                    for (source, similarity) in &results {
                        let linked = source
                            .linked_concept
                            .as_ref()
                            .map(|c| format!(" -> {c}"))
                            .unwrap_or_default();
                        println!(
                            "  [{:.0}%] {} ({}){}",
                            similarity * 100.0,
                            source.name,
                            source.source_type,
                            linked
                        );
                        println!("        {}", source.url);
                    }

                    if do_fetch {
                        println!("\n📡 Fetching content from top result...");
                        let (top_source, _) = &results[0];
                        match fetch::fetch_url(&top_source.url, &top_source.source_type).await {
                            Ok(result) => {
                                println!("---");
                                println!("[{}]", top_source.name);
                                println!("{}", result.content);
                                println!("---");
                            }
                            Err(e) => {
                                eprintln!("Error fetching source: {e}");
                            }
                        }
                    }
                }
            },
        },
        Commands::List { what } => match what {
            ListCommands::Patches => {
                let patches = graph::list_patches(&graph_conn, &ctx.namespaces).await?;
                if patches.is_empty() {
                    println!("No patches found");
                } else {
                    println!(
                        "Knowledge patches ({} namespace(s): {:?}):",
                        ctx.namespaces.len(),
                        ctx.namespaces
                    );
                    for (name, file, corrects, namespace) in patches {
                        let loc =
                            file.map_or_else(|| "inline".to_string(), |f| format!("file: {f}"));
                        let corr = corrects
                            .map(|c| format!(" -> corrects: {c}"))
                            .unwrap_or_default();
                        let ns = if namespace == "global" {
                            String::new()
                        } else {
                            format!(" [{namespace}]")
                        };
                        println!("  {name}{ns} ({loc}){corr}");
                    }
                }
            }
            ListCommands::Triggers { semantic } => {
                if semantic {
                    let triggers =
                        graph::list_semantic_triggers(&graph_conn, &ctx.namespaces).await?;
                    if triggers.is_empty() {
                        println!("No semantic triggers configured");
                    } else {
                        println!("Semantic triggers ({}):", triggers.len());
                        for t in triggers {
                            let threshold_str = t
                                .threshold
                                .map(|th| format!(" (threshold: {th:.2})"))
                                .unwrap_or_default();
                            let ns = if t.namespace == "global" {
                                String::new()
                            } else {
                                format!(" [{}]", t.namespace)
                            };
                            println!("  {}{}{}", t.name, ns, threshold_str);
                            if !t.description.is_empty() {
                                println!("    {}", t.description);
                            }
                        }
                    }
                } else {
                    let triggers = read_triggers(&ctx);
                    if triggers.is_empty() {
                        println!("No regex triggers configured");
                    } else {
                        println!("Regex triggers ({}):", triggers.len());
                        for t in triggers {
                            println!("  {t}");
                        }
                    }
                }
            }
            ListCommands::Sources => {
                let sources = graph::list_live_sources(&graph_conn, &ctx.namespaces).await?;
                if sources.is_empty() {
                    println!("No live sources configured");
                } else {
                    println!("Live sources ({}):", sources.len());
                    for s in sources {
                        let ns = if s.namespace == "global" {
                            String::new()
                        } else {
                            format!(" [{}]", s.namespace)
                        };
                        let linked = s
                            .linked_concept
                            .map(|c| format!(" -> {c}"))
                            .unwrap_or_default();
                        let indexed = s.last_indexed.map_or_else(
                            || " (not indexed)".to_string(),
                            |t| format!(" (indexed: {t})"),
                        );
                        println!(
                            "  {}{} ({}){}{}",
                            s.name, ns, s.source_type, linked, indexed
                        );
                        println!("    {}", s.url);
                    }
                }
            }
        },
        Commands::Trigger { action } => match action {
            TriggerCommands::Add {
                pattern,
                semantic,
                description,
                threshold,
                no_enrich,
            } => {
                if semantic {
                    let sem_config = config::SemanticConfig::load();
                    let client =
                        embeddings::OllamaClient::from_config(&sem_config).ok_or_else(|| {
                            anyhow::anyhow!("Semantic triggers not enabled in config")
                        })?;

                    let base_desc = description.unwrap_or_else(|| pattern.clone());

                    let desc = if no_enrich {
                        base_desc
                    } else {
                        println!("Enriching description with LLM...");
                        let related = graph::get_related_concept_names(
                            &graph_conn,
                            &pattern,
                            &ctx.namespaces,
                        )
                        .await?;
                        let enriched = enrich_trigger_description(
                            &sem_config,
                            &pattern,
                            &base_desc,
                            &ctx.namespace,
                            &related,
                        )
                        .await?;
                        println!("  Base: {base_desc}");
                        println!("  Enriched: {enriched}");
                        enriched
                    };

                    println!("Generating embedding for description...");
                    let embedding = client.embed(&desc).await?;

                    graph::ensure_semantic_trigger_index(&graph_conn).await?;
                    graph::add_semantic_trigger(
                        &graph_conn,
                        &pattern,
                        &desc,
                        &embedding,
                        &ctx.namespace,
                        threshold,
                    )
                    .await?;

                    println!("Added semantic trigger: {} [{}]", pattern, ctx.namespace);
                    println!("  Description: {desc}");
                    println!("  Embedding: {} dimensions", embedding.len());
                    if let Some(th) = threshold {
                        println!("  Custom threshold: {th:.2}");
                    }
                } else {
                    let mut triggers = read_triggers(&ctx);
                    if triggers.contains(&pattern) {
                        println!("Trigger already exists: {pattern}");
                    } else {
                        triggers.push(pattern.clone());
                        write_triggers(&triggers, &ctx);
                        println!("Added regex trigger: {} [{}]", pattern, ctx.namespace);
                    }
                }
            }
            TriggerCommands::Remove { pattern, semantic } => {
                if semantic {
                    let deleted =
                        graph::remove_semantic_trigger(&graph_conn, &pattern, &ctx.namespace)
                            .await?;
                    if deleted {
                        println!("Removed semantic trigger: {pattern}");
                    } else {
                        println!("Semantic trigger not found: {pattern}");
                    }
                } else {
                    let mut triggers = read_triggers(&ctx);
                    if let Some(pos) = triggers.iter().position(|t| t == &pattern) {
                        triggers.remove(pos);
                        write_triggers(&triggers, &ctx);
                        println!("Removed regex trigger: {pattern}");
                    } else {
                        println!("Trigger not found: {pattern}");
                    }
                }
            }
            TriggerCommands::Test { prompt } => {
                let prompt_lower = prompt.to_lowercase();

                println!("Testing prompt: \"{prompt}\"");
                println!("═══════════════════════════════════════");

                println!("\nRegex matches:");
                let regex_triggers = read_triggers(&ctx);
                let mut regex_matched = false;
                for pattern in &regex_triggers {
                    if pattern.is_empty() || pattern.starts_with('#') {
                        continue;
                    }
                    if let Ok(re) = regex::Regex::new(pattern) {
                        if re.is_match(&prompt_lower) {
                            println!("  ✓ {pattern}");
                            regex_matched = true;
                        }
                    } else if prompt_lower.contains(pattern) {
                        println!("  ✓ {pattern} (literal)");
                        regex_matched = true;
                    }
                }
                if !regex_matched {
                    println!("  (none)");
                }

                println!("\nSemantic matches:");
                let sem_config = config::SemanticConfig::load();
                if sem_config.enabled {
                    match embeddings::OllamaClient::from_config(&sem_config) {
                        Some(client) => match client.embed(&prompt_lower).await {
                            Ok(embedding) => {
                                let matches = graph::find_similar_triggers(
                                    &graph_conn,
                                    &embedding,
                                    sem_config.default_threshold,
                                    sem_config.query_floor_threshold,
                                    &ctx.namespaces,
                                )
                                .await?;

                                if matches.is_empty() {
                                    println!(
                                        "  (none above floor {:.2})",
                                        sem_config.query_floor_threshold
                                    );
                                } else {
                                    for m in matches {
                                        let sim = m.similarity.unwrap_or(0.0);
                                        println!("  ✓ \"{}\" (similarity: {:.2})", m.name, sim);
                                    }
                                }
                            }
                            Err(e) => {
                                println!("  (error getting embedding: {e})");
                            }
                        },
                        None => {
                            println!("  (ollama client not configured)");
                        }
                    }
                } else {
                    println!("  (semantic triggers disabled)");
                }
            }
            TriggerCommands::Match { prompt, json } => {
                let prompt_lower = prompt.to_lowercase();

                let regex_triggers = read_triggers(&ctx);
                for pattern in &regex_triggers {
                    if pattern.is_empty() || pattern.starts_with('#') {
                        continue;
                    }
                    let matched = if let Ok(re) = regex::Regex::new(pattern) {
                        re.is_match(&prompt_lower)
                    } else {
                        prompt_lower.contains(pattern)
                    };

                    if matched {
                        let topic = extract_topic(&prompt_lower, pattern);
                        if json {
                            println!(
                                r#"{{"type":"regex","topic":"{topic}","pattern":"{pattern}"}}"#
                            );
                        } else {
                            println!("{topic}");
                        }
                        return Ok(());
                    }
                }

                let sem_config = config::SemanticConfig::load();
                if sem_config.enabled
                    && let Some(client) = embeddings::OllamaClient::from_config(&sem_config)
                    && let Ok(embedding) = client.embed(&prompt_lower).await
                {
                    let matches = graph::find_similar_triggers(
                        &graph_conn,
                        &embedding,
                        sem_config.default_threshold,
                        sem_config.query_floor_threshold,
                        &ctx.namespaces,
                    )
                    .await?;

                    if let Some(m) = matches.first() {
                        if json {
                            println!(
                                r#"{{"type":"semantic","topic":"{}","similarity":{:.2}}}"#,
                                m.name,
                                m.similarity.unwrap_or(0.0)
                            );
                        } else {
                            println!("{}", m.name);
                        }
                        return Ok(());
                    }
                }

                if json {
                    println!(r#"{{"type":"none"}}"#);
                }
            }
        },
        Commands::Migrate => {
            let ns_count = graph::count_nodes_without_namespace(&graph_conn).await?;
            if ns_count == 0 {
                println!("✓ All nodes already have namespace property.");
            } else {
                println!("Found {ns_count} nodes without namespace property.");
                println!("Migrating to 'global' namespace...");
                graph::migrate_add_global_namespace(&graph_conn).await?;
                println!("✓ Namespace migration complete.");
            }

            let temporal_count = graph::count_nodes_without_temporal(&graph_conn).await?;
            if temporal_count == 0 {
                println!("✓ All concepts/patches already have temporal fields.");
            } else {
                println!("Found {temporal_count} concepts/patches without temporal fields.");
                println!(
                    "Adding bi-temporal fields (valid_at, invalid_at, expired_at, created_at)..."
                );
                graph::migrate_add_temporal_fields(&graph_conn).await?;
                println!("✓ Bi-temporal migration complete.");
            }

            print!("Creating fulltext indexes...");
            std::io::stdout().flush().ok();
            graph::ensure_concept_fulltext_index(&graph_conn).await?;
            graph::ensure_patch_fulltext_index(&graph_conn).await?;
            println!(" ✓");

            #[cfg(feature = "sessions")]
            {
                print!("Creating session indexes...");
                std::io::stdout().flush().ok();
                graph::ensure_session_indexes(&graph_conn).await?;
                println!(" ✓");

                print!("Creating turn/reflection/file/command indexes...");
                std::io::stdout().flush().ok();
                graph::ensure_turn_indexes(&graph_conn).await?;
                println!(" ✓");
            }

            println!("\nMigration finished.");
        }
        Commands::Describe {
            concept,
            description,
        } => {
            let semantic_config = config::SemanticConfig::load();
            let client = if let Some(c) = embeddings::OllamaClient::from_config(&semantic_config) {
                c
            } else {
                eprintln!("Error: Semantic search not configured. Check ~/.c0/config.toml");
                return Ok(());
            };

            let embed_text = format!("{concept}: {description}");
            let embedding = client.embed(&embed_text).await?;

            let updated = graph::update_concept_description(
                &graph_conn,
                &concept,
                &ctx.namespace,
                &description,
                &embedding,
            )
            .await?;

            if updated {
                println!("Updated concept: {concept}");
                println!("  description: {description}");
                println!("  embedding: regenerated from description");
            } else {
                eprintln!(
                    "Concept '{}' not found in namespace '{}'",
                    concept, ctx.namespace
                );
            }
        }
        Commands::Fetch {
            query,
            limit,
            all,
            fresh,
        } => {
            let semantic_config = config::SemanticConfig::load();
            let client = if let Some(c) = embeddings::OllamaClient::from_config(&semantic_config) {
                c
            } else {
                eprintln!("Error: Semantic search not configured");
                return Ok(());
            };

            println!("🔍 Searching live sources for: \"{query}\"");
            let query_embedding = client.embed(&query).await?;

            let results = graph::find_similar_live_sources(
                &graph_conn,
                &query_embedding,
                semantic_config.query_floor_threshold,
                &ctx.namespaces,
                limit,
            )
            .await?;

            if results.is_empty() {
                println!("No matching sources found.");
                println!("\nTip: Add sources with: c0 link source add <name> --url <url>");
                return Ok(());
            }

            println!("Found {} matching source(s)\n", results.len());

            let sources_to_fetch = if all { &results[..] } else { &results[..1] };

            for (source, similarity) in sources_to_fetch {
                println!(
                    "📡 [{:.0}%] {} ({})",
                    similarity * 100.0,
                    source.name,
                    source.source_type
                );
                println!("   {}", source.url);

                let fetch_result = if fresh {
                    fetch::fetch_url_no_cache(&source.url, &source.source_type).await
                } else {
                    fetch::fetch_url(&source.url, &source.source_type).await
                };
                match fetch_result {
                    Ok(result) => {
                        println!("---");
                        println!("{}", result.content);
                        println!("---\n");
                    }
                    Err(e) => {
                        eprintln!("   ✗ Error fetching: {e}\n");
                    }
                }
            }

            if !all && results.len() > 1 {
                println!("Other matches:");
                for (source, similarity) in &results[1..] {
                    println!(
                        "  [{:.0}%] {} - {}",
                        similarity * 100.0,
                        source.name,
                        source.url
                    );
                }
                println!("\nUse --all to fetch all matches");
            }
        }
        Commands::Cache { action } => match action {
            CacheCommands::Clear => {
                let count = fetch::clear_cache()?;
                println!("Cleared {count} cached entries.");
            }
        },
        Commands::Backfill { what } => match what {
            BackfillCommands::Embeddings { dry_run } => {
                let semantic_config = config::SemanticConfig::load();
                let client =
                    if let Some(c) = embeddings::OllamaClient::from_config(&semantic_config) {
                        c
                    } else {
                        eprintln!("Error: Semantic search not configured. Check ~/.c0/config.toml");
                        return Ok(());
                    };

                graph::ensure_concept_embedding_index(&graph_conn).await?;
                let concepts =
                    graph::get_concepts_without_embeddings(&graph_conn, &ctx.namespaces).await?;

                if concepts.is_empty() {
                    println!("All concepts already have embeddings.");
                    return Ok(());
                }

                println!("Found {} concepts without embeddings:", concepts.len());
                for (name, namespace) in &concepts {
                    println!("  - {name} [{namespace}]");
                }

                if dry_run {
                    println!("\n(dry run - no changes made)");
                    return Ok(());
                }

                println!("\nGenerating embeddings...");
                let mut success = 0;
                let mut failed = 0;
                for (name, namespace) in &concepts {
                    match client.embed(name).await {
                        Ok(embedding) => {
                            graph::update_concept_embedding(
                                &graph_conn,
                                name,
                                namespace,
                                &embedding,
                            )
                            .await?;
                            println!("  ✓ {name}");
                            success += 1;
                        }
                        Err(e) => {
                            eprintln!("  ✗ {name}: {e}");
                            failed += 1;
                        }
                    }
                }
                println!("\nBackfill complete: {success} success, {failed} failed");
            }
            BackfillCommands::PatchContent { dry_run } => {
                use std::path::{Path, PathBuf};

                let patches = graph::find_all_patches(&graph_conn).await?;

                // Build the set of directories to search for patch files:
                // parent dirs of patch refs that already resolve, plus a
                // depth-bounded scan of $HOME for `patches/` directories.
                let mut search_dirs: Vec<PathBuf> = Vec::new();
                let mut push_dir = |d: PathBuf, dirs: &mut Vec<PathBuf>| {
                    if d.is_dir() && !dirs.contains(&d) {
                        dirs.push(d);
                    }
                };
                for (_, _, file, _) in &patches {
                    if let Some(f) = file {
                        let expanded = shellexpand::tilde(f).to_string();
                        let p = Path::new(&expanded);
                        if p.is_absolute() && p.exists() {
                            if let Some(parent) = p.parent() {
                                push_dir(parent.to_path_buf(), &mut search_dirs);
                            }
                        }
                    }
                }
                if let Some(home) = dirs::home_dir() {
                    discover_patch_dirs(&home, 5, &mut search_dirs);
                }

                enum Resolution {
                    Found(PathBuf),
                    Ambiguous(Vec<PathBuf>),
                    NotFound,
                }

                let resolve = |file: &Option<String>, name: &str, namespace: &str| -> Resolution {
                    // 1. Absolute, existing ref — unambiguous, use directly.
                    if let Some(f) = file {
                        let exp = shellexpand::tilde(f).to_string();
                        let p = Path::new(&exp);
                        if p.is_absolute() && p.exists() {
                            return Resolution::Found(p.to_path_buf());
                        }
                    }
                    // 2. Otherwise gather every candidate by basename across dirs.
                    let basename = file
                        .as_ref()
                        .and_then(|f| {
                            Path::new(f)
                                .file_name()
                                .map(|s| s.to_string_lossy().to_string())
                        })
                        .unwrap_or_else(|| format!("{name}.md"));
                    let mut cands: Vec<PathBuf> = search_dirs
                        .iter()
                        .map(|d| d.join(&basename))
                        .filter(|c| c.exists())
                        .collect();
                    cands.sort();
                    cands.dedup();
                    if cands.len() <= 1 {
                        return cands.pop().map_or(Resolution::NotFound, Resolution::Found);
                    }
                    // Collision: prefer a candidate whose path is scoped to this
                    // namespace (e.g. .../<namespace>/.c0/patches/...). Refuse to
                    // guess if that doesn't disambiguate — inlining the wrong
                    // client's data is worse than leaving the patch empty.
                    let ns_seg = format!("/{namespace}/");
                    let ns_matches: Vec<PathBuf> = cands
                        .iter()
                        .filter(|c| c.to_string_lossy().contains(&ns_seg))
                        .cloned()
                        .collect();
                    match ns_matches.len() {
                        1 => Resolution::Found(ns_matches.into_iter().next().unwrap()),
                        _ => Resolution::Ambiguous(cands),
                    }
                };

                let needing: Vec<_> = patches
                    .iter()
                    .filter(|(_, _, _, has_content)| !has_content)
                    .collect();
                println!(
                    "Patches without inline content: {} (of {} total)",
                    needing.len(),
                    patches.len()
                );
                println!("Searching {} candidate directories\n", search_dirs.len());

                let mut recovered = 0;
                let mut unresolved: Vec<String> = Vec::new();
                let mut ambiguous: Vec<String> = Vec::new();
                for (name, namespace, file, _) in &needing {
                    match resolve(file, name, namespace) {
                        Resolution::Found(path) => match std::fs::read_to_string(&path) {
                            Ok(content) if !content.trim().is_empty() => {
                                let abs = path.to_string_lossy().to_string();
                                if dry_run {
                                    println!("  would inline  {name} [{namespace}]  <- {abs}");
                                } else if graph::set_patch_content(
                                    &graph_conn,
                                    name,
                                    namespace,
                                    &content,
                                    &abs,
                                )
                                .await?
                                {
                                    println!("  ✓ inlined     {name} [{namespace}]  <- {abs}");
                                    recovered += 1;
                                }
                            }
                            _ => unresolved
                                .push(format!("{name} [{namespace}] (file empty/unreadable)")),
                        },
                        Resolution::Ambiguous(cands) => {
                            ambiguous.push(format!(
                                "{name} [{namespace}] -> {} candidates: {}",
                                cands.len(),
                                cands
                                    .iter()
                                    .map(|c| c.to_string_lossy().to_string())
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            ));
                        }
                        Resolution::NotFound => {
                            unresolved.push(format!("{name} [{namespace}] (no source file found)"))
                        }
                    }
                }

                if !ambiguous.is_empty() {
                    println!(
                        "\nAmbiguous ({}): basename collides across namespaces, NOT inlined (fix manually) —",
                        ambiguous.len()
                    );
                    for a in &ambiguous {
                        println!("  ? {a}");
                    }
                }
                if !unresolved.is_empty() {
                    println!(
                        "\nUnresolved ({}): content not found on this machine —",
                        unresolved.len()
                    );
                    for u in &unresolved {
                        println!("  - {u}");
                    }
                    println!(
                        "  (these may have their source file on another host, or it was deleted)"
                    );
                }
                if dry_run {
                    println!("\n(dry run - no changes made)");
                } else {
                    println!(
                        "\nBackfill complete: {recovered} patches inlined, {} unresolved",
                        unresolved.len()
                    );
                }
            }
        },
        Commands::Extract {
            input,
            index,
            dry_run,
        } => {
            use std::path::PathBuf;

            let input_path = PathBuf::from(shellexpand::tilde(&input).to_string());
            if !input_path.exists() {
                eprintln!("Error: Input file not found: {}", input_path.display());
                return Ok(());
            }

            let extract_config = extract::extract_config();
            println!("📝 Extracting transcript from: {}", input_path.display());
            println!(
                "   Using model: {} on {}",
                extract_config.model, extract_config.host
            );
            println!("   This may take a few minutes...\n");

            let extraction = extract::extract_transcript(&input_path).await?;

            let summary_md = extract::generate_summary_markdown(&extraction, &ctx.namespace);
            let patch_name = extract::get_patch_name(
                &extraction.metadata.date,
                &extraction.metadata.title,
                &ctx.namespace,
            );

            println!("✓ Extraction complete!\n");
            println!("{summary_md}");

            if dry_run {
                println!("\n(dry run - no files written or graph changes made)");
                return Ok(());
            }

            let patches_dir = config::get_patches_dir(&ctx);
            let transcripts_dir = patches_dir.join("transcripts");
            std::fs::create_dir_all(&transcripts_dir)?;

            let summary_path =
                transcripts_dir.join(format!("{}-summary.md", extraction.metadata.date));
            std::fs::write(&summary_path, &summary_md)?;
            println!("📄 Wrote summary to: {}", summary_path.display());

            if index {
                println!("\n🔗 Indexing into c0 graph...");

                let summary_path_str = summary_path.to_string_lossy().to_string();
                graph::add_patch(
                    &graph_conn,
                    &patch_name,
                    None,
                    Some(&summary_path_str),
                    None,
                    &ctx.namespace,
                    Some("transcript-extraction"),
                    None,
                    None,
                )
                .await?;
                println!("   Created patch: {patch_name}");

                let ns_exists =
                    !graph::search_concepts(&graph_conn, &ctx.namespace, &ctx.namespaces)
                        .await?
                        .is_empty();
                if ns_exists {
                    graph::relate(
                        &graph_conn,
                        &ctx.namespace,
                        "HAS_PATCH",
                        &patch_name,
                        &ctx.namespaces,
                    )
                    .await?;
                    println!("   Linked: {} -[HAS_PATCH]-> {patch_name}", ctx.namespace);
                }

                let semantic_config = config::SemanticConfig::load();
                if let Some(client) = embeddings::OllamaClient::from_config(&semantic_config) {
                    graph::ensure_concept_unique_constraint(&graph_conn).await?;
                    graph::ensure_concept_embedding_index(&graph_conn).await?;

                    for topic in &extraction.topics {
                        let concept_name = &topic.normalized_name;
                        let embed_text = format!("{}: {}", topic.name, topic.context);

                        match client.embed(&embed_text).await {
                            Ok(embedding) => {
                                graph::add_concept(
                                    &graph_conn,
                                    concept_name,
                                    &ctx.namespace,
                                    Some(&topic.context),
                                    Some("transcript-extraction"),
                                    None,
                                    Some(&embedding),
                                    None,
                                )
                                .await?;
                                println!("   Created concept: {concept_name}");

                                graph::relate(
                                    &graph_conn,
                                    concept_name,
                                    "DISCUSSED_IN",
                                    &patch_name,
                                    &ctx.namespaces,
                                )
                                .await?;
                            }
                            Err(e) => {
                                eprintln!("   ⚠ Could not create concept {concept_name}: {e}");
                            }
                        }
                    }
                }

                println!("\n✓ Indexing complete!");
                println!("\nTest with:");
                println!("  c0 walk {}", ctx.namespace);
                for topic in extraction.topics.iter().take(3) {
                    println!("  c0 walk \"{}\"", topic.normalized_name);
                }
            }
        }
        Commands::Invalidate { what } => match what {
            InvalidateCommands::Concept {
                name,
                as_of,
                by,
                reason,
            } => {
                let invalid_at = as_of
                    .as_ref()
                    .map(|s| parse_date_to_datetime(s))
                    .transpose()?;

                match graph::invalidate_concept(
                    &graph_conn,
                    &name,
                    &ctx.namespace,
                    invalid_at,
                    by.as_deref(),
                    reason.as_deref(),
                    &ctx.namespaces,
                )
                .await?
                {
                    Some(invalid_dt) => {
                        println!("✓ Invalidated concept: {} [{}]", name, ctx.namespace);
                        println!("  invalid_at: {invalid_dt}");
                        if let Some(ref by_name) = by {
                            println!("  invalidated_by: {by_name}");
                        }
                        if let Some(ref r) = reason {
                            println!("  reason: {r}");
                        }
                    }
                    None => {
                        eprintln!(
                            "Concept '{}' not found or already invalidated in namespace '{}'",
                            name, ctx.namespace
                        );
                    }
                }
            }
            InvalidateCommands::Patch {
                name,
                as_of,
                by,
                reason,
            } => {
                let invalid_at = as_of
                    .as_ref()
                    .map(|s| parse_date_to_datetime(s))
                    .transpose()?;

                match graph::invalidate_patch(
                    &graph_conn,
                    &name,
                    &ctx.namespace,
                    invalid_at,
                    by.as_deref(),
                    reason.as_deref(),
                    &ctx.namespaces,
                )
                .await?
                {
                    Some(invalid_dt) => {
                        println!("✓ Invalidated patch: {name}");
                        println!("  invalid_at: {invalid_dt}");
                        if let Some(ref by_name) = by {
                            println!("  invalidated_by: {by_name}");
                        }
                        if let Some(ref r) = reason {
                            println!("  reason: {r}");
                        }
                    }
                    None => {
                        eprintln!("Patch '{name}' not found or already invalidated");
                    }
                }
            }
        },
        Commands::Supersede { old, with, as_of } => {
            let expired_at = as_of
                .as_ref()
                .map(|s| parse_date_to_datetime(s))
                .transpose()?;

            let success =
                graph::supersede_concept(&graph_conn, &old, &with, &ctx.namespaces, expired_at)
                    .await?;

            if success {
                println!("✓ Superseded: {old} → {with}");
                if let Some(ref dt) = as_of {
                    println!("  as_of: {dt}");
                }

                let chain =
                    graph::get_supersession_chain(&graph_conn, &with, &ctx.namespaces).await?;
                if chain.len() > 1 {
                    println!("\nSupersession chain:");
                    for (name, expired_at) in &chain {
                        if let Some(exp) = expired_at {
                            println!("  {name} (expired: {exp})");
                        } else {
                            println!("  {name} (current)");
                        }
                    }
                }
            } else {
                eprintln!("Failed to supersede: concepts not found or '{old}' already expired");
                eprintln!(
                    "Make sure both '{}' and '{}' exist in namespaces {:?}",
                    old, with, ctx.namespaces
                );
            }
        }
        Commands::ExtractConcepts {
            prompt,
            limit,
            known_only,
            json,
        } => {
            let extraction_config = config::ExtractionConfig::load();

            if !extraction_config.enabled {
                if json {
                    println!(
                        r#"{{"error":"extraction disabled","known":[],"unknown":[],"queued":false}}"#
                    );
                } else {
                    eprintln!("Concept extraction is disabled in config");
                }
                return Ok(());
            }

            let semantic_config = config::SemanticConfig::load();
            let client = claude::LlmClient::for_concept_extraction(
                &semantic_config,
                &extraction_config.model,
                extraction_config.timeout_secs,
            );

            let concepts = match client
                .extract_concepts(&prompt, &ctx.namespace, limit)
                .await
            {
                Ok(c) => c,
                Err(e) => {
                    if json {
                        println!(r#"{{"error":"{e}","known":[],"unknown":[],"queued":false}}"#);
                    } else {
                        eprintln!("Extraction failed: {e}");
                    }
                    return Ok(());
                }
            };

            if concepts.is_empty() {
                if json {
                    println!(r#"{{"known":[],"unknown":[],"queued":false}}"#);
                }
                return Ok(());
            }

            let mut known_concepts = Vec::new();
            let mut unknown_concepts = Vec::new();

            for concept in &concepts {
                let search_results =
                    graph::search_concepts(&graph_conn, concept, &ctx.namespaces).await?;
                let has_match = search_results.iter().any(|r| {
                    let r_lower = r.to_lowercase();
                    r_lower == *concept || r_lower.contains(concept) || concept.contains(&r_lower)
                });
                if has_match {
                    if let Some(best_match) = search_results.first() {
                        known_concepts.push(best_match.clone());
                    } else {
                        known_concepts.push(concept.clone());
                    }
                } else {
                    unknown_concepts.push(concept.clone());
                }
            }

            let queued = if extraction_config.queue_unknown && !unknown_concepts.is_empty() {
                let session_id = get_session_id().unwrap_or_else(|| "unknown".to_string());
                if let Some(home) = dirs::home_dir() {
                    let inbox_dir = home.join(".c0/reflector");
                    if std::fs::create_dir_all(&inbox_dir).is_ok() {
                        let inbox_file = inbox_dir.join("inbox.jsonl");
                        if let Ok(mut f) = std::fs::OpenOptions::new()
                            .create(true)
                            .append(true)
                            .open(inbox_file)
                        {
                            for concept in &unknown_concepts {
                                let entry = serde_json::json!({
                                    "timestamp": chrono::Utc::now().to_rfc3339(),
                                    "session": session_id,
                                    "namespace": ctx.namespace,
                                    "command": "extract-concepts",
                                    "query": concept,
                                    "context": format!("Extracted from prompt: {}", &prompt[..std::cmp::min(100, prompt.len())]),
                                });
                                let _ = writeln!(f, "{entry}");
                            }
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                false
            };

            if known_only {
                if json {
                    let result = claude::ConceptExtractionResult {
                        known: known_concepts.clone(),
                        unknown: vec![],
                        queued: false,
                    };
                    println!("{}", serde_json::to_string(&result)?);
                } else {
                    for concept in &known_concepts {
                        println!("{concept}");
                    }
                }
            } else if json {
                let result = claude::ConceptExtractionResult {
                    known: known_concepts,
                    unknown: unknown_concepts,
                    queued,
                };
                println!("{}", serde_json::to_string(&result)?);
            } else {
                if !known_concepts.is_empty() {
                    println!("Known concepts:");
                    for concept in &known_concepts {
                        println!("  c0 walk {concept}");
                    }
                }
                if !unknown_concepts.is_empty() {
                    println!("Unknown concepts (queued to reflector):");
                    for concept in &unknown_concepts {
                        println!("  {concept}");
                    }
                }
            }
        }
        Commands::InvalidationChain { name } => {
            let chain = graph::get_invalidation_chain(&graph_conn, &name, &ctx.namespaces).await?;
            if chain.is_empty() {
                println!("No invalidation records for '{name}'");
            } else {
                println!("Invalidation chain for '{name}':");
                for record in chain {
                    let invalid_at_str =
                        record.invalid_at.as_deref().unwrap_or("(not invalidated)");
                    println!("  {} (invalid_at: {})", record.name, invalid_at_str);
                    if let Some(by) = record.invalidated_by {
                        println!("    └─ INVALIDATED_BY: {by}");
                    }
                    if let Some(reason) = record.reason {
                        println!("       reason: \"{reason}\"");
                    }
                }
            }
        }
        Commands::Audit { action } => match action {
            AuditCommands::Staleness {
                namespace,
                days,
                json,
            } => {
                let ns = namespace.as_ref().unwrap_or(&ctx.namespace);
                audit::staleness(&graph_conn, ns, &ctx.namespaces, days, json).await?;
            }
            AuditCommands::Namespaces { suggest, json } => {
                audit::namespaces(&graph_conn, &ctx.namespaces, suggest, json).await?;
            }
            AuditCommands::All { json } => {
                audit::staleness(&graph_conn, &ctx.namespace, &ctx.namespaces, 90, json).await?;
                if !json {
                    println!();
                }
                audit::namespaces(&graph_conn, &ctx.namespaces, false, json).await?;
            }
            AuditCommands::Enrich {
                namespace,
                all,
                same_threshold,
                cross_threshold,
                max_links,
                dry_run,
                rollback,
                json,
            } => {
                if let Some(run) = rollback {
                    let run = if run.is_empty() {
                        None
                    } else {
                        Some(run.as_str())
                    };
                    audit::enrich_rollback(&graph_conn, run, json).await?;
                } else {
                    let targets: Vec<String> = if all {
                        ctx.namespaces.clone()
                    } else {
                        vec![namespace.unwrap_or_else(|| ctx.namespace.clone())]
                    };
                    audit::enrich(
                        &graph_conn,
                        &targets,
                        same_threshold,
                        cross_threshold,
                        max_links,
                        dry_run,
                        json,
                    )
                    .await?;
                }
            }
        },
        Commands::Move { what } => match what {
            MoveCommands::Concept {
                name,
                to,
                with_patches,
            } => match graph::move_concept(&graph_conn, &name, &to, with_patches).await? {
                Some(result) => {
                    println!(
                        "✓ Moved concept: {} [{} → {}]",
                        name, result.old_namespace, result.new_namespace
                    );
                    if result.patches_moved > 0 {
                        println!("  Also moved {} patch(es)", result.patches_moved);
                    }
                }
                None => {
                    eprintln!("Concept '{name}' not found or already in namespace '{to}'");
                }
            },
            MoveCommands::Prefix {
                prefix,
                to,
                from,
                with_patches,
                dry_run,
            } => {
                let concepts = graph::list_concepts_by_prefix(&graph_conn, &prefix, &from).await?;

                if concepts.is_empty() {
                    println!("No concepts with prefix '{prefix}' found in namespace '{from}'");
                    return Ok(());
                }

                println!(
                    "Found {} concept(s) with prefix '{}' in '{}':",
                    concepts.len(),
                    prefix,
                    from
                );
                for name in &concepts {
                    println!("  {name}");
                }

                if dry_run {
                    println!("\n(dry run - no changes made)");
                    println!("To execute: c0 move prefix {prefix} --from {from} --to {to}");
                    return Ok(());
                }

                let (concepts_moved, patches_moved) =
                    graph::move_concepts_by_prefix(&graph_conn, &prefix, &from, &to, with_patches)
                        .await?;

                println!("\n✓ Moved {concepts_moved} concept(s) from '{from}' to '{to}'");
                if patches_moved > 0 {
                    println!("  Also moved {patches_moved} patch(es)");
                }
            }
        },
        Commands::Search {
            query,
            limit,
            threshold,
            json,
            vector_only,
            keyword_only,
        } => {
            let hybrid_config = graph::HybridSearchConfig {
                vector_threshold: threshold,
                ..Default::default()
            };

            let results = if keyword_only {
                graph::search_concepts_fulltext(&graph_conn, &query, limit, &ctx.namespaces).await?
            } else {
                let semantic_config = config::SemanticConfig::load();
                let client =
                    if let Some(c) = embeddings::OllamaClient::from_config(&semantic_config) {
                        c
                    } else {
                        eprintln!("Error: Semantic search not configured. Check ~/.c0/config.toml");
                        return Ok(());
                    };
                let query_embedding = client.embed(&query).await?;

                if vector_only {
                    graph::search_concepts_semantic(
                        &graph_conn,
                        &query_embedding,
                        limit,
                        threshold,
                        &ctx.namespaces,
                    )
                    .await?
                } else {
                    graph::search_hybrid(
                        &graph_conn,
                        &query,
                        &query_embedding,
                        limit,
                        &ctx.namespaces,
                        &hybrid_config,
                    )
                    .await?
                }
            };

            if results.is_empty() {
                println!("No concepts found matching \"{query}\"");
                return Ok(());
            }

            let mode_label = if keyword_only {
                "keyword"
            } else if vector_only {
                "vector"
            } else {
                "hybrid"
            };

            if json {
                println!("{}", serde_json::to_string_pretty(&results)?);
            } else {
                if vector_only {
                    println!(
                        "{:<6} {:<16} {:<30} {}",
                        "score", "namespace", "name", "description"
                    );
                } else {
                    println!(
                        "{:<10} {:<16} {:<30} {}",
                        "score", "namespace", "name", "description"
                    );
                }
                println!("{}", "─".repeat(80));
                for r in &results {
                    let desc = r.description.as_deref().unwrap_or("");
                    let desc_truncated = if desc.len() > 40 {
                        format!("{}...", &desc[..37])
                    } else {
                        desc.to_string()
                    };
                    if vector_only {
                        println!(
                            "{:<6.0}% {:<16} {:<30} {}",
                            r.similarity * 100.0,
                            r.namespace,
                            r.name,
                            desc_truncated
                        );
                    } else {
                        println!(
                            "{:<10.6} {:<16} {:<30} {}",
                            r.similarity, r.namespace, r.name, desc_truncated
                        );
                    }
                }
                println!(
                    "\n({mode_label} search, {count} results)",
                    count = results.len()
                );
            }
        }
        Commands::Export {
            format,
            namespace,
            output,
            no_embeddings,
        } => {
            let graph_export =
                export::export_graph(&graph_conn, namespace.as_deref(), no_embeddings).await?;

            let content = match format.as_str() {
                "cypher" => export::format_cypher(&graph_export),
                _ => export::format_json(&graph_export)?,
            };

            if let Some(ref path) = output {
                std::fs::write(path, &content)?;
                println!(
                    "Exported {} nodes and {} relationships to {}",
                    graph_export.metadata.node_count,
                    graph_export.metadata.relationship_count,
                    path
                );
            } else {
                println!("{content}");
            }
        }
        #[cfg(feature = "sessions")]
        Commands::Sessions { .. } | Commands::Turns { .. } => unreachable!(),
        Commands::Init { .. }
        | Commands::Status
        | Commands::Reflector { .. }
        | Commands::Config { .. }
        | Commands::Health { .. } => unreachable!(),
    }

    Ok(())
}
