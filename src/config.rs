use anyhow::Result;
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
pub struct ProjectConfig {
    pub namespace: String,
    #[serde(default)]
    pub parent_namespace: Option<String>,
    #[serde(default = "default_inherit")]
    pub inherit_global: bool,
    #[serde(default)]
    pub project_type: Option<String>,
}

fn default_inherit() -> bool {
    true
}

#[derive(Debug, Deserialize, Clone)]
pub struct GlobalConfig {
    #[serde(default)]
    pub ollama: OllamaConfig,
    #[serde(default)]
    pub semantic: SemanticSettings,
    #[serde(default)]
    pub claude: ClaudeConfig,
    #[serde(default)]
    pub extraction: ExtractionSettings,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ExtractionSettings {
    #[serde(default = "default_concept_extraction_enabled")]
    pub enabled: bool,
    #[serde(default = "default_concept_extraction_model")]
    pub model: String,
    #[serde(default = "default_concept_extraction_max_concepts")]
    pub max_concepts: usize,
    #[serde(default = "default_concept_extraction_queue_unknown")]
    pub queue_unknown: bool,
    #[serde(default = "default_concept_extraction_timeout")]
    pub timeout_secs: u64,
}

impl Default for ExtractionSettings {
    fn default() -> Self {
        Self {
            enabled: default_concept_extraction_enabled(),
            model: default_concept_extraction_model(),
            max_concepts: default_concept_extraction_max_concepts(),
            queue_unknown: default_concept_extraction_queue_unknown(),
            timeout_secs: default_concept_extraction_timeout(),
        }
    }
}

fn default_concept_extraction_enabled() -> bool {
    true
}

fn default_concept_extraction_model() -> String {
    "haiku".to_string()
}

fn default_concept_extraction_max_concepts() -> usize {
    3
}

fn default_concept_extraction_queue_unknown() -> bool {
    true
}

fn default_concept_extraction_timeout() -> u64 {
    90
}

#[derive(Debug, Deserialize, Clone)]
pub struct ClaudeConfig {
    #[serde(default = "default_claude_enabled")]
    pub enabled: bool,
    #[serde(default = "default_llm_provider")]
    pub provider: String,
    #[serde(default = "default_classification_model")]
    pub classification_model: String,
    #[serde(default = "default_extraction_model")]
    pub extraction_model: String,
    #[serde(default = "default_enrichment_model")]
    pub enrichment_model: String,
    #[serde(default = "default_claude_timeout")]
    pub timeout_secs: u64,
    #[serde(default)]
    pub max_budget_usd: Option<f64>,
    #[serde(default)]
    pub reflector_session_file: Option<String>,
    #[serde(default)]
    pub binaries: LlmBinaries,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub enrichment_provider: Option<String>,
    #[serde(default)]
    pub classification_provider: Option<String>,
    #[serde(default)]
    pub extraction_provider: Option<String>,
    #[serde(default)]
    pub concept_extraction_provider: Option<String>,
}

impl ClaudeConfig {
    pub fn resolved_api_key(&self) -> Option<String> {
        std::env::var("ANTHROPIC_API_KEY")
            .ok()
            .filter(|s| !s.is_empty())
            .or_else(|| self.api_key.clone().filter(|s| !s.is_empty()))
    }

    pub fn provider_for(&self, task: &str) -> String {
        let override_value = match task {
            "enrichment" => self.enrichment_provider.as_ref(),
            "classification" => self.classification_provider.as_ref(),
            "extraction" => self.extraction_provider.as_ref(),
            "concept_extraction" => self.concept_extraction_provider.as_ref(),
            _ => None,
        };
        override_value
            .cloned()
            .unwrap_or_else(|| {
                if self.enabled {
                    self.provider.clone()
                } else {
                    "ollama".to_string()
                }
            })
    }
}

impl Default for ClaudeConfig {
    fn default() -> Self {
        Self {
            enabled: default_claude_enabled(),
            provider: default_llm_provider(),
            classification_model: default_classification_model(),
            extraction_model: default_extraction_model(),
            enrichment_model: default_enrichment_model(),
            timeout_secs: default_claude_timeout(),
            max_budget_usd: None,
            reflector_session_file: None,
            binaries: LlmBinaries::default(),
            api_key: None,
            enrichment_provider: None,
            classification_provider: None,
            extraction_provider: None,
            concept_extraction_provider: None,
        }
    }
}

fn default_claude_enabled() -> bool {
    false
}

fn default_llm_provider() -> String {
    "claude".to_string()
}

fn default_classification_model() -> String {
    "sonnet".to_string()
}

fn default_extraction_model() -> String {
    "sonnet".to_string()
}

fn default_enrichment_model() -> String {
    "haiku".to_string()
}

fn default_claude_timeout() -> u64 {
    120
}

#[derive(Debug, Deserialize, Clone)]
pub struct LlmBinaries {
    #[serde(default = "default_claude_binary")]
    pub claude: String,
    #[serde(default = "default_droid_binary")]
    pub droid: String,
    #[serde(default = "default_codex_binary")]
    pub codex: String,
    #[serde(default = "default_kilo_binary")]
    pub kilo: String,
    #[serde(default = "default_gemini_binary")]
    pub gemini: String,
}

impl Default for LlmBinaries {
    fn default() -> Self {
        Self {
            claude: default_claude_binary(),
            droid: default_droid_binary(),
            codex: default_codex_binary(),
            kilo: default_kilo_binary(),
            gemini: default_gemini_binary(),
        }
    }
}

fn default_claude_binary() -> String {
    "claude".to_string()
}

fn default_droid_binary() -> String {
    "droid".to_string()
}

fn default_codex_binary() -> String {
    "codex".to_string()
}

fn default_kilo_binary() -> String {
    "kilocode".to_string()
}

fn default_gemini_binary() -> String {
    "gemini".to_string()
}

#[derive(Debug, Deserialize, Clone)]
pub struct OllamaConfig {
    #[serde(default = "default_ollama_host")]
    pub host: String,
    #[serde(default = "default_ollama_model")]
    pub model: String,
    #[serde(default = "default_ollama_timeout")]
    pub timeout_ms: u64,
    #[serde(default = "default_reflector_model")]
    pub reflector_model: String,
    #[serde(default = "default_ollama_enrichment_model")]
    pub enrichment_model: String,
    #[serde(default = "default_ollama_extraction_model")]
    pub extraction_model: String,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            host: default_ollama_host(),
            model: default_ollama_model(),
            timeout_ms: default_ollama_timeout(),
            reflector_model: default_reflector_model(),
            enrichment_model: default_ollama_enrichment_model(),
            extraction_model: default_ollama_extraction_model(),
        }
    }
}

fn default_ollama_host() -> String {
    "http://localhost:11434".to_string()
}

fn default_ollama_model() -> String {
    "nomic-embed-text".to_string()
}

fn default_ollama_timeout() -> u64 {
    5000
}

fn default_reflector_model() -> String {
    "hermes3:8b".to_string()
}

fn default_ollama_enrichment_model() -> String {
    "qwen2.5:14b".to_string()
}

fn default_ollama_extraction_model() -> String {
    "qwen2.5:14b".to_string()
}

#[derive(Debug, Deserialize, Clone)]
pub struct SemanticSettings {
    #[serde(default = "default_semantic_enabled")]
    pub enabled: bool,
    #[serde(default = "default_semantic_threshold")]
    pub default_threshold: f32,
    #[serde(default = "default_fallback_to_regex")]
    pub fallback_to_regex: bool,
    #[serde(default = "default_query_floor_threshold")]
    pub query_floor_threshold: f32,
}

impl Default for SemanticSettings {
    fn default() -> Self {
        Self {
            enabled: default_semantic_enabled(),
            default_threshold: default_semantic_threshold(),
            fallback_to_regex: default_fallback_to_regex(),
            query_floor_threshold: default_query_floor_threshold(),
        }
    }
}

fn default_semantic_enabled() -> bool {
    true
}

fn default_semantic_threshold() -> f32 {
    0.75
}

fn default_fallback_to_regex() -> bool {
    true
}

fn default_query_floor_threshold() -> f32 {
    0.3
}

#[derive(Debug, Clone)]
pub struct SemanticConfig {
    pub enabled: bool,
    pub ollama_host: String,
    pub ollama_model: String,
    pub ollama_timeout_ms: u64,
    pub default_threshold: f32,
    pub fallback_to_regex: bool,
    pub reflector_model: String,
    pub ollama_enrichment_model: String,
    pub ollama_extraction_model: String,
    pub query_floor_threshold: f32,
    pub claude: ClaudeConfig,
}

impl SemanticConfig {
    pub fn load() -> Self {
        let global_config = load_global_config();
        Self {
            enabled: global_config.semantic.enabled,
            ollama_host: global_config.ollama.host,
            ollama_model: global_config.ollama.model,
            ollama_timeout_ms: global_config.ollama.timeout_ms,
            default_threshold: global_config.semantic.default_threshold,
            fallback_to_regex: global_config.semantic.fallback_to_regex,
            reflector_model: global_config.ollama.reflector_model,
            ollama_enrichment_model: global_config.ollama.enrichment_model,
            ollama_extraction_model: global_config.ollama.extraction_model,
            query_floor_threshold: global_config.semantic.query_floor_threshold,
            claude: global_config.claude,
        }
    }

    pub fn model_for(&self, task: &str, provider: &str) -> String {
        if provider.eq_ignore_ascii_case("gemini") {
            return String::new();
        }
        let is_ollama = provider.eq_ignore_ascii_case("ollama");
        match (task, is_ollama) {
            ("enrichment", true) => self.ollama_enrichment_model.clone(),
            ("classification", true) => self.reflector_model.clone(),
            ("extraction", true) => self.ollama_extraction_model.clone(),
            ("enrichment", false) => self.claude.enrichment_model.clone(),
            ("classification", false) => self.claude.classification_model.clone(),
            ("extraction", false) => self.claude.extraction_model.clone(),
            _ => self.claude.extraction_model.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExtractionConfig {
    pub enabled: bool,
    pub model: String,
    pub max_concepts: usize,
    pub queue_unknown: bool,
    pub timeout_secs: u64,
}

impl ExtractionConfig {
    pub fn load() -> Self {
        let global_config = load_global_config();
        Self {
            enabled: global_config.extraction.enabled,
            model: global_config.extraction.model,
            max_concepts: global_config.extraction.max_concepts,
            queue_unknown: global_config.extraction.queue_unknown,
            timeout_secs: global_config.extraction.timeout_secs,
        }
    }
}

fn load_global_config() -> GlobalConfig {
    let config_path = dirs::home_dir().map_or_else(|| PathBuf::from(".c0/config.toml"), |h| h.join(".c0/config.toml"));

    if config_path.exists()
        && let Ok(content) = std::fs::read_to_string(&config_path)
            && let Ok(config) = toml::from_str::<GlobalConfig>(&content) {
                return config;
            }

    GlobalConfig {
        ollama: OllamaConfig::default(),
        semantic: SemanticSettings::default(),
        claude: ClaudeConfig::default(),
        extraction: ExtractionSettings::default(),
    }
}

#[derive(Debug)]
pub struct NamespaceContext {
    pub namespace: String,
    pub project_dir: Option<PathBuf>,
    pub parent_dirs: Vec<PathBuf>,
    pub namespaces: Vec<String>,
    pub project_type: Option<String>,
}

impl NamespaceContext {
    pub fn global() -> Self {
        NamespaceContext {
            namespace: "global".to_string(),
            project_dir: None,
            parent_dirs: Vec::new(),
            namespaces: vec!["global".to_string()],
            project_type: None,
        }
    }
}

pub fn detect_namespace() -> NamespaceContext {
    let cwd = match std::env::current_dir() {
        Ok(d) => d,
        Err(_) => return NamespaceContext::global(),
    };

    let mut dir = cwd;
    loop {
        let c0_dir = dir.join(".c0");
        let config_file = c0_dir.join("config.toml");

        if config_file.exists()
            && let Ok(config) = read_config(&config_file) {
                let (parent_dirs, parent_namespaces) = resolve_parent_chain(
                    config.parent_namespace.as_deref(),
                    &c0_dir,
                );

                let mut namespaces = vec![config.namespace.clone()];
                namespaces.extend(parent_namespaces);
                if config.inherit_global && !namespaces.contains(&"global".to_string()) {
                    namespaces.push("global".to_string());
                }

                return NamespaceContext {
                    namespace: config.namespace,
                    project_dir: Some(c0_dir),
                    parent_dirs,
                    namespaces,
                    project_type: config.project_type,
                };
            }

        if !dir.pop() {
            break;
        }
    }

    NamespaceContext::global()
}

fn resolve_parent_chain(
    parent_namespace: Option<&str>,
    current_c0_dir: &Path,
) -> (Vec<PathBuf>, Vec<String>) {
    let mut parent_dirs = Vec::new();
    let mut parent_namespaces = Vec::new();
    let mut seen_namespaces = std::collections::HashSet::new();

    let mut current_parent = parent_namespace.map(std::string::ToString::to_string);
    let mut search_start = current_c0_dir
        .parent()
        .and_then(|p| p.parent())
        .map(std::path::Path::to_path_buf);

    while let Some(ref parent_ns) = current_parent {
        if seen_namespaces.contains(parent_ns) {
            eprintln!("Warning: Circular parent reference detected for '{parent_ns}'");
            break;
        }
        seen_namespaces.insert(parent_ns.clone());

        let found = find_namespace_dir(parent_ns, search_start.as_deref());

        if let Some((parent_c0_dir, parent_config)) = found {
            parent_dirs.push(parent_c0_dir.clone());
            parent_namespaces.push(parent_ns.clone());

            search_start = parent_c0_dir
                .parent()
                .and_then(|p| p.parent())
                .map(std::path::Path::to_path_buf);
            current_parent = parent_config.parent_namespace;
        } else {
            parent_namespaces.push(parent_ns.clone());
            break;
        }
    }

    (parent_dirs, parent_namespaces)
}

fn find_namespace_dir(namespace: &str, start_dir: Option<&Path>) -> Option<(PathBuf, ProjectConfig)> {
    let mut search_dir = start_dir.map(std::path::Path::to_path_buf);

    while let Some(ref dir) = search_dir {
        let c0_dir = dir.join(".c0");
        let config_file = c0_dir.join("config.toml");

        if config_file.exists()
            && let Ok(config) = read_config(&config_file)
                && config.namespace == namespace {
                    return Some((c0_dir, config));
                }

        search_dir = dir.parent().map(std::path::Path::to_path_buf);
    }

    if let Some(home) = dirs::home_dir() {
        let global_c0 = home.join(".c0");
        let config_file = global_c0.join("config.toml");
        if config_file.exists()
            && let Ok(config) = read_config(&config_file)
                && config.namespace == namespace {
                    return Some((global_c0, config));
                }
    }

    None
}

fn read_config(path: &Path) -> Result<ProjectConfig> {
    let content = std::fs::read_to_string(path)?;
    let config: ProjectConfig = toml::from_str(&content)?;
    Ok(config)
}

pub fn init_namespace(
    namespace: &str,
    project_type: &str,
    parent_namespace: Option<&str>,
) -> Result<PathBuf> {
    let cwd = std::env::current_dir()?;
    let c0_dir = cwd.join(".c0");

    std::fs::create_dir_all(&c0_dir)?;
    std::fs::create_dir_all(c0_dir.join("patches"))?;

    let mut config_content = format!(
        r#"namespace = "{namespace}"
"#
    );

    if let Some(parent) = parent_namespace {
        config_content.push_str(&format!("parent_namespace = \"{parent}\"\n"));
    }

    config_content.push_str("inherit_global = true\n");

    if project_type != "default" {
        config_content.push_str(&format!("project_type = \"{project_type}\"\n"));
    }

    std::fs::write(c0_dir.join("config.toml"), config_content)?;

    let triggers_content = if project_type == "solution" {
        format!(
            r"# {namespace} Memory Trigger Patterns (Solution Architecture Project)
# These triggers prompt c0 memory checks during conversations

# Process phases
discovery
kickoff
onboarding
build phase
uat
launch
post-launch

# Key artifacts
tech stack
field mapping
site map
features and functionality
solution architecture

# Decision points
paid discovery
data transition
acceptance criteria

# Client-specific (add as you learn)
"
        )
    } else {
        format!(
            "# {namespace} Memory Trigger Patterns\n# Add project-specific triggers here\n\n"
        )
    };
    std::fs::write(c0_dir.join("triggers.txt"), triggers_content)?;

    if project_type == "solution" {
        let client_overview = format!(
            r"# {} - Client Overview

> Source: Initial discovery / Intake form

## Client Info
- **Company**:
- **Industry**:
- **Main Contact**:

## Project Scope
- **Type**: (Shopify Build / Migration / Managed Services)
- **Timeline**:
- **Budget Range**:

## Key Decisions
<!-- Add decisions as they're made, with sources -->

## Tech Stack
<!-- Track technology decisions here -->

## Open Questions
<!-- Questions to resolve during discovery -->
",
            namespace,
        );
        std::fs::write(c0_dir.join("patches").join("client-overview.md"), client_overview)?;

        let claude_md = format!(
            r#"# {} - Solution Project

This is a solution architecture project.

## On Session Start

1. Run `c0 status` to confirm namespace is active
2. Run `c0 walk <topic>` before answering SA questions
3. Update `.c0/patches/client-overview.md` with new learnings

## Key Commands

- `c0 walk discovery` - Load SA methodology
- `c0 walk <topic>` - Check for relevant knowledge
- `c0 add concept <name> --source "meeting notes"` - Capture new knowledge
"#,
            namespace,
        );
        std::fs::write(cwd.join("CLAUDE.md"), claude_md)?;
    }

    Ok(c0_dir)
}

pub fn get_triggers_files(ctx: &NamespaceContext) -> Vec<PathBuf> {
    let mut files = Vec::new();

    if let Some(home) = dirs::home_dir() {
        let global = home.join(".c0/triggers.txt");
        if global.exists() {
            files.push(global);
        }
    }

    for parent_dir in ctx.parent_dirs.iter().rev() {
        let triggers = parent_dir.join("triggers.txt");
        if triggers.exists() {
            files.push(triggers);
        }
    }

    if let Some(ref project_dir) = ctx.project_dir {
        let project = project_dir.join("triggers.txt");
        if project.exists() {
            files.push(project);
        }
    }

    files
}

pub fn get_patches_dir(ctx: &NamespaceContext) -> PathBuf {
    if let Some(ref project_dir) = ctx.project_dir {
        project_dir.join("patches")
    } else {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".c0/patches")
    }
}

pub fn get_all_patches_dirs(ctx: &NamespaceContext) -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    if let Some(home) = dirs::home_dir() {
        dirs.push(home.join(".c0/patches"));
    }

    for parent_dir in ctx.parent_dirs.iter().rev() {
        dirs.push(parent_dir.join("patches"));
    }

    if let Some(ref project_dir) = ctx.project_dir {
        dirs.push(project_dir.join("patches"));
    }

    dirs.into_iter().filter(|d| d.exists()).collect()
}
