use anyhow::{anyhow, Result};
use serde::Deserialize;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

use crate::config::{ClaudeConfig, LlmBinaries, SemanticConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmProvider {
    Claude,
    ClaudeCli,
    Droid,
    Codex,
    Kilo,
    Gemini,
    Ollama,
}

impl LlmProvider {
    pub fn parse(value: &str) -> Self {
        match value.to_lowercase().as_str() {
            "droid" => Self::Droid,
            "codex" => Self::Codex,
            "kilo" | "kilocode" => Self::Kilo,
            "gemini" => Self::Gemini,
            "ollama" => Self::Ollama,
            "claude-cli" | "claude_cli" | "cli" | "subscription" => Self::ClaudeCli,
            _ => Self::Claude,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LlmClient {
    provider: LlmProvider,
    pub model: String,
    pub timeout_secs: u64,
    pub max_budget_usd: Option<f64>,
    binary: String,
    api_key: Option<String>,
    ollama_host: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct LlmResponse {
    pub result: String,
    #[serde(default)]
    pub total_cost_usd: Option<f64>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub is_error: bool,
}

#[derive(Debug, Deserialize)]
struct ClaudeStreamMessage {
    #[serde(rename = "type")]
    msg_type: String,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    result: Option<String>,
    #[serde(default)]
    total_cost_usd: Option<f64>,
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    is_error: Option<bool>,
}

impl LlmClient {
    pub fn from_config(config: &ClaudeConfig, model: &str) -> Self {
        Self::from_config_with_timeout(config, model, config.timeout_secs)
    }

    pub fn from_config_with_timeout(config: &ClaudeConfig, model: &str, timeout_secs: u64) -> Self {
        let provider = LlmProvider::parse(&config.provider);
        let binary = provider_binary(provider, &config.binaries);
        Self {
            provider,
            model: model.to_string(),
            timeout_secs,
            max_budget_usd: config.max_budget_usd,
            binary,
            api_key: config.resolved_api_key(),
            ollama_host: None,
        }
    }

    pub fn for_task(semantic_config: &SemanticConfig, task: &str, timeout_secs: u64) -> Self {
        let provider_name = semantic_config.claude.provider_for(task);
        let provider = LlmProvider::parse(&provider_name);
        let model = semantic_config.model_for(task, &provider_name);
        let binary = provider_binary(provider, &semantic_config.claude.binaries);
        Self {
            provider,
            model,
            timeout_secs,
            max_budget_usd: semantic_config.claude.max_budget_usd,
            binary,
            api_key: semantic_config.claude.resolved_api_key(),
            ollama_host: Some(semantic_config.ollama_host.clone()),
        }
    }

    pub fn for_concept_extraction(
        semantic_config: &SemanticConfig,
        claude_model: &str,
        timeout_secs: u64,
    ) -> Self {
        let provider_name = semantic_config.claude.provider_for("concept_extraction");
        let provider = LlmProvider::parse(&provider_name);
        let model = if matches!(provider, LlmProvider::Ollama) {
            semantic_config.ollama_extraction_model.clone()
        } else if matches!(provider, LlmProvider::Gemini) {
            String::new()
        } else {
            claude_model.to_string()
        };
        let binary = provider_binary(provider, &semantic_config.claude.binaries);
        Self {
            provider,
            model,
            timeout_secs,
            max_budget_usd: semantic_config.claude.max_budget_usd,
            binary,
            api_key: semantic_config.claude.resolved_api_key(),
            ollama_host: Some(semantic_config.ollama_host.clone()),
        }
    }

    pub fn provider_name(&self) -> &'static str {
        match self.provider {
            LlmProvider::Claude => "claude",
            LlmProvider::ClaudeCli => "claude-cli",
            LlmProvider::Droid => "droid",
            LlmProvider::Codex => "codex",
            LlmProvider::Kilo => "kilo",
            LlmProvider::Gemini => "gemini",
            LlmProvider::Ollama => "ollama",
        }
    }

    pub async fn generate(&self, prompt: &str, json_schema: Option<&str>) -> Result<LlmResponse> {
        self.generate_internal(prompt, json_schema, None).await
    }

    pub async fn generate_resume(
        &self,
        prompt: &str,
        session_id: &str,
        json_schema: Option<&str>,
    ) -> Result<LlmResponse> {
        self.generate_internal(prompt, json_schema, Some(session_id)).await
    }

    async fn generate_internal(
        &self,
        prompt: &str,
        json_schema: Option<&str>,
        resume_session: Option<&str>,
    ) -> Result<LlmResponse> {
        if matches!(self.provider, LlmProvider::Ollama) {
            return self.generate_ollama(prompt, json_schema).await;
        }
        if matches!(self.provider, LlmProvider::ClaudeCli) {
            return self.generate_claude(prompt, json_schema, resume_session).await;
        }
        if self.api_key.is_some() && matches!(self.provider, LlmProvider::Claude) {
            return self.generate_anthropic_api(prompt, json_schema).await;
        }
        match self.provider {
            LlmProvider::Claude => self.generate_claude(prompt, json_schema, resume_session).await,
            LlmProvider::Droid => self.generate_droid(prompt, resume_session).await,
            LlmProvider::Codex => self.generate_codex(prompt, json_schema, resume_session).await,
            LlmProvider::Gemini => self.generate_gemini(prompt, json_schema).await,
            LlmProvider::Kilo => self.generate_kilo(prompt).await,
            LlmProvider::Ollama => unreachable!("Ollama handled above"),
            LlmProvider::ClaudeCli => unreachable!("ClaudeCli handled above"),
        }
    }

    async fn generate_ollama(
        &self,
        prompt: &str,
        json_schema: Option<&str>,
    ) -> Result<LlmResponse> {
        let host = self
            .ollama_host
            .as_ref()
            .ok_or_else(|| anyhow!("Ollama host not configured for this LlmClient"))?;

        let mut body = serde_json::json!({
            "model": self.model,
            "prompt": prompt,
            "stream": false,
        });
        if json_schema.is_some() {
            body["format"] = serde_json::json!("json");
        }

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(self.timeout_secs))
            .connect_timeout(Duration::from_secs(10))
            .build()?;

        let resp = client
            .post(format!("{host}/api/generate"))
            .json(&body)
            .send()
            .await
            .map_err(|e| anyhow!("Ollama POST to {host} failed: {e}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body_text = resp.text().await.unwrap_or_default();
            anyhow::bail!("Ollama returned {status}: {body_text}");
        }

        let parsed: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| anyhow!("Failed to parse Ollama response: {e}"))?;

        let result = parsed
            .get("response")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();

        Ok(LlmResponse {
            result,
            total_cost_usd: Some(0.0),
            session_id: None,
            is_error: false,
        })
    }

    async fn generate_claude(
        &self,
        prompt: &str,
        json_schema: Option<&str>,
        resume_session: Option<&str>,
    ) -> Result<LlmResponse> {
        let has_schema = json_schema.is_some();
        // smithersai/claude-p is a wrapper that *is* the -p emulator,
        // driving interactive claude through a PTY. Two adjustments
        // when it's in use: don't pass `-p` (the wrapper would forward
        // it to its inner claude, which is already in interactive
        // mode); and don't pass `--json-schema` (interactive claude
        // doesn't honor it). For schema requests we inline the schema
        // into the prompt, same trick generate_gemini uses.
        let via_wrapper = std::path::Path::new(&self.binary)
            .file_name()
            .and_then(|s| s.to_str())
            .map(|s| s == "claude-p")
            .unwrap_or(false);

        let effective_prompt = if via_wrapper && has_schema {
            format!(
                "{prompt}\n\nRespond with ONLY a JSON object matching this schema. \
                 No prose, no markdown fences, no explanation.\n\nSchema:\n{}",
                json_schema.unwrap()
            )
        } else {
            prompt.to_string()
        };

        let mut args = Vec::new();
        if !via_wrapper {
            args.push("-p".to_string());
        }
        args.push(effective_prompt);
        args.push("--output-format".to_string());
        args.push(if has_schema { "json".to_string() } else { "stream-json".to_string() });
        args.push("--max-turns".to_string());
        args.push(if has_schema { "3".to_string() } else { "1".to_string() });
        args.push("--model".to_string());
        args.push(self.model.clone());
        args.push("--strict-mcp-config".to_string());
        args.push("--mcp-config".to_string());
        args.push("{\"mcpServers\":{}}".to_string());
        args.push("--allowedTools".to_string());
        // Native --json-schema path uses the StructuredOutput tool to
        // format its response. The wrapper inline-schema path uses no
        // tools (claude replies in plain JSON because the prompt asks
        // for it). Everything else runs tool-free.
        args.push(if has_schema && !via_wrapper {
            "StructuredOutput".to_string()
        } else {
            String::new()
        });

        // --verbose makes the non-schema stream-json path emit the result
        // event; for the schema path it turns the output into a noisy event
        // array instead of a clean object with `structured_output`.
        if !has_schema {
            args.push("--verbose".to_string());
        }

        if let Some(schema) = json_schema {
            if !via_wrapper {
                args.push("--json-schema".to_string());
                args.push(schema.to_string());
            }
        }

        if let Some(session_id) = resume_session {
            args.push("--resume".to_string());
            args.push(session_id.to_string());
        }

        let output = self.run_cli(&args, None).await?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut response = if has_schema {
            if via_wrapper {
                parse_claude_p_schema_output(&stdout)?
            } else {
                parse_claude_json_output(&stdout)?
            }
        } else {
            self.parse_claude_stream_output(&stdout)?
        };
        if matches!(self.provider, LlmProvider::ClaudeCli) {
            // `claude -p` reports the equivalent-API cost even when the call
            // is billed against a Max subscription — zero it so usage logs
            // reflect the actual marginal cost.
            response.total_cost_usd = Some(0.0);
        }
        Ok(response)
    }

    async fn generate_anthropic_api(
        &self,
        prompt: &str,
        json_schema: Option<&str>,
    ) -> Result<LlmResponse> {
        let api_key = self.api_key.as_ref()
            .ok_or_else(|| anyhow!("ANTHROPIC API key not set"))?;

        let model = resolve_anthropic_model_alias(&self.model);

        let body = if let Some(schema) = json_schema {
            let input_schema: serde_json::Value = serde_json::from_str(schema)
                .map_err(|e| anyhow!("Invalid JSON schema: {e}"))?;
            serde_json::json!({
                "model": model,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
                "tools": [{
                    "name": "structured_output",
                    "description": "Return the result in the required structured format.",
                    "input_schema": input_schema,
                }],
                "tool_choice": {"type": "tool", "name": "structured_output"},
            })
        } else {
            serde_json::json!({
                "model": model,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
            })
        };

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(self.timeout_secs))
            .connect_timeout(Duration::from_secs(10))
            .build()?;

        let resp = client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| anyhow!("Anthropic API request failed: {e}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body_text = resp.text().await.unwrap_or_default();
            anyhow::bail!("Anthropic API returned {status}: {body_text}");
        }

        let parsed: serde_json::Value = resp.json().await
            .map_err(|e| anyhow!("Failed to parse Anthropic API response: {e}"))?;

        let mut result_text = String::new();
        if let Some(content_arr) = parsed.get("content").and_then(|c| c.as_array()) {
            for block in content_arr {
                let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
                match block_type {
                    "text" => {
                        if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
                            if !result_text.is_empty() { result_text.push('\n'); }
                            result_text.push_str(t);
                        }
                    }
                    "tool_use" => {
                        if let Some(input) = block.get("input") {
                            result_text = serde_json::to_string(input).unwrap_or_default();
                        }
                    }
                    _ => {}
                }
            }
        }

        let cost = parsed.get("usage").and_then(|u| {
            let input_tokens = u.get("input_tokens").and_then(serde_json::Value::as_f64).unwrap_or(0.0);
            let output_tokens = u.get("output_tokens").and_then(serde_json::Value::as_f64).unwrap_or(0.0);
            let cache_read = u.get("cache_read_input_tokens").and_then(serde_json::Value::as_f64).unwrap_or(0.0);
            let cache_create = u.get("cache_creation_input_tokens").and_then(serde_json::Value::as_f64).unwrap_or(0.0);
            estimate_cost_usd(&model, input_tokens, output_tokens, cache_read, cache_create)
        });

        Ok(LlmResponse {
            result: result_text,
            total_cost_usd: cost,
            session_id: parsed.get("id").and_then(|v| v.as_str()).map(std::string::ToString::to_string),
            is_error: false,
        })
    }

    async fn generate_droid(
        &self,
        prompt: &str,
        resume_session: Option<&str>,
    ) -> Result<LlmResponse> {
        let mut args = vec![
            "exec".to_string(),
            "--output-format".to_string(),
            "json".to_string(),
        ];

        if !self.model.is_empty() {
            args.push("--model".to_string());
            args.push(self.model.clone());
        }

        if let Some(session_id) = resume_session {
            args.push("--session-id".to_string());
            args.push(session_id.to_string());
        }

        args.push(prompt.to_string());

        let output = self.run_cli(&args, None).await?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        parse_json_response(&stdout)
    }

    async fn generate_codex(
        &self,
        prompt: &str,
        json_schema: Option<&str>,
        resume_session: Option<&str>,
    ) -> Result<LlmResponse> {
        let output_path = temp_file_path("codex-output", "txt");
        let schema_path = if let Some(schema) = json_schema {
            let path = temp_file_path("codex-schema", "json");
            std::fs::write(&path, schema)?;
            Some(path)
        } else {
            None
        };

        let mut args = vec!["exec".to_string()];
        if resume_session.is_some() {
            args.push("resume".to_string());
        }

        if let Some(schema_path) = &schema_path {
            args.push("--output-schema".to_string());
            args.push(schema_path.to_string_lossy().to_string());
        }

        args.push("--output-last-message".to_string());
        args.push(output_path.to_string_lossy().to_string());

        if !self.model.is_empty() {
            args.push("--model".to_string());
            args.push(self.model.clone());
        }

        if let Some(session_id) = resume_session {
            args.push(session_id.to_string());
        }

        args.push(prompt.to_string());

        let output = self.run_cli(&args, None).await?;
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let result = std::fs::read_to_string(&output_path).unwrap_or(stdout);

        let _ = std::fs::remove_file(&output_path);
        if let Some(schema_path) = schema_path {
            let _ = std::fs::remove_file(schema_path);
        }

        Ok(response_from_text(&result))
    }

    async fn generate_gemini(
        &self,
        prompt: &str,
        json_schema: Option<&str>,
    ) -> Result<LlmResponse> {
        let has_schema = json_schema.is_some();

        let effective_prompt = if let Some(schema) = json_schema {
            format!(
                "{prompt}\n\nRespond with ONLY a JSON object matching this schema. \
                 No prose, no markdown fences, no explanation.\n\nSchema:\n{schema}"
            )
        } else {
            prompt.to_string()
        };

        let mut args = vec![
            "-p".to_string(),
            effective_prompt,
            "--allowed-tools".to_string(),
            String::new(),
            "--allowed-mcp-server-names".to_string(),
            String::new(),
            "-o".to_string(),
            "json".to_string(),
        ];

        if !self.model.is_empty() {
            args.push("-m".to_string());
            args.push(self.model.clone());
        }

        let output = self.run_cli(&args, None).await?;
        let stdout = String::from_utf8_lossy(&output.stdout);

        let wrapper: serde_json::Value = serde_json::from_str(stdout.trim())
            .map_err(|e| anyhow!("Failed to parse gemini -o json output: {e}\nOutput: {}", stdout.trim()))?;

        let response_text = wrapper
            .get("response")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("gemini wrapper missing .response field\nOutput: {}", stdout.trim()))?
            .trim();

        let session_id = wrapper
            .get("session_id")
            .and_then(|v| v.as_str())
            .map(std::string::ToString::to_string);

        if has_schema {
            let stripped = if response_text.starts_with("```") {
                response_text
                    .trim_start_matches("```json")
                    .trim_start_matches("```")
                    .trim_end_matches("```")
                    .trim()
            } else {
                response_text
            };
            serde_json::from_str::<serde_json::Value>(stripped).map_err(|e| {
                anyhow!(
                    "Gemini returned non-JSON inside .response despite schema request: {e}\nInner: {stripped}"
                )
            })?;
            Ok(LlmResponse {
                result: stripped.to_string(),
                total_cost_usd: Some(0.0),
                session_id,
                is_error: false,
            })
        } else {
            Ok(LlmResponse {
                result: response_text.to_string(),
                total_cost_usd: Some(0.0),
                session_id,
                is_error: false,
            })
        }
    }

    async fn generate_kilo(&self, prompt: &str) -> Result<LlmResponse> {
        let args = vec!["--auto".to_string(), prompt.to_string()];

        let output = self.run_cli(&args, None).await?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(response_from_text(&stdout))
    }

    async fn run_cli(
        &self,
        args: &[String],
        stdin_payload: Option<&str>,
    ) -> Result<std::process::Output> {
        let timeout = Duration::from_secs(self.timeout_secs);
        let binary = self.binary.clone();
        let args = args.to_vec();
        let strip_claude_env = matches!(
            self.provider,
            LlmProvider::Claude | LlmProvider::ClaudeCli
        );
        let force_subscription = matches!(self.provider, LlmProvider::ClaudeCli);
        let sandbox_dir = if matches!(self.provider, LlmProvider::Gemini) {
            let base = dirs::cache_dir().unwrap_or_else(std::env::temp_dir);
            let dir = base.join("c0/gemini-sandbox");
            std::fs::create_dir_all(&dir).ok();
            Some(dir)
        } else {
            None
        };

        let result = tokio::time::timeout(timeout, async move {
            let mut cmd = Command::new(&binary);
            cmd.args(&args)
                .env("C0_NO_HOOKS", "1")
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            if let Some(ref dir) = sandbox_dir {
                cmd.current_dir(dir);
            }
            if strip_claude_env {
                for var in [
                    "CLAUDECODE",
                    "CLAUDE_CODE_SESSION_ID",
                    "CLAUDE_CODE_ENTRYPOINT",
                    "CLAUDE_CODE_EXECPATH",
                    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS",
                    "CLAUDE_EFFORT",
                    "CLAUDE_PROJECT_DIR",
                ] {
                    cmd.env_remove(var);
                }
            }
            if force_subscription {
                cmd.env_remove("ANTHROPIC_API_KEY");
                cmd.env_remove("ANTHROPIC_AUTH_TOKEN");
            }
            let mut child = cmd
                .spawn()
                .map_err(|e| anyhow!("Failed to spawn {binary} process: {e}"))?;

            if let Some(mut stdin) = child.stdin.take() {
                if let Some(payload) = stdin_payload {
                    stdin.write_all(payload.as_bytes()).await?;
                }
                stdin.shutdown().await?;
            }

            let output = child
                .wait_with_output()
                .await
                .map_err(|e| anyhow!("Failed to wait for {binary} process: {e}"))?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(anyhow!("{} CLI failed ({}): {}", binary, output.status, stderr));
            }

            Ok(output)
        })
        .await;

        match result {
            Ok(output) => output,
            Err(_) => Err(anyhow!(
                "{} CLI timed out after {} seconds",
                self.binary,
                self.timeout_secs
            )),
        }
    }

    fn parse_claude_stream_output(&self, output: &str) -> Result<LlmResponse> {
        let mut result_content = String::new();
        let mut total_cost: Option<f64> = None;
        let mut session_id: Option<String> = None;
        let mut is_error = false;

        for line in output.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            if let Ok(msg) = serde_json::from_str::<ClaudeStreamMessage>(line) {
                match msg.msg_type.as_str() {
                    "assistant" => {
                        if let Some(content) = msg.content {
                            result_content.push_str(&content);
                        }
                    }
                    "result" => {
                        if let Some(r) = msg.result {
                            result_content = r;
                        }
                        if let Some(cost) = msg.total_cost_usd {
                            total_cost = Some(cost);
                        }
                        if let Some(sid) = msg.session_id {
                            session_id = Some(sid);
                        }
                        if let Some(err) = msg.is_error {
                            is_error = err;
                        }
                    }
                    _ => {}
                }
            }
        }

        if result_content.is_empty() && !output.is_empty() {
            if let Ok(direct) = serde_json::from_str::<LlmResponse>(output.trim()) {
                return Ok(direct);
            }
            result_content = output.to_string();
        }

        Ok(LlmResponse {
            result: result_content,
            total_cost_usd: total_cost,
            session_id,
            is_error,
        })
    }
}

fn resolve_anthropic_model_alias(model: &str) -> String {
    match model.to_lowercase().as_str() {
        "haiku" | "haiku-4-5" | "claude-haiku-4-5" => "claude-haiku-4-5".to_string(),
        "sonnet" | "sonnet-4-6" | "claude-sonnet-4-6" => "claude-sonnet-4-6".to_string(),
        "opus" | "opus-4-7" | "claude-opus-4-7" => "claude-opus-4-7".to_string(),
        _ => model.to_string(),
    }
}

fn estimate_cost_usd(
    model: &str,
    input_tokens: f64,
    output_tokens: f64,
    cache_read_tokens: f64,
    cache_create_tokens: f64,
) -> Option<f64> {
    let (in_rate, out_rate) = match resolve_anthropic_model_alias(model).as_str() {
        "claude-haiku-4-5" => (1.0, 5.0),
        "claude-sonnet-4-6" => (3.0, 15.0),
        "claude-opus-4-7" => (15.0, 75.0),
        _ => return None,
    };
    let cache_read_rate = in_rate * 0.1;
    let cache_create_rate = in_rate * 1.25;
    let cost = (input_tokens * in_rate
        + output_tokens * out_rate
        + cache_read_tokens * cache_read_rate
        + cache_create_tokens * cache_create_rate)
        / 1_000_000.0;
    Some(cost)
}

fn provider_binary(provider: LlmProvider, binaries: &LlmBinaries) -> String {
    match provider {
        LlmProvider::Claude | LlmProvider::ClaudeCli => binaries.claude.clone(),
        LlmProvider::Droid => binaries.droid.clone(),
        LlmProvider::Codex => binaries.codex.clone(),
        LlmProvider::Kilo => binaries.kilo.clone(),
        LlmProvider::Gemini => binaries.gemini.clone(),
        LlmProvider::Ollama => String::new(),
    }
}

fn temp_file_path(prefix: &str, extension: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_nanos();
    path.push(format!("c0-{}-{}-{}.{}", prefix, std::process::id(), nanos, extension));
    path
}

fn parse_json_response(output: &str) -> Result<LlmResponse> {
    let trimmed = output.trim();
    if trimmed.is_empty() {
        return Ok(response_from_text(""));
    }
    if let Ok(parsed) = serde_json::from_str::<LlmResponse>(trimmed) {
        return Ok(parsed);
    }
    Ok(response_from_text(output))
}

/// Parse the output of `claude -p --output-format json --json-schema X`
/// (without `--verbose`). The structured payload lives in the
/// `structured_output` field, not `result`.
fn parse_claude_json_output(output: &str) -> Result<LlmResponse> {
    let trimmed = output.trim();
    if trimmed.is_empty() {
        anyhow::bail!("claude CLI returned empty output");
    }
    let value: serde_json::Value = serde_json::from_str(trimmed)
        .map_err(|e| anyhow!("Failed to parse claude JSON output: {e}\nOutput: {trimmed}"))?;

    let obj = if value.is_array() {
        // --verbose form: an array of events; find the final `result` event.
        value
            .as_array()
            .and_then(|arr| arr.iter().rev().find(|v| v.get("type").and_then(|t| t.as_str()) == Some("result")))
            .cloned()
            .unwrap_or(value)
    } else {
        value
    };

    let result_text = if let Some(structured) = obj.get("structured_output").filter(|v| !v.is_null()) {
        serde_json::to_string(structured).unwrap_or_default()
    } else {
        // Fall back to the `result` string (used by non-schema responses
        // that happen to come through this path).
        obj.get("result").and_then(|r| r.as_str()).unwrap_or("").to_string()
    };

    if result_text.is_empty() {
        anyhow::bail!("claude CLI returned no structured_output or result\nOutput: {trimmed}");
    }

    let is_error = obj.get("is_error").and_then(serde_json::Value::as_bool).unwrap_or(false);
    let session_id = obj.get("session_id").and_then(|v| v.as_str()).map(std::string::ToString::to_string);
    let total_cost_usd = obj.get("total_cost_usd").and_then(serde_json::Value::as_f64);

    Ok(LlmResponse {
        result: result_text,
        total_cost_usd,
        session_id,
        is_error,
    })
}

/// Parse the output of `claude-p --output-format json` when the schema
/// was inlined into the prompt (the wrapper has no `--json-schema`
/// flag, so the model returns plain JSON in the `.result` field
/// instead of a `structured_output` block).
fn parse_claude_p_schema_output(output: &str) -> Result<LlmResponse> {
    let trimmed = output.trim();
    if trimmed.is_empty() {
        anyhow::bail!("claude-p returned empty output");
    }
    let obj: serde_json::Value = serde_json::from_str(trimmed)
        .map_err(|e| anyhow!("Failed to parse claude-p JSON output: {e}\nOutput: {trimmed}"))?;

    let result_str = obj
        .get("result")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("claude-p output missing .result\nOutput: {trimmed}"))?
        .trim();

    let stripped = if result_str.starts_with("```") {
        result_str
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim()
    } else {
        result_str
    };

    serde_json::from_str::<serde_json::Value>(stripped).map_err(|e| {
        anyhow!("claude-p .result is not valid JSON despite inline schema: {e}\nInner: {stripped}")
    })?;

    Ok(LlmResponse {
        result: stripped.to_string(),
        total_cost_usd: obj.get("total_cost_usd").and_then(serde_json::Value::as_f64),
        session_id: obj.get("session_id").and_then(|v| v.as_str()).map(String::from),
        is_error: obj.get("is_error").and_then(serde_json::Value::as_bool).unwrap_or(false),
    })
}

fn response_from_text(output: &str) -> LlmResponse {
    LlmResponse {
        result: output.trim().to_string(),
        total_cost_usd: None,
        session_id: None,
        is_error: false,
    }
}

pub fn log_usage(task: &str, model: &str, cost_usd: f64) {
    if let Some(home) = dirs::home_dir() {
        let usage_path = home.join(".c0/usage.jsonl");
        let entry = serde_json::json!({
            "ts": chrono::Utc::now().to_rfc3339(),
            "kind": "llm",
            "task": task,
            "model": model,
            "cost_usd": cost_usd,
        });
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(usage_path)
        {
            let _ = writeln!(f, "{entry}");
        }
    }
}

pub fn log_cmd(cmd: &str, ns: &str, latency_ms: u64) {
    if let Some(home) = dirs::home_dir() {
        let usage_path = home.join(".c0/usage.jsonl");
        let entry = serde_json::json!({
            "ts": chrono::Utc::now().to_rfc3339(),
            "kind": "cmd",
            "cmd": cmd,
            "ns": ns,
            "latency_ms": latency_ms,
        });
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(usage_path)
        {
            let _ = writeln!(f, "{entry}");
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConceptExtractionResult {
    pub known: Vec<String>,
    pub unknown: Vec<String>,
    pub queued: bool,
}

#[cfg(feature = "sessions")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExtractedConcept {
    pub name: String,
    pub description: String,
}

#[cfg(feature = "sessions")]
const SESSION_CONCEPT_PROMPT: &str = r#"You are a knowledge curator. Extract distinct technology concepts, libraries, frameworks, patterns, methodologies, or domain ideas discussed in this Claude Code session.

For each concept return:
- name: lowercase kebab-case identifier (e.g. "rust-async", "neo4j-vector-index", "systemd-timer")
- description: one-sentence summary of what was discussed about it (max 200 chars)

Rules:
- Extract up to {max} distinct concepts, ranked by relevance
- Names must be lowercase kebab-case, alphanumeric + hyphens only
- Skip generic terms: database, api, code, app, file, function, variable, system, hook, type, stuff, things, service
- Skip pronouns, articles, and one-off local names (file paths, variable names)
- Skip people's names
- Each concept should be something you'd actually want to look up later

Return ONLY valid JSON in this exact format (no prose, no markdown):
{"concepts": [{"name": "...", "description": "..."}]}

If no extractable concepts, return: {"concepts": []}

Session text:
{text}"#;

#[cfg(feature = "sessions")]
const SESSION_CONCEPT_SCHEMA: &str = r#"{"type":"object","properties":{"concepts":{"type":"array","items":{"type":"object","properties":{"name":{"type":"string"},"description":{"type":"string"}},"required":["name","description"],"additionalProperties":false}}},"required":["concepts"],"additionalProperties":false}"#;

const CONCEPT_EXTRACTION_PROMPT: &str = r#"Extract 1-3 specific technology concepts from this prompt that would benefit from knowledge lookup.

Rules:
- ONLY extract concrete, named technologies, APIs, or tools (e.g., "Shopify", "Redis", "webhooks", "Neo4j")
- DO NOT extract generic terms like "database", "api", "code", "app", "stuff", "things"
- DO NOT extract words from these instructions
- If the prompt contains NO specific technology concepts, respond with exactly: NONE
- Normalize to lowercase, kebab-case

Prompt: "{prompt}"

Output (concepts separated by commas, or NONE):"#;

impl LlmClient {
    pub async fn extract_concepts(
        &self,
        prompt: &str,
        _namespace: &str,
        max_concepts: usize,
    ) -> Result<Vec<String>> {
        let formatted_prompt = CONCEPT_EXTRACTION_PROMPT.replace("{prompt}", prompt);

        let response = self.generate(&formatted_prompt, None).await?;

        let result = response.result.trim().to_lowercase();

        if result == "none" || result.is_empty() {
            if let Some(cost) = response.total_cost_usd {
                log_usage("concept-extraction", &self.model, cost);
            }
            return Ok(Vec::new());
        }

        let bad_patterns = [
            "type", "system", "hook", "subtype", "empty", "none",
            "database", "api", "code", "app", "stuff", "things",
            "service", "domain", "specific", "terms", "concepts",
        ];

        let concepts: Vec<String> = response
            .result
            .trim()
            .split(',')
            .map(|s| s.trim().to_lowercase().replace(' ', "-"))
            .filter(|s| {
                !s.is_empty()
                    && s.len() > 2
                    && s.len() < 50
                    && s.chars().all(|c| c.is_alphanumeric() || c == '-')
                    && !bad_patterns.iter().any(|p| s.contains(p))
            })
            .take(max_concepts)
            .collect();

        if let Some(cost) = response.total_cost_usd {
            log_usage("concept-extraction", &self.model, cost);
        }

        Ok(concepts)
    }

    #[cfg(feature = "sessions")]
    pub async fn extract_session_concepts(
        &self,
        text: &str,
        max_concepts: usize,
    ) -> Result<Vec<ExtractedConcept>> {
        let prompt = SESSION_CONCEPT_PROMPT
            .replace("{max}", &max_concepts.to_string())
            .replace("{text}", text);

        let response = self.generate(&prompt, Some(SESSION_CONCEPT_SCHEMA)).await?;

        if let Some(cost) = response.total_cost_usd {
            log_usage("session-enrichment", &self.model, cost);
        }

        parse_session_concepts(&response.result, max_concepts)
    }
}

#[cfg(feature = "sessions")]
#[derive(Deserialize)]
struct SessionConceptsResponse {
    #[serde(default)]
    concepts: Vec<ExtractedConcept>,
}

#[cfg(feature = "sessions")]
fn parse_session_concepts(raw: &str, max: usize) -> Result<Vec<ExtractedConcept>> {
    let trimmed = raw.trim();
    let cleaned = trimmed
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    let parsed: SessionConceptsResponse = serde_json::from_str(cleaned)
        .map_err(|e| anyhow!("Failed to parse session concepts JSON: {e}\nRaw: {cleaned}"))?;

    let bad_patterns = [
        "type", "system", "hook", "subtype", "empty", "none",
        "database", "api", "code", "app", "stuff", "things",
        "service", "domain", "generic", "term", "concept",
    ];

    let out: Vec<ExtractedConcept> = parsed
        .concepts
        .into_iter()
        .filter_map(|c| {
            let name = c.name.trim().to_lowercase().replace(' ', "-");
            if name.is_empty() || name.len() < 3 || name.len() > 60 {
                return None;
            }
            if !name.chars().all(|ch| ch.is_alphanumeric() || ch == '-') {
                return None;
            }
            if bad_patterns.iter().any(|p| name == *p) {
                return None;
            }
            let description = c.description.trim().to_string();
            if description.is_empty() {
                return None;
            }
            let description = if description.len() > 220 {
                let mut end = 220;
                while !description.is_char_boundary(end) && end > 0 { end -= 1; }
                format!("{}...", &description[..end])
            } else {
                description
            };
            Some(ExtractedConcept { name, description })
        })
        .take(max)
        .collect();

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_stream_output() {
        let binaries = LlmBinaries::default();
        let client = LlmClient {
            provider: LlmProvider::Claude,
            model: "haiku".to_string(),
            timeout_secs: 60,
            max_budget_usd: None,
            binary: binaries.claude,
            api_key: None,
            ollama_host: None,
        };

        let output = r#"{"type":"assistant","content":"Hello"}
{"type":"result","result":"Hello world","total_cost_usd":0.001,"session_id":"abc123"}"#;

        let response = client.parse_claude_stream_output(output).unwrap();
        assert_eq!(response.result, "Hello world");
        assert_eq!(response.total_cost_usd, Some(0.001));
        assert_eq!(response.session_id, Some("abc123".to_string()));
    }

    #[test]
    fn test_client_creation() {
        let config = ClaudeConfig::default();
        let client = LlmClient::from_config_with_timeout(&config, "sonnet", 120);
        assert_eq!(client.model, "sonnet");
        assert_eq!(client.timeout_secs, 120);
        assert_eq!(client.max_budget_usd, config.max_budget_usd);
    }
}
