use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::env;
use std::path::Path;
use std::time::Duration;

use crate::claude::{LlmClient, log_usage};
use crate::config::SemanticConfig;

const DEFAULT_EXTRACT_MODEL: &str = "qwen2.5:14b";
const MAX_TRANSCRIPT_CHARS: usize = 20_000;
const CLAUDE_MAX_TRANSCRIPT_CHARS: usize = 100_000;

#[derive(Debug, Clone)]
pub struct ExtractConfig {
    pub host: String,
    pub model: String,
}

pub fn extract_config() -> ExtractConfig {
    let host = env::var("C0_OLLAMA_HOST")
        .or_else(|_| env::var("OLLAMA_HOST"))
        .unwrap_or_else(|_| crate::config::SemanticConfig::load().ollama_host);

    let model = env::var("C0_EXTRACT_MODEL")
        .or_else(|_| env::var("EXTRACT_MODEL"))
        .unwrap_or_else(|_| DEFAULT_EXTRACT_MODEL.to_string());

    ExtractConfig { host, model }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingMetadata {
    pub title: String,
    pub date: String,
    pub duration_minutes: Option<u32>,
    pub attendees: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Topic {
    pub name: String,
    pub normalized_name: String,
    pub context: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionItem {
    pub action: String,
    pub owner: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quote {
    pub text: String,
    pub speaker: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptExtraction {
    pub metadata: MeetingMetadata,
    pub topics: Vec<Topic>,
    pub decisions: Vec<String>,
    pub action_items: Vec<ActionItem>,
    pub technical_details: Vec<String>,
    pub quotes: Vec<Quote>,
    pub summary: String,
}

#[derive(Deserialize)]
struct OllamaResponse {
    response: String,
}

fn parse_transcript_file(path: &Path) -> Result<String> {
    let content = std::fs::read_to_string(path)?;

    if path.extension().is_some_and(|ext| ext == "json") {
        #[derive(Deserialize)]
        struct TldvMessage {
            #[serde(rename = "type")]
            msg_type: Option<String>,
            text: Option<String>,
        }

        if let Ok(messages) = serde_json::from_str::<Vec<TldvMessage>>(&content) {
            let transcript_text: Vec<String> = messages
                .iter()
                .filter_map(|m| m.text.as_ref())
                .cloned()
                .collect();
            return Ok(transcript_text.join("\n\n"));
        }

        if let Ok(arr) = serde_json::from_str::<Vec<String>>(&content) {
            return Ok(arr.join("\n\n"));
        }
    }

    Ok(content)
}

fn extract_date_from_filename(path: &Path) -> String {
    let filename = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    let parts: Vec<&str> = filename.split('-').collect();
    if parts.len() >= 3
        && let (Ok(_y), Ok(_m), Ok(_d)) = (
            parts[0].parse::<u32>(),
            parts[1].parse::<u32>(),
            parts[2].parse::<u32>(),
        )
    {
        return format!("{}-{}-{}", parts[0], parts[1], parts[2]);
    }
    "unknown".to_string()
}

fn extract_title_from_filename(path: &Path) -> String {
    let filename = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("meeting");

    let parts: Vec<&str> = filename.split('-').collect();
    if parts.len() > 3 {
        let title_parts: Vec<&str> = parts[3..]
            .iter()
            .filter(|p| **p != "raw")
            .copied()
            .collect();
        if !title_parts.is_empty() {
            return title_parts
                .join(" ")
                .replace('_', " ")
                .split_whitespace()
                .map(|w| {
                    let mut c = w.chars();
                    match c.next() {
                        None => String::new(),
                        Some(f) => f.to_uppercase().chain(c).collect(),
                    }
                })
                .collect::<Vec<_>>()
                .join(" ");
        }
    }
    "Meeting".to_string()
}

const EXTRACTION_SCHEMA: &str = r#"{"type":"object","properties":{"attendees":{"type":"array","items":{"type":"string"}},"duration_minutes":{"type":["integer","null"]},"topics":{"type":"array","items":{"type":"object","properties":{"name":{"type":"string"},"normalized_name":{"type":"string"},"context":{"type":"string"}},"required":["name","normalized_name","context"]}},"decisions":{"type":"array","items":{"type":"string"}},"action_items":{"type":"array","items":{"type":"object","properties":{"action":{"type":"string"},"owner":{"type":["string","null"]}},"required":["action"]}},"technical_details":{"type":"array","items":{"type":"string"}},"quotes":{"type":"array","items":{"type":"object","properties":{"text":{"type":"string"},"speaker":{"type":"string"}},"required":["text","speaker"]}},"summary":{"type":"string"}},"required":["attendees","topics","decisions","action_items","technical_details","quotes","summary"]}"#;

fn build_extraction_prompt(transcript: &str) -> String {
    format!(
        r#"You are analyzing a meeting transcript. Extract structured information in JSON format.

TRANSCRIPT:
{transcript}

INSTRUCTIONS:
Extract the following and return as valid JSON (no markdown, just JSON):

{{
  "attendees": ["list of speaker names found in transcript"],
  "duration_minutes": null,
  "topics": [
    {{
      "name": "Human readable topic name",
      "normalized_name": "kebab-case-for-graph",
      "context": "Brief context about what was discussed"
    }}
  ],
  "decisions": ["List of decisions made during the meeting"],
  "action_items": [
    {{
      "action": "What needs to be done",
      "owner": "Who is responsible (or null)"
    }}
  ],
  "technical_details": ["Technical systems, integrations, data flows mentioned"],
  "quotes": [
    {{
      "text": "Notable quote",
      "speaker": "Who said it"
    }}
  ],
  "summary": "2-3 sentence summary of the meeting"
}}

Focus on:
- Key topics discussed (normalize names like "field-mappings", "waitlist-logic", "payment-plans")
- Decisions that were made
- Action items with owners when clear
- Technical details about systems and integrations
- Notable quotes that capture key insights

Return ONLY the JSON object, no other text."#
    )
}

#[derive(Deserialize)]
struct ExtractedData {
    attendees: Vec<String>,
    duration_minutes: Option<u32>,
    topics: Vec<Topic>,
    decisions: Vec<String>,
    action_items: Vec<ActionItem>,
    technical_details: Vec<String>,
    quotes: Vec<Quote>,
    summary: String,
}

async fn extract_with_ollama(
    transcript: &str,
    ollama_config: &ExtractConfig,
) -> Result<ExtractedData> {
    let prompt = build_extraction_prompt(transcript);

    eprintln!("   Sending to Ollama...");
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(1800))
        .connect_timeout(Duration::from_secs(30))
        .tcp_keepalive(Duration::from_secs(60))
        .build()?;

    let url = format!("{}/api/generate", ollama_config.host.trim_end_matches('/'));
    eprintln!("   URL: {url}");
    eprintln!("   Prompt length: {} chars", prompt.len());

    let body = serde_json::json!({
        "model": ollama_config.model,
        "prompt": prompt,
        "stream": false,
        "options": {
            "num_ctx": 8192
        }
    });
    eprintln!(
        "   Request body size: {} bytes",
        serde_json::to_string(&body)?.len()
    );

    let response = client.post(&url).json(&body).send().await.map_err(|e| {
        eprintln!("   Error details: {e:?}");
        if e.is_timeout() {
            anyhow!("Request timed out after 600 seconds")
        } else if e.is_connect() {
            anyhow!("Failed to connect to Ollama: connection error")
        } else {
            anyhow!(
                "Failed to send request to Ollama at {}: {}",
                ollama_config.host,
                e
            )
        }
    })?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(anyhow!("Ollama returned error {status}: {body}"));
    }

    let ollama_resp: OllamaResponse = response.json().await?;
    let json_str = ollama_resp.response.trim();

    let json_str = if json_str.starts_with("```") {
        json_str
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim()
    } else {
        json_str
    };

    let extracted: ExtractedData = serde_json::from_str(json_str).map_err(|e| {
        anyhow!("Failed to parse LLM response as JSON: {e}\nResponse was: {json_str}")
    })?;

    Ok(extracted)
}

async fn extract_with_llm(
    transcript: &str,
    semantic_config: &SemanticConfig,
) -> Result<ExtractedData> {
    let prompt = build_extraction_prompt(transcript);

    let llm = LlmClient::for_task(semantic_config, "extraction", 600);

    eprintln!("   Sending to LLM CLI...");
    eprintln!("   Provider: {}", llm.provider_name());
    eprintln!("   Model: {}", llm.model);
    eprintln!("   Prompt length: {} chars", prompt.len());

    let response = llm.generate(&prompt, Some(EXTRACTION_SCHEMA)).await?;

    if let Some(cost) = response.total_cost_usd {
        log_usage("extraction", &llm.model, cost);
        eprintln!("   Cost: ${cost:.4}");
    }

    let json_str = response.result.trim();
    let json_str = if json_str.starts_with("```") {
        json_str
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim()
    } else {
        json_str
    };

    let extracted: ExtractedData = serde_json::from_str(json_str).map_err(|e| {
        anyhow!("Failed to parse LLM response as JSON: {e}\nResponse was: {json_str}")
    })?;

    Ok(extracted)
}

pub async fn extract_transcript(input_path: &Path) -> Result<TranscriptExtraction> {
    let transcript = parse_transcript_file(input_path)?;
    let date = extract_date_from_filename(input_path);
    let title = extract_title_from_filename(input_path);

    let semantic_config = SemanticConfig::load();
    let extraction_provider = semantic_config.claude.provider_for("extraction");
    let use_ollama_direct = extraction_provider == "ollama";

    let max_chars = if use_ollama_direct {
        MAX_TRANSCRIPT_CHARS
    } else {
        CLAUDE_MAX_TRANSCRIPT_CHARS
    };

    let truncated = if transcript.len() > max_chars {
        eprintln!(
            "   Truncating transcript from {} to {} chars",
            transcript.len(),
            max_chars
        );
        transcript.chars().take(max_chars).collect::<String>()
    } else {
        transcript.clone()
    };
    eprintln!("   Transcript length: {} chars", truncated.len());

    let extracted = if use_ollama_direct {
        let ollama_config = extract_config();
        extract_with_ollama(&truncated, &ollama_config).await?
    } else {
        extract_with_llm(&truncated, &semantic_config).await?
    };

    Ok(TranscriptExtraction {
        metadata: MeetingMetadata {
            title,
            date,
            duration_minutes: extracted.duration_minutes,
            attendees: extracted.attendees,
        },
        topics: extracted.topics,
        decisions: extracted.decisions,
        action_items: extracted.action_items,
        technical_details: extracted.technical_details,
        quotes: extracted.quotes,
        summary: extracted.summary,
    })
}

pub fn generate_summary_markdown(extraction: &TranscriptExtraction, namespace: &str) -> String {
    let mut md = String::new();

    md.push_str(&format!(
        "# {namespace} Meeting: {}\n\n",
        extraction.metadata.title
    ));
    md.push_str(&format!("> Date: {}", extraction.metadata.date));
    if let Some(duration) = extraction.metadata.duration_minutes {
        md.push_str(&format!(" | Duration: {duration} mins"));
    }
    md.push('\n');
    if !extraction.metadata.attendees.is_empty() {
        md.push_str(&format!(
            "> Attendees: {}\n",
            extraction.metadata.attendees.join(", ")
        ));
    }
    md.push('\n');

    md.push_str("## Summary\n\n");
    md.push_str(&extraction.summary);
    md.push_str("\n\n");

    if !extraction.topics.is_empty() {
        md.push_str("## Key Topics\n\n");
        for topic in &extraction.topics {
            md.push_str(&format!(
                "- **{}** (`{}`): {}\n",
                topic.name, topic.normalized_name, topic.context
            ));
        }
        md.push('\n');
    }

    if !extraction.decisions.is_empty() {
        md.push_str("## Decisions Made\n\n");
        for decision in &extraction.decisions {
            md.push_str(&format!("- {decision}\n"));
        }
        md.push('\n');
    }

    if !extraction.action_items.is_empty() {
        md.push_str("## Action Items\n\n");
        for item in &extraction.action_items {
            let owner = item
                .owner
                .as_ref()
                .map(|o| format!(" ({o})"))
                .unwrap_or_default();
            md.push_str(&format!("- [ ] {}{}\n", item.action, owner));
        }
        md.push('\n');
    }

    if !extraction.technical_details.is_empty() {
        md.push_str("## Technical Details\n\n");
        for detail in &extraction.technical_details {
            md.push_str(&format!("- {detail}\n"));
        }
        md.push('\n');
    }

    if !extraction.quotes.is_empty() {
        md.push_str("## Notable Quotes\n\n");
        for quote in &extraction.quotes {
            md.push_str(&format!("> \"{}\" - {}\n\n", quote.text, quote.speaker));
        }
    }

    md
}

pub fn get_patch_name(date: &str, title: &str, namespace: &str) -> String {
    let slug = title
        .to_lowercase()
        .replace(' ', "-")
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '-')
        .collect::<String>();
    format!("{namespace}-transcript-{date}-{slug}")
}
