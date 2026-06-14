use anyhow::{Result, anyhow};
use regex::Regex;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::LazyLock;
use std::time::{Duration, SystemTime};

const DEFAULT_CACHE_TTL_SECS: u64 = 300;
const DEFAULT_HTTP_TIMEOUT_SECS: u64 = 30;
const DEFAULT_CONNECT_TIMEOUT_SECS: u64 = 10;

fn http_client_builder() -> reqwest::ClientBuilder {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(DEFAULT_HTTP_TIMEOUT_SECS))
        .connect_timeout(Duration::from_secs(DEFAULT_CONNECT_TIMEOUT_SECS))
        .redirect(reqwest::redirect::Policy::limited(5))
}

pub struct FetchResult {
    pub content: String,
    pub source_type: String,
    pub url: String,
    pub cached: bool,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct CacheEntry {
    content: String,
    source_type: String,
    url: String,
    timestamp: u64,
}

fn get_cache_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".c0/cache/fetch")
}

fn url_to_cache_key(url: &str) -> String {
    let mut hasher = DefaultHasher::new();
    url.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

fn get_cache_path(url: &str) -> PathBuf {
    get_cache_dir().join(format!("{}.json", url_to_cache_key(url)))
}

fn read_cache(url: &str, ttl_secs: u64) -> Option<FetchResult> {
    let cache_path = get_cache_path(url);
    let content = std::fs::read_to_string(&cache_path).ok()?;
    let entry: CacheEntry = serde_json::from_str(&content).ok()?;

    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .ok()?
        .as_secs();

    if now - entry.timestamp > ttl_secs {
        return None;
    }

    Some(FetchResult {
        content: entry.content,
        source_type: entry.source_type,
        url: entry.url,
        cached: true,
    })
}

fn write_cache(result: &FetchResult) -> Result<()> {
    let cache_dir = get_cache_dir();
    std::fs::create_dir_all(&cache_dir)?;

    let entry = CacheEntry {
        content: result.content.clone(),
        source_type: result.source_type.clone(),
        url: result.url.clone(),
        timestamp: SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)?
            .as_secs(),
    };

    let cache_path = get_cache_path(&result.url);
    std::fs::write(cache_path, serde_json::to_string(&entry)?)?;
    Ok(())
}

pub async fn fetch_url(url: &str, source_type: &str) -> Result<FetchResult> {
    fetch_url_with_cache(url, source_type, DEFAULT_CACHE_TTL_SECS).await
}

pub async fn fetch_url_no_cache(url: &str, source_type: &str) -> Result<FetchResult> {
    fetch_url_with_cache(url, source_type, 0).await
}

pub async fn fetch_url_with_cache(url: &str, source_type: &str, ttl_secs: u64) -> Result<FetchResult> {
    if ttl_secs > 0
        && let Some(cached) = read_cache(url, ttl_secs) {
            return Ok(cached);
        }

    let mut result = match source_type {
        "gdoc" => fetch_google_doc(url).await?,
        "gsheet" => fetch_google_sheet(url).await?,
        "raw" => fetch_raw_url(url).await?,
        _ => fetch_generic_url(url).await?,
    };
    result.cached = false;

    if ttl_secs > 0 {
        let _ = write_cache(&result);
    }

    Ok(result)
}

pub fn clear_cache() -> Result<usize> {
    let cache_dir = get_cache_dir();
    if !cache_dir.exists() {
        return Ok(0);
    }

    let mut count = 0;
    for entry in std::fs::read_dir(&cache_dir)? {
        if let Ok(entry) = entry
            && entry.path().extension().is_some_and(|e| e == "json") {
                std::fs::remove_file(entry.path())?;
                count += 1;
            }
    }
    Ok(count)
}

pub fn clear_expired_cache(ttl_secs: u64) -> Result<usize> {
    let cache_dir = get_cache_dir();
    if !cache_dir.exists() {
        return Ok(0);
    }

    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)?
        .as_secs();

    let mut count = 0;
    for entry in (std::fs::read_dir(&cache_dir)?).flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "json")
            && let Ok(content) = std::fs::read_to_string(&path)
                && let Ok(cache_entry) = serde_json::from_str::<CacheEntry>(&content)
                    && now - cache_entry.timestamp > ttl_secs {
                        std::fs::remove_file(&path)?;
                        count += 1;
                    }
    }
    Ok(count)
}

pub fn detect_source_type(url: &str) -> &'static str {
    if url.contains("docs.google.com/document") {
        "gdoc"
    } else if url.contains("docs.google.com/spreadsheets") {
        "gsheet"
    } else if url.contains("raw.githubusercontent.com") || url.ends_with(".txt") || url.ends_with(".md") {
        "raw"
    } else {
        "url"
    }
}

fn extract_google_doc_id(url: &str) -> Option<String> {
    let re = Regex::new(r"/d/([a-zA-Z0-9_-]+)").ok()?;
    re.captures(url).map(|c| c[1].to_string())
}

async fn fetch_google_doc(url: &str) -> Result<FetchResult> {
    let doc_id = extract_google_doc_id(url)
        .ok_or_else(|| anyhow!("Could not extract Google Doc ID from URL: {url}"))?;

    let export_url = format!(
        "https://docs.google.com/document/d/{doc_id}/export?format=txt"
    );

    let client = http_client_builder().build()?;

    let resp = client.get(&export_url).send().await?;

    if !resp.status().is_success() {
        return Err(anyhow!(
            "Failed to fetch Google Doc (status {}). Make sure the doc is shared with 'Anyone with the link'.",
            resp.status()
        ));
    }

    let content = resp.text().await?;

    Ok(FetchResult {
        content,
        source_type: "gdoc".to_string(),
        url: url.to_string(),
        cached: false,
    })
}

async fn fetch_google_sheet(url: &str) -> Result<FetchResult> {
    let doc_id = extract_google_doc_id(url)
        .ok_or_else(|| anyhow!("Could not extract Google Sheet ID from URL: {url}"))?;

    let export_url = format!(
        "https://docs.google.com/spreadsheets/d/{doc_id}/export?format=csv"
    );

    let client = http_client_builder().build()?;

    let resp = client.get(&export_url).send().await?;

    if !resp.status().is_success() {
        return Err(anyhow!(
            "Failed to fetch Google Sheet (status {}). Make sure the sheet is shared with 'Anyone with the link'.",
            resp.status()
        ));
    }

    let csv_content = resp.text().await?;
    let content = csv_to_markdown_table(&csv_content);

    Ok(FetchResult {
        content,
        source_type: "gsheet".to_string(),
        url: url.to_string(),
        cached: false,
    })
}

fn csv_to_markdown_table(csv: &str) -> String {
    let mut rows: Vec<Vec<String>> = Vec::new();

    for line in csv.lines() {
        if line.trim().is_empty() {
            continue;
        }
        rows.push(parse_csv_line(line));
    }

    if rows.is_empty() {
        return String::new();
    }

    let col_count = rows.iter().map(std::vec::Vec::len).max().unwrap_or(0);
    if col_count == 0 {
        return String::new();
    }

    let mut col_widths = vec![0usize; col_count];
    for row in &rows {
        for (i, cell) in row.iter().enumerate() {
            col_widths[i] = col_widths[i].max(cell.len());
        }
    }

    let mut output = String::new();

    if let Some(header) = rows.first() {
        output.push('|');
        for (i, cell) in header.iter().enumerate() {
            let width = col_widths.get(i).copied().unwrap_or(0);
            output.push_str(&format!(" {cell:width$} |"));
        }
        for i in header.len()..col_count {
            let width = col_widths.get(i).copied().unwrap_or(0);
            output.push_str(&format!(" {:width$} |", "", width = width));
        }
        output.push('\n');

        output.push('|');
        for &width in &col_widths {
            output.push_str(&format!("-{}-|", "-".repeat(width)));
        }
        output.push('\n');
    }

    for row in rows.iter().skip(1) {
        output.push('|');
        for (i, cell) in row.iter().enumerate() {
            let width = col_widths.get(i).copied().unwrap_or(0);
            output.push_str(&format!(" {cell:width$} |"));
        }
        for i in row.len()..col_count {
            let width = col_widths.get(i).copied().unwrap_or(0);
            output.push_str(&format!(" {:width$} |", "", width = width));
        }
        output.push('\n');
    }

    output
}

fn parse_csv_line(line: &str) -> Vec<String> {
    let mut cells = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            '"' if !in_quotes => {
                in_quotes = true;
            }
            '"' if in_quotes => {
                if chars.peek() == Some(&'"') {
                    current.push('"');
                    chars.next();
                } else {
                    in_quotes = false;
                }
            }
            ',' if !in_quotes => {
                cells.push(current.trim().to_string());
                current = String::new();
            }
            _ => {
                current.push(c);
            }
        }
    }
    cells.push(current.trim().to_string());
    cells
}

async fn fetch_raw_url(url: &str) -> Result<FetchResult> {
    let client = http_client_builder().build()?;
    let resp = client.get(url).send().await?;

    if !resp.status().is_success() {
        return Err(anyhow!("Failed to fetch URL (status {}): {}", resp.status(), url));
    }

    let content = resp.text().await?;

    Ok(FetchResult {
        content,
        source_type: "raw".to_string(),
        url: url.to_string(),
        cached: false,
    })
}

async fn fetch_generic_url(url: &str) -> Result<FetchResult> {
    let client = http_client_builder()
        .user_agent("c0-memory-agent/1.0")
        .build()?;

    let resp = client.get(url).send().await?;

    if !resp.status().is_success() {
        return Err(anyhow!("Failed to fetch URL (status {}): {}", resp.status(), url));
    }

    let is_html = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.contains("text/html"));

    let text = resp.text().await?;

    let content = if is_html {
        extract_text_from_html(&text)
    } else {
        text
    };

    Ok(FetchResult {
        content,
        source_type: "url".to_string(),
        url: url.to_string(),
        cached: false,
    })
}

fn extract_text_from_html(html: &str) -> String {
    static SCRIPT_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?is)<script[^>]*>.*?</script>").expect("valid regex"));
    static STYLE_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?is)<style[^>]*>.*?</style>").expect("valid regex"));
    static TAG_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<[^>]+>").expect("valid regex"));
    static WHITESPACE_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s+").expect("valid regex"));

    let script_re = &*SCRIPT_RE;
    let style_re = &*STYLE_RE;
    let tag_re = &*TAG_RE;
    let whitespace_re = &*WHITESPACE_RE;

    let text = script_re.replace_all(html, "");
    let text = style_re.replace_all(&text, "");
    let text = tag_re.replace_all(&text, " ");
    let text = whitespace_re.replace_all(&text, " ");

    html_escape_decode(&text).trim().to_string()
}

fn html_escape_decode(text: &str) -> String {
    text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
}

pub fn truncate_for_embedding(content: &str, max_chars: usize) -> String {
    if content.len() <= max_chars {
        content.to_string()
    } else {
        let truncated: String = content.chars().take(max_chars).collect();
        if let Some(last_period) = truncated.rfind(". ") {
            truncated[..=last_period].to_string()
        } else if let Some(last_newline) = truncated.rfind('\n') {
            truncated[..last_newline].to_string()
        } else {
            truncated
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_source_type() {
        assert_eq!(detect_source_type("https://docs.google.com/document/d/abc123/edit"), "gdoc");
        assert_eq!(detect_source_type("https://docs.google.com/spreadsheets/d/abc123/edit"), "gsheet");
        assert_eq!(detect_source_type("https://raw.githubusercontent.com/user/repo/main/file.md"), "raw");
        assert_eq!(detect_source_type("https://example.com/page.txt"), "raw");
        assert_eq!(detect_source_type("https://example.com/page"), "url");
    }

    #[test]
    fn test_extract_google_doc_id() {
        assert_eq!(
            extract_google_doc_id("https://docs.google.com/document/d/1abc123XYZ/edit"),
            Some("1abc123XYZ".to_string())
        );
        assert_eq!(
            extract_google_doc_id("https://docs.google.com/document/d/1abc-123_XYZ/edit#heading=h.123"),
            Some("1abc-123_XYZ".to_string())
        );
    }

    #[test]
    fn test_extract_text_from_html() {
        let html = r#"<html><head><script>var x = 1;</script></head>
                      <body><h1>Title</h1><p>Hello &amp; world!</p></body></html>"#;
        let text = extract_text_from_html(html);
        assert!(text.contains("Title"));
        assert!(text.contains("Hello & world!"));
        assert!(!text.contains("var x"));
    }

    #[test]
    fn test_truncate_for_embedding() {
        let content = "First sentence. Second sentence. Third sentence.";
        let truncated = truncate_for_embedding(content, 25);
        assert_eq!(truncated, "First sentence.");
    }

    #[test]
    fn test_csv_to_markdown_table() {
        let csv = "Name,Age,City\nAlice,30,NYC\nBob,25,LA";
        let table = csv_to_markdown_table(csv);
        assert!(table.contains("| Name"));
        assert!(table.contains("| Alice"));
        assert!(table.contains("|-----"));
    }

    #[test]
    fn test_parse_csv_line_with_quotes() {
        let line = r#"Name,"City, State",Age"#;
        let cells = parse_csv_line(line);
        assert_eq!(cells, vec!["Name", "City, State", "Age"]);
    }

    #[test]
    fn test_parse_csv_line_with_escaped_quotes() {
        let line = r#"Name,"She said ""Hello""",Age"#;
        let cells = parse_csv_line(line);
        assert_eq!(cells, vec!["Name", r#"She said "Hello""#, "Age"]);
    }
}
