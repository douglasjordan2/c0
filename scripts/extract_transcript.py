#!/usr/bin/env python3
"""
Extract meeting information from transcripts using Ollama.
Usage: python extract_transcript.py <input_file> [--index]
"""

import json
import sys
import os
import re
import subprocess
from pathlib import Path
import requests

OLLAMA_HOST = os.getenv("C0_OLLAMA_HOST") or os.getenv("OLLAMA_HOST") or "http://localhost:11434"
MODEL = os.getenv("C0_EXTRACT_MODEL") or os.getenv("EXTRACT_MODEL") or "hermes3:8b"  # Faster than qwen2.5:14b
MAX_CHARS = 12000

def parse_transcript(path: Path) -> str:
    content = path.read_text()

    if path.suffix == '.json':
        try:
            data = json.loads(content)
            if isinstance(data, list) and data and 'text' in data[0]:
                return data[0]['text'][:MAX_CHARS]
            elif isinstance(data, list):
                return '\n\n'.join(str(item) for item in data)[:MAX_CHARS]
        except json.JSONDecodeError:
            pass

    return content[:MAX_CHARS]

def extract_date_from_filename(path: Path) -> str:
    name = path.stem
    parts = name.split('-')
    if len(parts) >= 3:
        try:
            int(parts[0])
            int(parts[1])
            int(parts[2])
            return f"{parts[0]}-{parts[1]}-{parts[2]}"
        except ValueError:
            pass
    return "unknown"

def extract_title_from_filename(path: Path) -> str:
    name = path.stem
    parts = name.split('-')
    if len(parts) > 3:
        title_parts = [p for p in parts[3:] if p != 'raw']
        if title_parts:
            return ' '.join(title_parts).replace('_', ' ').title()
    return "Meeting"

def call_ollama(prompt: str) -> str:
    print(f"   Sending to Ollama ({len(prompt)} chars)...", file=sys.stderr)

    response = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"num_ctx": 8192}
        },
        timeout=1200
    )
    response.raise_for_status()
    return response.json()['response']

def extract_info(transcript: str) -> dict:
    prompt = f'''You are analyzing a meeting transcript. Extract structured information in JSON format.

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

Return ONLY the JSON object, no other text.'''

    response = call_ollama(prompt)

    json_str = response.strip()
    if json_str.startswith('```'):
        json_str = re.sub(r'^```json?\n?', '', json_str)
        json_str = re.sub(r'\n?```$', '', json_str)

    return json.loads(json_str)

def generate_markdown(extraction: dict, metadata: dict) -> str:
    md = []
    md.append(f"# Project Meeting: {metadata['title']}\n")
    md.append(f"> Date: {metadata['date']}")
    if extraction.get('duration_minutes'):
        md.append(f" | Duration: {extraction['duration_minutes']} mins")
    md.append("\n")
    if extraction.get('attendees'):
        md.append(f"> Attendees: {', '.join(extraction['attendees'])}\n")
    md.append("\n")

    md.append("## Summary\n\n")
    md.append(extraction.get('summary', 'No summary available.'))
    md.append("\n\n")

    if extraction.get('topics'):
        md.append("## Key Topics\n\n")
        for topic in extraction['topics']:
            md.append(f"- **{topic['name']}** (`{topic['normalized_name']}`): {topic['context']}\n")
        md.append("\n")

    if extraction.get('decisions'):
        md.append("## Decisions Made\n\n")
        for decision in extraction['decisions']:
            md.append(f"- {decision}\n")
        md.append("\n")

    if extraction.get('action_items'):
        md.append("## Action Items\n\n")
        for item in extraction['action_items']:
            owner = f" ({item['owner']})" if item.get('owner') else ""
            md.append(f"- [ ] {item['action']}{owner}\n")
        md.append("\n")

    if extraction.get('technical_details'):
        md.append("## Technical Details\n\n")
        for detail in extraction['technical_details']:
            md.append(f"- {detail}\n")
        md.append("\n")

    if extraction.get('quotes'):
        md.append("## Notable Quotes\n\n")
        for quote in extraction['quotes']:
            md.append(f"> \"{quote['text']}\" - {quote['speaker']}\n\n")

    return ''.join(md)

def main():
    if len(sys.argv) < 2:
        print("Usage: extract_transcript.py <input_file> [--index]", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1]).expanduser().resolve()
    do_index = '--index' in sys.argv

    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"📝 Extracting transcript from: {input_path}", file=sys.stderr)
    print(f"   Using model: {MODEL} on {OLLAMA_HOST}", file=sys.stderr)
    print(f"   This may take a few minutes...\n", file=sys.stderr)

    transcript = parse_transcript(input_path)
    print(f"   Transcript length: {len(transcript)} chars", file=sys.stderr)

    date = extract_date_from_filename(input_path)
    title = extract_title_from_filename(input_path)
    metadata = {'date': date, 'title': title}

    extraction = extract_info(transcript)
    markdown = generate_markdown(extraction, metadata)

    print("✓ Extraction complete!\n", file=sys.stderr)
    print(markdown)

    if do_index:
        transcripts_dir = Path('.c0/patches/transcripts')
        transcripts_dir.mkdir(parents=True, exist_ok=True)

        summary_path = transcripts_dir / f"{date}-summary.md"
        summary_path.write_text(markdown)
        print(f"📄 Wrote summary to: {summary_path}", file=sys.stderr)

        patch_name = f"project-transcript-{date}-{title.lower().replace(' ', '-')}"

        print(f"\n🔗 Run these commands to index:", file=sys.stderr)
        print(f"   c0 add patch {patch_name} --file {summary_path}", file=sys.stderr)
        print(f"   c0 relate project HAS_PATCH {patch_name}", file=sys.stderr)

        for topic in extraction.get('topics', [])[:5]:
            name = topic['normalized_name']
            print(f"   c0 add concept {name} --force", file=sys.stderr)
            print(f"   c0 relate {name} RELATES_TO project", file=sys.stderr)
            print(f"   c0 relate project RELATES_TO {name}", file=sys.stderr)

if __name__ == '__main__':
    main()
