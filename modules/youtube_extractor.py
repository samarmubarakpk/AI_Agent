"""
youtube_extractor.py
Fetches YouTube transcripts and extracts counseling insights using OpenAI.
"""

import os
import re
import json
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv
from modules.prompts import youtube_prompt
from modules.video_transcriber import chunk_transcript
from modules.retry_handler import safe_api_call
from modules.logger import get_logger, log_failure, log_skipped

load_dotenv()
logger = get_logger("youtube_extractor")

# ── Supported / Unsupported URL Detection ─────────────────────────────────────────

YOUTUBE_PATTERN = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/shorts/)([a-zA-Z0-9_-]{11})"
)

UNSUPPORTED_DOMAINS = [
    "instagram.com", "tiktok.com", "facebook.com",
    "twitter.com", "x.com", "snapchat.com", "pinterest.com"
]

def classify_url(url: str) -> str:
    url_lower = url.lower()
    if YOUTUBE_PATTERN.search(url_lower):
        return "youtube"
    for domain in UNSUPPORTED_DOMAINS:
        if domain in url_lower:
            return "unsupported"
    return "unknown"

def extract_video_id(url: str) -> str | None:
    match = YOUTUBE_PATTERN.search(url)
    return match.group(1) if match else None

# ── Transcript Fetching ───────────────────────────────────────────────────────────

def get_transcript(video_id: str) -> str | None:
    """Fetch YouTube transcript. Handles both older dicts and newer objects."""
    try:
        fetched_data = YouTubeTranscriptApi.get_transcript(
            video_id, languages=["en", "en-US", "en-GB"]
        )

        text_chunks = []
        for chunk in fetched_data:
            # If the library returns the new 'FetchedTranscriptSnippet' object
            if hasattr(chunk, 'text'):
                text_chunks.append(chunk.text)
            # If the library returns the old dictionary format
            elif isinstance(chunk, dict) and "text" in chunk:
                text_chunks.append(chunk["text"])

        return " ".join(text_chunks)
    
    except Exception as e:
        logger.warning(f"Transcript fetch error for {video_id}: {e}")
        return None

# ── Insight Extraction ────────────────────────────────────────────────────────────

def _parse_json_response(raw: str, url: str) -> list[dict]:
    try:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(cleaned)
        insights = data.get("extracted_insights", [])
        for item in insights:
            item["content_type"] = "youtube"
            item["source"] = item.get("source") or url
        return [i for i in insights if i.get("insight") and i.get("category")]
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed for {url}: {e}")
        return []

def extract_insights(transcript: str, url: str) -> list[dict]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    chunks = chunk_transcript(transcript)
    all_insights = []

    for i, chunk in enumerate(chunks):
        logger.info(f"  Chunk {i+1}/{len(chunks)}: {url[:60]}...")

        def _call(c=chunk):
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": youtube_prompt(url)},
                    {"role": "user", "content": f"Transcript:\n{c}"}
                ],
                response_format={"type": "json_object"}
            )

        response = safe_api_call(_call, label=f"OpenAI YouTube [{url[:50]}] chunk {i+1}")
        if response:
            all_insights.extend(_parse_json_response(response.choices[0].message.content, url))

    return all_insights

# ── Main Entry ────────────────────────────────────────────────────────────────────

def process_youtube_url(url: str) -> list[dict]:
    url = url.strip()
    url_type = classify_url(url)

    if url_type == "unsupported":
        log_skipped(url, "youtube", f"Unsupported platform — skipped safely")
        return []

    if url_type == "unknown":
        log_skipped(url, "youtube", "Unrecognized URL format — skipped")
        return []

    video_id = extract_video_id(url)
    if not video_id:
        log_failure(url, "youtube", "Could not extract video ID from URL")
        return []

    logger.info(f"Fetching transcript: {url}")
    transcript = get_transcript(video_id)

    if not transcript:
        log_skipped(url, "youtube", "No transcript available")
        return []

    logger.info(f"Extracting insights ({len(transcript.split())} words): {url[:60]}")
    insights = extract_insights(transcript, url)

    if not insights:
        log_skipped(url, "youtube", "No relevant counseling content found")
    else:
        logger.info(f"  → {len(insights)} insight(s) from {url[:60]}")

    return insights