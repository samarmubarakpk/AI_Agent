"""
youtube_extractor.py
Fetches YouTube transcripts (free) and extracts counseling insights.

UPGRADES:
- Strict JSON output
- Token-aware chunking for long videos
- Instagram/unsupported link detection — skip safely with log
- Retry logic
- Structured logging
"""

import os
import re
import json
import anthropic
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
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
    """Return 'youtube', 'unsupported', or 'unknown'."""
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
    """Fetch YouTube transcript. Returns None if unavailable."""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(
            video_id, languages=["en", "en-US", "en-GB"]
        )
        return " ".join(chunk["text"] for chunk in transcript_list)
    except TranscriptsDisabled:
        logger.warning(f"Transcripts disabled for video: {video_id}")
        return None
    except NoTranscriptFound:
        logger.warning(f"No English transcript for video: {video_id}")
        return None
    except Exception as e:
        logger.error(f"Transcript fetch error for {video_id}: {e}")
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
    """Process full transcript with token-aware chunking."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    chunks = chunk_transcript(transcript)
    all_insights = []

    for i, chunk in enumerate(chunks):
        logger.info(f"  Chunk {i+1}/{len(chunks)}: {url[:60]}...")

        def _call(c=chunk):
            return client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": youtube_prompt(url) + f"\n\nTranscript:\n{c}"
                }]
            )

        response = safe_api_call(_call, label=f"Claude YouTube [{url[:50]}] chunk {i+1}")
        if response:
            all_insights.extend(_parse_json_response(response.content[0].text, url))

    return all_insights


# ── Main Entry ────────────────────────────────────────────────────────────────────

def process_youtube_url(url: str) -> list[dict]:
    """
    Full pipeline for a YouTube URL.
    Safely skips unsupported platforms (Instagram, TikTok, etc.)
    """
    url = url.strip()
    url_type = classify_url(url)

    # ✅ Handle unsupported links gracefully — pipeline does NOT crash
    if url_type == "unsupported":
        log_skipped(url, "youtube", f"Unsupported platform — skipped safely")
        logger.warning(f"Unsupported link skipped: {url}")
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
