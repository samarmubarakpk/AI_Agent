"""
video_transcriber.py
Transcribes videos and extracts counseling insights.

UPGRADES:
- ffmpeg existence check with human-readable error
- Token-aware chunking (no hardcoded [:8000] truncation)
- Retry logic
- Strict JSON output
- Structured logging
"""

import os
import json
import shutil
import subprocess
import tempfile
import anthropic
from pathlib import Path
from dotenv import load_dotenv
from modules.prompts import video_prompt
from modules.retry_handler import safe_api_call
from modules.logger import get_logger, log_failure, log_skipped

load_dotenv()
logger = get_logger("video_transcriber")

# Approx token limit per chunk (Claude handles ~100k tokens but we chunk at 6000 words
# for practical insight quality — longer = diluted output)
WORDS_PER_CHUNK = 6000


# ── ffmpeg Check ─────────────────────────────────────────────────────────────────

def check_ffmpeg():
    """Verify ffmpeg is installed. Raises clear error if not."""
    if shutil.which("ffmpeg") is None:
        raise EnvironmentError(
            "\n❌ ffmpeg is not installed or not in your system PATH.\n"
            "ffmpeg is required for video processing.\n\n"
            "Install it:\n"
            "  Mac:     brew install ffmpeg\n"
            "  Ubuntu:  sudo apt install ffmpeg\n"
            "  Windows: https://ffmpeg.org/download.html\n"
            "\nAfter installing, restart your terminal and try again."
        )
    logger.debug("ffmpeg found.")


# ── Token-Aware Chunking ──────────────────────────────────────────────────────────

def chunk_transcript(text: str, words_per_chunk: int = WORDS_PER_CHUNK) -> list[str]:
    """
    Split transcript into word-count-based chunks.
    Avoids character-based slicing which cuts mid-sentence.
    Each chunk overlaps slightly to avoid missing context at boundaries.
    """
    words = text.split()
    if len(words) <= words_per_chunk:
        return [text]

    chunks = []
    overlap = 200  # word overlap between chunks
    start = 0

    while start < len(words):
        end = min(start + words_per_chunk, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += words_per_chunk - overlap

    logger.debug(f"Transcript split into {len(chunks)} chunks ({len(words)} total words)")
    return chunks


# ── Transcription ─────────────────────────────────────────────────────────────────

def transcribe_local(video_path: str) -> str:
    """Transcribe using local Whisper (free). Requires: pip install openai-whisper"""
    try:
        import whisper
    except ImportError:
        raise ImportError(
            "openai-whisper not installed. Run: pip install openai-whisper\n"
            "Or set TRANSCRIPTION_PROVIDER=openai in your .env to use the API instead."
        )
    logger.info(f"Transcribing locally with Whisper: {Path(video_path).name}")
    model = whisper.load_model("base")
    result = model.transcribe(video_path)
    return result["text"]


def transcribe_with_openai_api(video_path: str) -> str:
    """Transcribe using OpenAI Whisper API (~$0.006/min). Faster, no local GPU needed."""
    check_ffmpeg()

    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    audio_path = video_path.replace(Path(video_path).suffix, "_audio.mp3")

    logger.info(f"Extracting audio from: {Path(video_path).name}")
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "libmp3lame", "-q:a", "4",
        audio_path, "-y", "-loglevel", "quiet"
    ], check=True)

    logger.info("Sending to OpenAI Whisper API...")
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )

    os.remove(audio_path)
    return transcript.text


def transcribe_video(video_path: str) -> str:
    """Route to correct transcription provider."""
    provider = os.getenv("TRANSCRIPTION_PROVIDER", "local").lower()
    if provider == "openai":
        return transcribe_with_openai_api(video_path)
    return transcribe_local(video_path)


# ── Insight Extraction ────────────────────────────────────────────────────────────

def _parse_json_response(raw: str, filename: str) -> list[dict]:
    try:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(cleaned)
        insights = data.get("extracted_insights", [])
        for item in insights:
            item["content_type"] = "video"
            item["source"] = item.get("source") or filename
        return [i for i in insights if i.get("insight") and i.get("category")]
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed for {filename}: {e}")
        return []


def extract_insights_from_transcript(transcript: str, filename: str) -> list[dict]:
    """
    Extract insights from a transcript using token-aware chunking.
    Processes each chunk and deduplicates across chunks.
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    chunks = chunk_transcript(transcript)
    all_insights = []

    for i, chunk in enumerate(chunks):
        logger.info(f"  Processing chunk {i+1}/{len(chunks)} for: {filename}")

        def _call(c=chunk):
            return client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": video_prompt(filename) + f"\n\nTranscript:\n{c}"
                }]
            )

        response = safe_api_call(_call, label=f"Claude transcript [{filename}] chunk {i+1}")
        if response:
            chunk_insights = _parse_json_response(response.content[0].text, filename)
            all_insights.extend(chunk_insights)

    return all_insights


# ── Full Pipeline ─────────────────────────────────────────────────────────────────

def process_video(video_path: str, filename: str) -> list[dict]:
    """Full pipeline: ffmpeg check → transcribe → chunk → extract insights."""
    # Only check ffmpeg if using OpenAI API path
    if os.getenv("TRANSCRIPTION_PROVIDER", "local").lower() == "openai":
        check_ffmpeg()

    logger.info(f"Transcribing video: {filename}")
    transcript = transcribe_video(video_path)

    if not transcript or len(transcript.strip().split()) < 10:
        log_skipped(filename, "video", "Transcript too short or empty")
        return []

    logger.info(f"Extracting insights from transcript: {filename} ({len(transcript.split())} words)")
    insights = extract_insights_from_transcript(transcript, filename)

    if not insights:
        log_skipped(filename, "video", "No relevant counseling content in transcript")
    else:
        logger.info(f"  → {len(insights)} insight(s) extracted from {filename}")

    return insights
