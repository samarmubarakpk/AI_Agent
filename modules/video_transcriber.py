"""
video_transcriber.py
Transcribes videos natively and extracts counseling insights using OpenAI.
"""

import os
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from modules.prompts import video_prompt
from modules.retry_handler import safe_api_call
from modules.logger import get_logger, log_failure, log_skipped

load_dotenv()
logger = get_logger("video_transcriber")

WORDS_PER_CHUNK = 6000

# ── Token-Aware Chunking ──────────────────────────────────────────────────────────

def chunk_transcript(text: str, words_per_chunk: int = WORDS_PER_CHUNK) -> list[str]:
    words = text.split()
    if len(words) <= words_per_chunk:
        return [text]

    chunks = []
    overlap = 200  
    start = 0

    while start < len(words):
        end = min(start + words_per_chunk, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += words_per_chunk - overlap

    logger.debug(f"Transcript split into {len(chunks)} chunks")
    return chunks

# ── Transcription ─────────────────────────────────────────────────────────────────

def transcribe_with_openai_api(video_path: str) -> str:
    # ULTIMATE FIX: OpenAI accepts .mp4 natively! No need for ffmpeg!
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    logger.info(f"Sending directly to OpenAI Whisper API: {Path(video_path).name}")
    
    # Read the video file directly into the Whisper API
    with open(video_path, "rb") as video_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=video_file
        )

    return transcript.text

def transcribe_video(video_path: str) -> str:
    return transcribe_with_openai_api(video_path)

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
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    chunks = chunk_transcript(transcript)
    all_insights = []

    for i, chunk in enumerate(chunks):
        logger.info(f"  Processing chunk {i+1}/{len(chunks)} for: {filename}")

        def _call(c=chunk):
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": video_prompt(filename)},
                    {"role": "user", "content": f"Transcript:\n{c}"}
                ],
                response_format={"type": "json_object"}
            )

        response = safe_api_call(_call, label=f"OpenAI transcript [{filename}] chunk {i+1}")
        if response:
            chunk_insights = _parse_json_response(response.choices[0].message.content, filename)
            all_insights.extend(chunk_insights)

    return all_insights

# ── Full Pipeline ─────────────────────────────────────────────────────────────────

def process_video(video_path: str, filename: str) -> list[dict]:
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