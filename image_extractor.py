"""
image_extractor.py
Extracts counseling insights from screenshots using Claude or Gemini vision.

UPGRADES:
- Strict JSON output with fixed categories
- Retry logic via retry_handler
- Noise filtering in prompts
- Structured logging
"""

import os
import base64
import json
import requests
import anthropic
from dotenv import load_dotenv
from modules.prompts import screenshot_prompt
from modules.retry_handler import safe_api_call
from modules.logger import get_logger, log_failure, log_skipped

load_dotenv()
logger = get_logger("image_extractor")


def _parse_json_response(raw: str, filename: str, content_type: str = "screenshot") -> list[dict]:
    """Parse strict JSON response from AI. Returns list of insight dicts."""
    try:
        # Strip any accidental markdown fences
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(cleaned)
        insights = data.get("extracted_insights", [])

        # Validate and enforce content_type field
        valid = []
        for item in insights:
            if item.get("insight") and item.get("category"):
                item["content_type"] = content_type
                item["source"] = item.get("source") or filename
                valid.append(item)

        return valid

    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed for {filename}: {e} | Raw: {raw[:200]}")
        return []


# ── Claude Vision ────────────────────────────────────────────────────────────────

def _call_claude(image_b64: str, media_type: str, filename: str) -> list[dict]:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def _make_request():
        return client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1000,
            system=screenshot_prompt(filename),
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": image_b64}
                    },
                    {"type": "text", "text": "Extract counseling-relevant insights from this screenshot."}
                ]
            }]
        )

    response = safe_api_call(_make_request, label=f"Claude vision [{filename}]")
    if not response:
        return []

    return _parse_json_response(response.content[0].text, filename)


# ── Gemini Vision (Free) ─────────────────────────────────────────────────────────

def _call_gemini(image_b64: str, media_type: str, filename: str) -> list[dict]:
    api_key = os.getenv("GEMINI_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

    payload = {
        "contents": [{
            "parts": [
                {"text": screenshot_prompt(filename) + "\n\nExtract counseling-relevant insights."},
                {"inline_data": {"mime_type": media_type, "data": image_b64}}
            ]
        }],
        "generationConfig": {"temperature": 0.1}
    }

    def _make_request():
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        return resp

    response = safe_api_call(_make_request, label=f"Gemini vision [{filename}]")
    if not response:
        return []

    raw = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    return _parse_json_response(raw, filename)


# ── Main Entry ───────────────────────────────────────────────────────────────────

def extract_from_image(image_bytes: bytes, filename: str) -> list[dict]:
    """
    Extract counseling insights from an image.
    Returns list of insight dicts (may be empty if irrelevant).
    """
    ext = filename.lower().rsplit(".", 1)[-1]
    media_map = {
        "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "png": "image/png", "webp": "image/webp", "gif": "image/gif"
    }
    media_type = media_map.get(ext, "image/jpeg")
    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    provider = os.getenv("VISION_PROVIDER", "claude").lower()
    logger.info(f"Processing screenshot [{provider}]: {filename}")

    if provider == "gemini":
        insights = _call_gemini(image_b64, media_type, filename)
    else:
        insights = _call_claude(image_b64, media_type, filename)

    if not insights:
        log_skipped(filename, "screenshot", "No relevant counseling content found")
    else:
        logger.info(f"  → {len(insights)} insight(s) extracted from {filename}")

    return insights
