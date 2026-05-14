"""
image_extractor.py
Extracts counseling insights from screenshots using OpenAI or Gemini vision.
"""

import base64
import io
import json
import os
from typing import Any

import requests
from dotenv import load_dotenv
from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError

from modules.logger import get_logger, log_failure, log_skipped
from modules.prompts import (
    VALID_CATEGORIES,
    VALID_THERAPEUTIC_USES,
    screenshot_prompt,
)
from modules.retry_handler import safe_api_call

load_dotenv()
logger = get_logger("image_extractor")

DEFAULT_VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4.1")
MIN_CONFIDENCE = float(os.getenv("MIN_VISION_CONFIDENCE", "0.55"))
MAX_IMAGE_EDGE = int(os.getenv("VISION_MAX_IMAGE_EDGE", "2200"))
MIN_TEXT_EDGE = int(os.getenv("VISION_MIN_TEXT_EDGE", "1200"))


OPENAI_INSIGHT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "counseling_screenshot_insights",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "extracted_insights": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "category": {"type": "string", "enum": VALID_CATEGORIES},
                            "topic": {"type": "string"},
                            "insight": {"type": "string"},
                            "therapeutic_use": {
                                "type": "string",
                                "enum": VALID_THERAPEUTIC_USES,
                            },
                            "evidence": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "source": {"type": "string"},
                            "content_type": {"type": "string", "enum": ["screenshot"]},
                        },
                        "required": [
                            "category",
                            "topic",
                            "insight",
                            "therapeutic_use",
                            "evidence",
                            "confidence",
                            "source",
                            "content_type",
                        ],
                    },
                }
            },
            "required": ["extracted_insights"],
        },
    },
}


def _coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))


def _clean_json_text(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def _parse_json_response(raw: str, filename: str, content_type: str = "screenshot") -> list[dict]:
    """Parse and validate model JSON. Returns production-safe insight dicts."""
    try:
        data = json.loads(_clean_json_text(raw))
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed for {filename}: {e} | Raw: {raw[:200]}")
        return []

    if isinstance(data, list):
        insights = data
    else:
        insights = data.get("extracted_insights", [])

    valid = []
    for item in insights:
        if not isinstance(item, dict):
            continue

        insight = " ".join(str(item.get("insight", "")).split())
        category = str(item.get("category", "")).strip()
        therapeutic_use = str(item.get("therapeutic_use", "Other")).strip()
        confidence = _coerce_confidence(item.get("confidence", 0.0))

        if category not in VALID_CATEGORIES:
            logger.debug(f"Rejected invalid category for {filename}: {category}")
            continue
        if therapeutic_use not in VALID_THERAPEUTIC_USES:
            therapeutic_use = "Other"
        if len(insight) < 18:
            continue
        if confidence < MIN_CONFIDENCE:
            logger.debug(
                f"Rejected low-confidence insight for {filename}: "
                f"{confidence:.2f} < {MIN_CONFIDENCE:.2f}"
            )
            continue

        valid.append(
            {
                "category": category,
                "topic": " ".join(str(item.get("topic", "")).split())[:80] or category,
                "insight": insight,
                "therapeutic_use": therapeutic_use,
                "evidence": " ".join(str(item.get("evidence", "")).split())[:300],
                "confidence": round(confidence, 2),
                "source": str(item.get("source") or filename).strip(),
                "content_type": content_type,
            }
        )

    return valid


def _encode_bytes(image_bytes: bytes) -> str:
    return base64.standard_b64encode(image_bytes).decode("utf-8")


def _resize_for_vision(image: Image.Image) -> Image.Image:
    width, height = image.size
    longest = max(width, height)

    if longest > MAX_IMAGE_EDGE:
        ratio = MAX_IMAGE_EDGE / longest
        new_size = (max(1, int(width * ratio)), max(1, int(height * ratio)))
        return image.resize(new_size, Image.Resampling.LANCZOS)

    if longest < MIN_TEXT_EDGE:
        ratio = MIN_TEXT_EDGE / longest
        new_size = (max(1, int(width * ratio)), max(1, int(height * ratio)))
        return image.resize(new_size, Image.Resampling.LANCZOS)

    return image


def _prepare_image_for_vision(image_bytes: bytes, filename: str) -> tuple[str, str, dict]:
    """
    Normalize screenshot bytes for text-heavy vision extraction.
    Returns base64 PNG, media type, and lightweight quality metadata.
    """
    quality = {"variant": "original", "width": None, "height": None}

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image)

        if getattr(image, "is_animated", False):
            image.seek(0)

        image = image.convert("RGB")
        original_size = image.size
        image = _resize_for_vision(image)
        image = ImageOps.autocontrast(image)
        image = ImageEnhance.Sharpness(image).enhance(1.25)

        output = io.BytesIO()
        image.save(output, format="PNG", optimize=True)

        quality = {
            "variant": "preprocessed",
            "width": image.size[0],
            "height": image.size[1],
            "original_width": original_size[0],
            "original_height": original_size[1],
        }
        return _encode_bytes(output.getvalue()), "image/png", quality
    except (UnidentifiedImageError, OSError) as e:
        logger.warning(f"Could not preprocess {filename}; using original bytes: {e}")
        return _encode_bytes(image_bytes), _guess_media_type(filename), quality


def _guess_media_type(filename: str) -> str:
    ext = filename.lower().rsplit(".", 1)[-1]
    media_map = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
        "gif": "image/gif",
    }
    return media_map.get(ext, "image/jpeg")


def _call_openai(image_b64: str, media_type: str, filename: str, quality: dict) -> list[dict]:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("VISION_MODEL", DEFAULT_VISION_MODEL)

    user_text = (
        "Extract counseling-relevant insights from this screenshot. "
        "Use the source file name exactly. "
        f"Image preparation metadata: {json.dumps(quality, sort_keys=True)}"
    )

    def _make_request():
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": screenshot_prompt(filename)},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_b64}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
            temperature=0,
            max_tokens=1800,
            response_format=OPENAI_INSIGHT_SCHEMA,
        )

    response = safe_api_call(_make_request, label=f"OpenAI vision [{filename}]")
    if not response:
        return []

    return _parse_json_response(response.choices[0].message.content, filename)


def _call_gemini(image_b64: str, media_type: str, filename: str) -> list[dict]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        log_failure(filename, "screenshot", "GEMINI_API_KEY is not configured")
        return []

    model = os.getenv("GEMINI_VISION_MODEL", "gemini-1.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": screenshot_prompt(filename)},
                    {"inline_data": {"mime_type": media_type, "data": image_b64}},
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "response_mime_type": "application/json",
        },
    }

    def _make_request():
        resp = requests.post(url, json=payload, timeout=45)
        resp.raise_for_status()
        return resp

    response = safe_api_call(_make_request, label=f"Gemini vision [{filename}]")
    if not response:
        return []

    raw = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    return _parse_json_response(raw, filename)


def extract_from_image(image_bytes: bytes, filename: str) -> list[dict]:
    """
    Extract counseling insights from an image.
    Returns list of insight dicts; empty means no reliable counseling content.
    """
    image_b64, media_type, quality = _prepare_image_for_vision(image_bytes, filename)
    provider = os.getenv("VISION_PROVIDER", "openai").lower()
    logger.info(f"Processing screenshot [{provider}]: {filename}")

    if provider == "gemini":
        insights = _call_gemini(image_b64, media_type, filename)
    else:
        insights = _call_openai(image_b64, media_type, filename, quality)

    if not insights:
        log_skipped(filename, "screenshot", "No reliable counseling content found")
    else:
        for insight in insights:
            insight["image_quality"] = quality.get("variant", "unknown")
        avg_conf = sum(i.get("confidence", 0) for i in insights) / len(insights)
        logger.info(
            f"  -> {len(insights)} insight(s) extracted from {filename} "
            f"(avg confidence {avg_conf:.2f})"
        )

    return insights
