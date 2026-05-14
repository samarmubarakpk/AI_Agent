"""
postprocessor.py
Final post-processing layer before writing to Google Sheets.
Processes in batches to minimize API calls.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from modules.prompts import postprocess_prompt, VALID_CATEGORIES, VALID_THERAPEUTIC_USES
from modules.retry_handler import safe_api_call
from modules.logger import get_logger

load_dotenv()
logger = get_logger("postprocessor")

BATCH_SIZE = 20  # Process this many insights per API call

def validate_category(category: str) -> str:
    if category in VALID_CATEGORIES:
        return category
    for valid in VALID_CATEGORIES:
        if category.lower() == valid.lower():
            return valid
    for valid in VALID_CATEGORIES:
        if valid.lower() in category.lower() or category.lower() in valid.lower():
            return valid
    return "Anxiety"

def validate_therapeutic_use(value: str) -> str:
    if value in VALID_THERAPEUTIC_USES:
        return value
    for valid in VALID_THERAPEUTIC_USES:
        if str(value).lower() == valid.lower():
            return valid
    return "Other"

def clean_confidence(value) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(max(0.0, min(1.0, confidence)), 2)

def clean_insight_text(text: str) -> str:
    if not text: return ""
    text = " ".join(text.split())
    text = text.lstrip("•-*→►▸ ")
    if text and not text.endswith((".", "!", "?")):
        text += "."
    return text

def normalize_batch(batch: list[dict]) -> list[dict]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    batch_json = json.dumps({"extracted_insights": batch}, indent=2)

    def _call():
        return client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": postprocess_prompt(batch_json)
            }],
            response_format={"type": "json_object"}
        )

    response = safe_api_call(_call, label=f"Postprocess batch of {len(batch)}")
    if not response:
        logger.warning("Postprocess API call failed — returning uncleaned batch")
        return batch

    try:
        raw = response.choices[0].message.content.strip()
        cleaned = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(cleaned)
        return data.get("extracted_insights", batch)
    except json.JSONDecodeError as e:
        logger.warning(f"Postprocess JSON parse error: {e} — returning uncleaned batch")
        return batch

def postprocess(insights: list[dict]) -> list[dict]:
    if not insights: return []

    logger.info(f"Post-processing {len(insights)} insights...")

    cleaned = []
    for item in insights:
        item["insight"] = clean_insight_text(item.get("insight", ""))
        item["category"] = validate_category(item.get("category", ""))
        item["topic"] = " ".join(str(item.get("topic", item["category"])).split())[:80] or item["category"]
        item["therapeutic_use"] = validate_therapeutic_use(item.get("therapeutic_use", "Other"))
        item["evidence"] = " ".join(str(item.get("evidence", "")).split())[:300]
        item["confidence"] = clean_confidence(item.get("confidence", 0.0))
        item["source"] = str(item.get("source", "")).strip()
        item["content_type"] = str(item.get("content_type", "")).strip().lower()

        if not item["insight"] or len(item["insight"]) < 10:
            continue
        cleaned.append(item)

    logger.info(f"After basic cleaning: {len(cleaned)} insights")

    normalized_all = []
    for i in range(0, len(cleaned), BATCH_SIZE):
        batch = cleaned[i:i + BATCH_SIZE]
        logger.info(f"  Normalizing batch {i//BATCH_SIZE + 1}/{(len(cleaned)-1)//BATCH_SIZE + 1}...")
        normalized_batch = normalize_batch(batch)
        normalized_all.extend(normalized_batch)

    logger.info(f"Post-processing complete: {len(normalized_all)} insights ready")
    return normalized_all
