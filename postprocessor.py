"""
postprocessor.py
Final post-processing layer before writing to Google Sheets.

Steps:
1. Normalize wording via Claude
2. Ensure consistent professional tone
3. Remove within-batch redundancy
4. Validate category against fixed taxonomy
5. Ensure all insight fields are clean strings

Processes in batches to minimize API calls.
"""

import os
import json
import anthropic
from dotenv import load_dotenv
from modules.prompts import postprocess_prompt, VALID_CATEGORIES
from modules.retry_handler import safe_api_call
from modules.logger import get_logger

load_dotenv()
logger = get_logger("postprocessor")

BATCH_SIZE = 20  # Process this many insights per API call


def validate_category(category: str) -> str:
    """Ensure category is in the fixed taxonomy. Default to best match or 'Anxiety'."""
    if category in VALID_CATEGORIES:
        return category

    # Try case-insensitive match
    for valid in VALID_CATEGORIES:
        if category.lower() == valid.lower():
            return valid

    # Try partial match
    for valid in VALID_CATEGORIES:
        if valid.lower() in category.lower() or category.lower() in valid.lower():
            logger.debug(f"Category '{category}' mapped to '{valid}'")
            return valid

    logger.warning(f"Unknown category '{category}' — defaulting to 'Anxiety'")
    return "Anxiety"


def clean_insight_text(text: str) -> str:
    """Basic string cleaning before AI normalization."""
    if not text:
        return ""
    # Remove excessive whitespace
    text = " ".join(text.split())
    # Remove leading bullets/symbols
    text = text.lstrip("•-*→►▸ ")
    # Ensure it ends with a period
    if text and not text.endswith((".", "!", "?")):
        text += "."
    return text


def normalize_batch(batch: list[dict]) -> list[dict]:
    """Send a batch of insights to Claude for professional normalization."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    batch_json = json.dumps({"extracted_insights": batch}, indent=2)

    def _call():
        return client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": postprocess_prompt(batch_json)
            }]
        )

    response = safe_api_call(_call, label=f"Postprocess batch of {len(batch)}")
    if not response:
        logger.warning("Postprocess API call failed — returning uncleaned batch")
        return batch

    raw = response.content[0].text.strip()

    try:
        cleaned = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(cleaned)
        normalized = data.get("extracted_insights", batch)
        logger.debug(f"Normalized {len(normalized)} insights in batch")
        return normalized
    except json.JSONDecodeError as e:
        logger.warning(f"Postprocess JSON parse error: {e} — returning uncleaned batch")
        return batch


def postprocess(insights: list[dict]) -> list[dict]:
    """
    Full post-processing pipeline:
    1. Basic cleaning (string normalization)
    2. Category validation
    3. AI-based professional tone normalization (in batches)
    """
    if not insights:
        return []

    logger.info(f"Post-processing {len(insights)} insights...")

    # Step 1: Basic cleaning + category validation
    cleaned = []
    for item in insights:
        item["insight"] = clean_insight_text(item.get("insight", ""))
        item["category"] = validate_category(item.get("category", ""))
        item["source"] = str(item.get("source", "")).strip()
        item["content_type"] = str(item.get("content_type", "")).strip().lower()

        # Skip empty insights
        if not item["insight"] or len(item["insight"]) < 10:
            continue

        cleaned.append(item)

    logger.info(f"After basic cleaning: {len(cleaned)} insights")

    # Step 2: AI normalization in batches
    normalized_all = []
    for i in range(0, len(cleaned), BATCH_SIZE):
        batch = cleaned[i:i + BATCH_SIZE]
        logger.info(f"  Normalizing batch {i//BATCH_SIZE + 1}/{(len(cleaned)-1)//BATCH_SIZE + 1}...")
        normalized_batch = normalize_batch(batch)
        normalized_all.extend(normalized_batch)

    logger.info(f"Post-processing complete: {len(normalized_all)} insights ready")
    return normalized_all
