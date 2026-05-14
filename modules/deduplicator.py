"""
deduplicator.py
Semantic deduplication of extracted insights before writing to Google Sheets.

Uses sentence-transformers to compute embeddings and cosine similarity.
Insights above the similarity threshold are considered duplicates.
The most complete version is kept, others are discarded.

SIMILARITY THRESHOLD: 0.85 (configurable via env DEDUP_THRESHOLD)
"""

import os
import numpy as np
from dotenv import load_dotenv
from modules.logger import get_logger

load_dotenv()
logger = get_logger("deduplicator")

SIMILARITY_THRESHOLD = float(os.getenv("DEDUP_THRESHOLD", "0.85"))


def _load_model():
    """Lazy-load the sentence transformer model."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")  # Fast, small, good quality
        logger.info("Sentence transformer model loaded.")
        return model
    except ImportError:
        logger.warning(
            "sentence-transformers not installed. Skipping semantic dedup.\n"
            "Run: pip install sentence-transformers"
        )
        return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def pick_better(a: dict, b: dict) -> dict:
    """Return the more complete/longer insight of two duplicates."""
    len_a = len(a.get("insight", ""))
    len_b = len(b.get("insight", ""))
    return a if len_a >= len_b else b


def deduplicate(insights: list[dict]) -> list[dict]:
    """
    Semantic deduplication of insight list.
    Returns deduplicated list — keeping best version of each duplicate group.

    If sentence-transformers is unavailable, falls back to exact-string dedup.
    """
    if not insights:
        return []

    original_count = len(insights)
    model = _load_model()

    if model is None:
        # Fallback: simple exact-text dedup
        seen = set()
        unique = []
        for item in insights:
            text = item.get("insight", "").strip().lower()
            if text not in seen:
                seen.add(text)
                unique.append(item)
        logger.info(f"Exact dedup: {original_count} → {len(unique)} insights")
        return unique

    # Compute embeddings
    texts = [item.get("insight", "") for item in insights]
    logger.info(f"Computing embeddings for {len(texts)} insights...")
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    # Greedy dedup — O(n²) but fine for <5000 insights
    kept = []
    discarded = set()

    for i in range(len(insights)):
        if i in discarded:
            continue

        best = insights[i]
        best_idx = i

        for j in range(i + 1, len(insights)):
            if j in discarded:
                continue

            sim = cosine_similarity(embeddings[i], embeddings[j])

            if sim >= SIMILARITY_THRESHOLD:
                # Pick the better one, discard the other
                winner = pick_better(best, insights[j])
                if winner is insights[j]:
                    discarded.add(best_idx)
                    best = insights[j]
                    best_idx = j
                else:
                    discarded.add(j)

                logger.debug(
                    f"Duplicate (sim={sim:.2f}): \"{insights[i]['insight'][:60]}\" "
                    f"≈ \"{insights[j]['insight'][:60]}\""
                )

        kept.append(best)

    removed = original_count - len(kept)
    logger.info(f"Deduplication: {original_count} → {len(kept)} insights ({removed} duplicates removed)")
    return kept
