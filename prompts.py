"""
prompts.py
Central place for all AI prompts.
Enforces strict JSON output, fixed category taxonomy, and noise filtering.
"""

import os
from dotenv import load_dotenv

load_dotenv()

COUNSELING_CONTEXT = os.getenv(
    "COUNSELING_CONTEXT",
    "mental health counseling, therapy, mindfulness, anxiety, depression, self-care"
)

# ── Fixed Category Taxonomy (STRICT — no custom categories allowed) ─────────────
VALID_CATEGORIES = [
    "Anxiety",
    "Depression",
    "Self-esteem",
    "Relationships",
    "Trauma",
    "Career Stress"
]

CATEGORIES_STR = ", ".join(VALID_CATEGORIES)

# ── Noise to explicitly ignore ──────────────────────────────────────────────────
NOISE_INSTRUCTIONS = """
Explicitly IGNORE and do NOT extract:
- UI elements (buttons, menus, icons, navigation bars)
- Usernames, handles, profile names (@mentions)
- Emojis used as decoration (🔥❤️👍 etc.)
- Timestamps, dates, view counts, like counts
- Filler text ("link in bio", "follow for more", "swipe left")
- Ads, sponsored content labels
- Platform watermarks or branding
"""

# ── JSON Output Schema ──────────────────────────────────────────────────────────
JSON_SCHEMA = """
Return ONLY valid JSON. No markdown. No explanation. No text before or after.
Use this exact format:

{
  "extracted_insights": [
    {
      "category": "<one of: Anxiety, Depression, Self-esteem, Relationships, Trauma, Career Stress>",
      "insight": "<clear, rewritten therapeutic insight in plain English>",
      "source": "<filename or URL>",
      "content_type": "<screenshot | video | youtube | blog>"
    }
  ]
}

If there are NO relevant counseling insights, return:
{"extracted_insights": []}

Rules:
- category MUST be exactly one of: """ + CATEGORIES_STR + """
- insight must be a complete, meaningful sentence — not a fragment
- Do NOT include raw quotes, UI text, usernames, or emojis in insights
- Each insight must stand alone and make sense without seeing the original
"""

# ── Per-source system prompts ────────────────────────────────────────────────────

def screenshot_prompt(filename: str) -> str:
    return f"""You are a counseling research assistant helping a professional counselor organize research materials.
Your job: extract ONLY counseling-relevant insights from this screenshot.

Counselor's focus areas: {COUNSELING_CONTEXT}

{NOISE_INSTRUCTIONS}

Source file: {filename}
Content type: screenshot

{JSON_SCHEMA}"""


def video_prompt(filename: str) -> str:
    return f"""You are a counseling research assistant helping a professional counselor organize research materials.
Your job: extract ONLY counseling-relevant insights from this video transcript.

Counselor's focus areas: {COUNSELING_CONTEXT}

{NOISE_INSTRUCTIONS}

Source file: {filename}
Content type: video

{JSON_SCHEMA}"""


def youtube_prompt(url: str) -> str:
    return f"""You are a counseling research assistant helping a professional counselor organize research materials.
Your job: extract ONLY counseling-relevant insights from this YouTube video transcript.

Counselor's focus areas: {COUNSELING_CONTEXT}

{NOISE_INSTRUCTIONS}

Source: {url}
Content type: youtube

{JSON_SCHEMA}"""


def blog_prompt(url: str) -> str:
    return f"""You are a counseling research assistant helping a professional counselor organize research materials.
Your job: extract ONLY counseling-relevant insights from this article or blog post.

Counselor's focus areas: {COUNSELING_CONTEXT}

{NOISE_INSTRUCTIONS}

Source: {url}
Content type: blog

{JSON_SCHEMA}"""


def postprocess_prompt(insights_json: str) -> str:
    return f"""You are editing a counseling knowledge base for a professional counselor.

You will receive a JSON array of extracted insights. Your job:
1. Normalize wording to be clear, professional, and consistent
2. Ensure each insight is a complete, standalone sentence
3. Remove redundancy within the list
4. Fix grammar and tone — should sound like a professional counseling reference
5. Keep all category and source fields exactly as-is
6. Do NOT add or remove insights — only clean the "insight" field text

Input:
{insights_json}

{JSON_SCHEMA}"""
