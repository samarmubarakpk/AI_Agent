"""
prompts.py
Central place for all AI prompts.
Enforces strict JSON output, fixed taxonomy, and noise filtering.
"""

import os
from dotenv import load_dotenv

load_dotenv()

COUNSELING_CONTEXT = os.getenv(
    "COUNSELING_CONTEXT",
    "mental health counseling, therapy, mindfulness, anxiety, depression, self-care",
)

VALID_CATEGORIES = [
    "Anxiety",
    "Depression",
    "Self-esteem",
    "Relationships",
    "Trauma",
    "Career Stress",
]

VALID_THERAPEUTIC_USES = [
    "Psychoeducation",
    "Client Discussion",
    "Reflection Prompt",
    "Coping Skill",
    "Assessment Cue",
    "Resource Idea",
    "Other",
]

CATEGORIES_STR = ", ".join(VALID_CATEGORIES)
THERAPEUTIC_USES_STR = ", ".join(VALID_THERAPEUTIC_USES)

NOISE_INSTRUCTIONS = """
Explicitly IGNORE and do NOT extract:
- UI elements such as buttons, menus, icons, navigation bars, notification bars, and watermarks
- Usernames, handles, profile names, @mentions, and follower counts
- Decorative emojis, stickers, filters, and reaction icons
- Timestamps, dates, view counts, like counts, comments, and share counts
- Filler text such as "link in bio", "follow for more", "swipe left", or calls to subscribe
- Ads, sponsored labels, coupons, and unrelated product promotion
- Background text that is unrelated to the main counseling-relevant content
"""

JSON_SCHEMA = """
Return ONLY valid JSON. No markdown. No explanation. No text before or after.
Use this exact format:

{
  "extracted_insights": [
    {
      "category": "<one of: Anxiety, Depression, Self-esteem, Relationships, Trauma, Career Stress>",
      "topic": "<short specific topic, 2-6 words>",
      "insight": "<clear, rewritten therapeutic insight in plain English>",
      "therapeutic_use": "<one of: Psychoeducation, Client Discussion, Reflection Prompt, Coping Skill, Assessment Cue, Resource Idea, Other>",
      "evidence": "<brief source cue that supports the insight, without UI noise>",
      "confidence": 0.0,
      "source": "<filename or URL>",
      "content_type": "<screenshot | video | youtube | blog>"
    }
  ]
}

If there are NO relevant counseling insights, return:
{"extracted_insights": []}

Rules:
- category MUST be exactly one of: """ + CATEGORIES_STR + """
- therapeutic_use MUST be exactly one of: """ + THERAPEUTIC_USES_STR + """
- confidence must be a number from 0.0 to 1.0 based on how clearly the source supports the insight
- topic must be specific, concise, and useful for spreadsheet filtering
- insight must be a complete, meaningful sentence, not a fragment
- evidence should mention only the relevant visible/transcribed idea, not usernames, buttons, counts, or unrelated UI
- Do NOT include raw long quotes, UI text, usernames, or emojis in insights
- Each insight must stand alone and make sense without seeing the original
"""


def screenshot_prompt(filename: str) -> str:
    return f"""You are a counseling research assistant helping a professional counselor organize research materials.
Your job: extract ONLY counseling-relevant insights from this screenshot.

Counselor's focus areas: {COUNSELING_CONTEXT}

Screenshot extraction policy:
- First identify the main post/content area. Ignore app chrome, comments, likes, profile headers, menus, notification bars, and unrelated surrounding content.
- Extract an insight only when the screenshot contains a counseling, mental health, relationship, self-development, trauma, stress, or career-wellbeing idea that a counselor could reuse.
- Prefer the meaning of the relevant content over copying exact text.
- If the image is blurry, cropped, decorative, or too ambiguous, return no insight rather than guessing.
- Separate distinct therapeutic ideas into separate items, but do not split one idea into many tiny fragments.
- Do not diagnose people in the screenshot.

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
4. Fix grammar and tone so it sounds like a professional counseling reference
5. Keep all category, topic, therapeutic_use, evidence, confidence, source, and content_type fields exactly as-is unless a value is invalid
6. Do NOT add or remove insights; only clean the "insight" field text

Input:
{insights_json}

{JSON_SCHEMA}"""
