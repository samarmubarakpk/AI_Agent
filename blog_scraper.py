"""
blog_scraper.py
Scrapes blog/article URLs and extracts counseling insights.

UPGRADES:
- Strict JSON output
- Token-aware chunking for long articles
- Retry logic
- Unsupported/unreachable URL handling
- Structured logging
"""

import os
import json
import requests
import anthropic
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from modules.prompts import blog_prompt
from modules.video_transcriber import chunk_transcript
from modules.retry_handler import safe_api_call
from modules.logger import get_logger, log_failure, log_skipped

load_dotenv()
logger = get_logger("blog_scraper")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# URLs that are not scrapeable (social media, login-walled, etc.)
UNSUPPORTED_DOMAINS = [
    "instagram.com", "tiktok.com", "twitter.com", "x.com",
    "facebook.com", "linkedin.com", "reddit.com",
    "youtube.com", "youtu.be"
]


def is_unsupported(url: str) -> bool:
    url_lower = url.lower()
    return any(domain in url_lower for domain in UNSUPPORTED_DOMAINS)


# ── Scraping ──────────────────────────────────────────────────────────────────────

def scrape_article(url: str) -> str | None:
    """Scrape clean text from a blog/article URL."""
    if is_unsupported(url):
        log_skipped(url, "blog", "Unsupported domain — not scrapeable")
        return None

    try:
        response = requests.get(url.strip(), headers=HEADERS, timeout=20)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        log_failure(url, "blog", "Request timed out")
        return None
    except requests.exceptions.ConnectionError:
        log_failure(url, "blog", "Connection error — site may be down")
        return None
    except requests.exceptions.HTTPError as e:
        log_failure(url, "blog", f"HTTP error: {e.response.status_code}")
        return None
    except Exception as e:
        log_failure(url, "blog", f"Unexpected scrape error", error=e)
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove clutter
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]):
        tag.decompose()

    # Find main content
    main = (
        soup.find("article") or
        soup.find("main") or
        soup.find(class_=lambda c: c and any(
            x in str(c).lower() for x in ["article", "content", "post-body", "entry"]
        )) or
        soup.find("body")
    )

    if not main:
        return None

    lines = [line.strip() for line in main.get_text(separator="\n").split("\n") if line.strip()]
    text = "\n".join(lines)

    if len(text) < 100:
        return None

    return text


# ── Insight Extraction ────────────────────────────────────────────────────────────

def _parse_json_response(raw: str, url: str) -> list[dict]:
    try:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(cleaned)
        insights = data.get("extracted_insights", [])
        for item in insights:
            item["content_type"] = "blog"
            item["source"] = item.get("source") or url
        return [i for i in insights if i.get("insight") and i.get("category")]
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed for {url}: {e}")
        return []


def extract_insights(article_text: str, url: str) -> list[dict]:
    """Extract insights from article text with chunking for long content."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    chunks = chunk_transcript(article_text, words_per_chunk=5000)
    all_insights = []

    for i, chunk in enumerate(chunks):
        logger.info(f"  Chunk {i+1}/{len(chunks)}: {url[:60]}")

        def _call(c=chunk):
            return client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": blog_prompt(url) + f"\n\nArticle content:\n{c}"
                }]
            )

        response = safe_api_call(_call, label=f"Claude blog [{url[:50]}] chunk {i+1}")
        if response:
            all_insights.extend(_parse_json_response(response.content[0].text, url))

    return all_insights


# ── Main Entry ────────────────────────────────────────────────────────────────────

def process_blog_url(url: str) -> list[dict]:
    """Full pipeline: validate → scrape → chunk → extract insights."""
    url = url.strip()

    if is_unsupported(url):
        log_skipped(url, "blog", "Unsupported domain — skipped safely")
        return []

    logger.info(f"Scraping: {url}")
    article_text = scrape_article(url)

    if not article_text:
        return []

    logger.info(f"Extracting insights ({len(article_text.split())} words): {url[:60]}")
    insights = extract_insights(article_text, url)

    if not insights:
        log_skipped(url, "blog", "No relevant counseling content found")
    else:
        logger.info(f"  → {len(insights)} insight(s) from {url[:60]}")

    return insights
