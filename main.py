"""
main.py
Production-ready orchestrator for the Counseling Data Agent v2.

Pipeline:
  Drive → Extract → Deduplicate → Post-process → Write to Sheets

Usage:
    python main.py                         # Process all folders
    python main.py --folder screenshots    # One folder only
    python main.py --limit 10 --dry-run   # Test on 10 files
    python main.py --skip-dedup           # Skip deduplication
    python main.py --skip-postprocess     # Skip AI normalization
"""

import os
import sys
import argparse
import tempfile
import time
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

from modules.drive_reader import (
    get_drive_service, list_files, get_file_bytes,
    download_file, read_text_file,
    IMAGE_TYPES, VIDEO_TYPES, GOOGLE_DOC_TYPE, classify_file
)
from modules.image_extractor import extract_from_image
from modules.video_transcriber import process_video
from modules.youtube_extractor import process_youtube_url
from modules.blog_scraper import process_blog_url
from modules.deduplicator import deduplicate
from modules.postprocessor import postprocess
from modules.sheets_writer import write_results
from modules.logger import get_logger, log_failure, write_failure_report

logger = get_logger("main")

# ── Config ───────────────────────────────────────────────────────────────────────

FOLDER_IDS = {
    "screenshots": os.getenv("SCREENSHOTS_FOLDER_ID"),
    "videos":      os.getenv("VIDEOS_FOLDER_ID"),
    "youtube":     os.getenv("YOUTUBE_LINKS_FOLDER_ID"),
    "blogs":       os.getenv("BLOGS_FOLDER_ID"),
}

SPREADSHEET_ID   = os.getenv("SPREADSHEET_ID")
CREDENTIALS_PATH = "credentials/service_account.json"


# ── Folder Processors ─────────────────────────────────────────────────────────────

def process_screenshots(service, folder_id: str, limit: int = None) -> list[dict]:
    logger.info("\n📸 Processing Screenshots...")
    files = list_files(service, folder_id)
    image_files = [f for f in files if f.get("mimeType") in IMAGE_TYPES]

    if limit:
        image_files = image_files[:limit]

    logger.info(f"Found {len(image_files)} images to process")
    all_insights = []

    for file in tqdm(image_files, desc="Screenshots"):
        try:
            image_bytes = get_file_bytes(service, file["id"])
            insights = extract_from_image(image_bytes, file["name"])
            all_insights.extend(insights)
        except Exception as e:
            log_failure(file["name"], "screenshot", "Unexpected error", error=e)

    logger.info(f"Screenshots → {len(all_insights)} total insights")
    return all_insights


def process_videos(service, folder_id: str, limit: int = None) -> list[dict]:
    logger.info("\n🎥 Processing Videos...")
    files = list_files(service, folder_id)
    video_files = [f for f in files if f.get("mimeType") in VIDEO_TYPES]

    if limit:
        video_files = video_files[:limit]

    logger.info(f"Found {len(video_files)} videos to process")
    all_insights = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for file in tqdm(video_files, desc="Videos"):
            try:
                local_path = os.path.join(tmpdir, file["name"])
                logger.info(f"Downloading: {file['name']}...")
                download_file(service, file["id"], local_path)
                insights = process_video(local_path, file["name"])
                all_insights.extend(insights)
            except Exception as e:
                log_failure(file["name"], "video", "Unexpected error", error=e)

    logger.info(f"Videos → {len(all_insights)} total insights")
    return all_insights


def process_youtube_folder(service, folder_id: str, limit: int = None) -> list[dict]:
    logger.info("\n▶️  Processing YouTube Links...")
    files = list_files(service, folder_id)

    all_urls = []
    for file in files:
        try:
            content = read_text_file(service, file["id"], mime_type=file.get("mimeType", "text/plain"))
            urls = [
                line.strip() for line in content.split("\n")
                if line.strip() and line.strip().startswith("http")
            ]
            all_urls.extend(urls)
        except Exception as e:
            log_failure(file["name"], "youtube-folder", "Could not read URL file", error=e)

    if limit:
        all_urls = all_urls[:limit]

    logger.info(f"Found {len(all_urls)} URLs to process")
    all_insights = []

    for url in tqdm(all_urls, desc="YouTube"):
        try:
            insights = process_youtube_url(url)
            all_insights.extend(insights)
        except Exception as e:
            log_failure(url, "youtube", "Unexpected error", error=e)

    logger.info(f"YouTube → {len(all_insights)} total insights")
    return all_insights


def process_blogs_folder(service, folder_id: str, limit: int = None) -> list[dict]:
    logger.info("\n📝 Processing Blog Links...")
    files = list_files(service, folder_id)

    all_urls = []
    for file in files:
        try:
            content = read_text_file(service, file["id"], mime_type=file.get("mimeType", "text/plain"))
            urls = [
                line.strip() for line in content.split("\n")
                if line.strip() and line.strip().startswith("http")
            ]
            all_urls.extend(urls)
        except Exception as e:
            log_failure(file["name"], "blogs-folder", "Could not read URL file", error=e)

    if limit:
        all_urls = all_urls[:limit]

    logger.info(f"Found {len(all_urls)} URLs to process")
    all_insights = []

    for url in tqdm(all_urls, desc="Blogs"):
        try:
            insights = process_blog_url(url)
            all_insights.extend(insights)
        except Exception as e:
            log_failure(url, "blog", "Unexpected error", error=e)

    logger.info(f"Blogs → {len(all_insights)} total insights")
    return all_insights


# ── Main ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Counseling Data Agent v2")
    parser.add_argument("--folder", choices=["screenshots", "videos", "youtube", "blogs"])
    parser.add_argument("--dry-run", action="store_true", help="Don't write to Sheets")
    parser.add_argument("--limit", type=int, default=None, help="Max files per folder (for testing)")
    parser.add_argument("--skip-dedup", action="store_true", help="Skip deduplication")
    parser.add_argument("--skip-postprocess", action="store_true", help="Skip AI post-processing")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("🚀 Counseling Data Agent v2 Starting")
    logger.info(f"   Vision:          {os.getenv('VISION_PROVIDER', 'claude')}")
    logger.info(f"   Transcription:   {os.getenv('TRANSCRIPTION_PROVIDER', 'local')}")
    logger.info(f"   Deduplication:   {'OFF' if args.skip_dedup else 'ON'}")
    logger.info(f"   Post-processing: {'OFF' if args.skip_postprocess else 'ON'}")
    logger.info(f"   Dry run:         {args.dry_run}")
    logger.info("=" * 60)

    # Connect to Drive
    logger.info("Connecting to Google Drive...")
    service = get_drive_service(CREDENTIALS_PATH)

    all_insights = []
    total_processed = 0

    folders_to_run = [args.folder] if args.folder else ["screenshots", "videos", "youtube", "blogs"]

    for folder_name in folders_to_run:
        folder_id = FOLDER_IDS.get(folder_name)
        if not folder_id:
            logger.warning(f"Skipping '{folder_name}': no folder ID in .env")
            continue

        if folder_name == "screenshots":
            insights = process_screenshots(service, folder_id, args.limit)
        elif folder_name == "videos":
            insights = process_videos(service, folder_id, args.limit)
        elif folder_name == "youtube":
            insights = process_youtube_folder(service, folder_id, args.limit)
        elif folder_name == "blogs":
            insights = process_blogs_folder(service, folder_id, args.limit)
        else:
            insights = []

        all_insights.extend(insights)
        total_processed += len(insights)

    logger.info(f"\n📊 Raw insights collected: {len(all_insights)}")

    if not all_insights:
        logger.warning("No insights collected. Check folder IDs and permissions.")
        write_failure_report()
        return

    # ── Stage 2: Deduplication ──────────────────────────────────────────────────
    pre_dedup_count = len(all_insights)
    if not args.skip_dedup:
        logger.info("\n🔍 Running semantic deduplication...")
        all_insights = deduplicate(all_insights)
    duplicates_removed = pre_dedup_count - len(all_insights)

    # ── Stage 3: Post-processing ────────────────────────────────────────────────
    if not args.skip_postprocess:
        logger.info("\n✍️  Running post-processing...")
        all_insights = postprocess(all_insights)

    # ── Stage 4: Write to Sheets ────────────────────────────────────────────────
    logger.info(f"\n📤 Final insight count: {len(all_insights)}")

    if args.dry_run:
        logger.info("[DRY RUN] Skipping Google Sheets write.")
        if all_insights:
            logger.info("Sample insight:")
            logger.info(str(all_insights[0]))
    else:
        logger.info("Writing to Google Sheets...")
        write_results(
            all_insights,
            SPREADSHEET_ID,
            CREDENTIALS_PATH,
            total_processed=total_processed,
            duplicates_removed=duplicates_removed
        )

    # ── Final Report ────────────────────────────────────────────────────────────
    write_failure_report()
    logger.info("\n🎉 Pipeline complete!")


if __name__ == "__main__":
    main()
