"""
sheets_writer.py
Writes deduplicated, post-processed insights to Google Sheets.

Updated for new flat JSON insight format:
{category, insight, source, content_type}

Sheet tabs:
- Insights (main — all data)
- By Category (pivot-style view per category)
- Run Log (per-run summary)
"""

import os
from datetime import datetime
from collections import defaultdict
import gspread
from google.oauth2 import service_account
from dotenv import load_dotenv
from modules.logger import get_logger, get_run_summary

load_dotenv()
logger = get_logger("sheets_writer")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

MAIN_HEADERS = [
    "Date Processed", "Category", "Insight",
    "Source", "Content Type"
]

RUN_LOG_HEADERS = [
    "Run Date", "Total Processed", "Total Insights",
    "Duplicates Removed", "Failures", "Categories"
]


def get_sheets_client(credentials_path: str) -> gspread.Client:
    creds = service_account.Credentials.from_service_account_file(
        credentials_path, scopes=SCOPES
    )
    return gspread.authorize(creds)


def ensure_tab(spreadsheet, title: str, headers: list[str]) -> gspread.Worksheet:
    """Get or create a worksheet tab with correct headers."""
    try:
        ws = spreadsheet.worksheet(title)
    except gspread.exceptions.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=title, rows=5000, cols=len(headers) + 2)
        ws.append_row(headers, value_input_option="RAW")
        logger.info(f"Created tab: {title}")
    return ws


def write_results(
    insights: list[dict],
    spreadsheet_id: str,
    credentials_path: str = "credentials/service_account.json",
    total_processed: int = 0,
    duplicates_removed: int = 0
):
    """
    Write final insights to Google Sheets.

    Args:
        insights: List of post-processed insight dicts
        spreadsheet_id: Google Sheet ID
        credentials_path: Path to service account JSON
        total_processed: Total files processed this run (for run log)
        duplicates_removed: How many duplicates were removed
    """
    if not insights:
        logger.warning("No insights to write.")
        return

    logger.info(f"Connecting to Google Sheets...")
    client = get_sheets_client(credentials_path)
    spreadsheet = client.open_by_key(spreadsheet_id)

    today = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Tab 1: Main Insights ────────────────────────────────────────────────────
    main_ws = ensure_tab(spreadsheet, "Insights", MAIN_HEADERS)

    rows = []
    categories_found = set()

    for item in insights:
        category = item.get("category", "Uncategorized")
        categories_found.add(category)
        rows.append([
            today,
            category,
            item.get("insight", ""),
            item.get("source", ""),
            item.get("content_type", "").capitalize()
        ])

    if rows:
        main_ws.append_rows(rows, value_input_option="RAW")
        logger.info(f"Wrote {len(rows)} insights to 'Insights' tab")

    # ── Tab 2: By Category (one tab per category for easy filtering) ────────────
    by_category = defaultdict(list)
    for item in insights:
        by_category[item.get("category", "Uncategorized")].append(item)

    for category, cat_insights in by_category.items():
        tab_name = f"📁 {category}"[:30]  # Sheets tab name limit
        cat_ws = ensure_tab(spreadsheet, tab_name, ["Date", "Insight", "Source", "Content Type"])
        cat_rows = [
            [today, i.get("insight", ""), i.get("source", ""), i.get("content_type", "").capitalize()]
            for i in cat_insights
        ]
        cat_ws.append_rows(cat_rows, value_input_option="RAW")
        logger.info(f"  Wrote {len(cat_rows)} insights to '{tab_name}' tab")

    # ── Tab 3: Run Log ──────────────────────────────────────────────────────────
    run_ws = ensure_tab(spreadsheet, "Run Log", RUN_LOG_HEADERS)
    run_summary = get_run_summary()

    run_ws.append_row([
        today,
        total_processed,
        len(insights),
        duplicates_removed,
        run_summary["failures"],
        ", ".join(sorted(categories_found))
    ], value_input_option="RAW")

    logger.info(
        f"\n✅ Write complete:\n"
        f"   Insights written:    {len(insights)}\n"
        f"   Categories:          {', '.join(sorted(categories_found))}\n"
        f"   Failures this run:   {run_summary['failures']}"
    )
