"""
sheets_writer.py
Writes deduplicated, post-processed insights to Google Sheets.
"""

import os
from collections import defaultdict
from datetime import datetime

import gspread
from dotenv import load_dotenv
from google.oauth2 import service_account

from modules.logger import get_logger, get_run_summary

load_dotenv()
logger = get_logger("sheets_writer")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

MAIN_HEADERS = [
    "Date Processed",
    "Category",
    "Topic",
    "Therapeutic Use",
    "Insight",
    "Evidence",
    "Confidence",
    "Source",
    "Content Type",
]

CATEGORY_HEADERS = [
    "Date",
    "Topic",
    "Therapeutic Use",
    "Insight",
    "Evidence",
    "Confidence",
    "Source",
    "Content Type",
]

RUN_LOG_HEADERS = [
    "Run Date",
    "Total Processed",
    "Total Insights",
    "Duplicates Removed",
    "Failures",
    "Categories",
]


def get_sheets_client(credentials_path: str) -> gspread.Client:
    creds = service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=SCOPES,
    )
    return gspread.authorize(creds)


def ensure_tab(spreadsheet, title: str, headers: list[str]) -> gspread.Worksheet:
    """Get or create a worksheet tab and make sure expected headers exist."""
    try:
        ws = spreadsheet.worksheet(title)
    except gspread.exceptions.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=title, rows=5000, cols=len(headers) + 2)
        ws.append_row(headers, value_input_option="RAW")
        logger.info(f"Created tab: {title}")
        return ws

    current_headers = ws.row_values(1)
    if not current_headers:
        ws.update("1:1", [headers], value_input_option="RAW")
        return ws

    merged_headers = current_headers[:]
    for header in headers:
        if header not in merged_headers:
            merged_headers.append(header)

    if merged_headers != current_headers:
        ws.update("1:1", [merged_headers], value_input_option="RAW")
        logger.info(f"Updated headers for tab: {title}")

    return ws


def _main_row(item: dict, today: str) -> list:
    return [
        today,
        item.get("category", "Uncategorized"),
        item.get("topic", ""),
        item.get("therapeutic_use", ""),
        item.get("insight", ""),
        item.get("evidence", ""),
        item.get("confidence", ""),
        item.get("source", ""),
        item.get("content_type", "").capitalize(),
    ]


def _category_row(item: dict, today: str) -> list:
    return [
        today,
        item.get("topic", ""),
        item.get("therapeutic_use", ""),
        item.get("insight", ""),
        item.get("evidence", ""),
        item.get("confidence", ""),
        item.get("source", ""),
        item.get("content_type", "").capitalize(),
    ]


def write_results(
    insights: list[dict],
    spreadsheet_id: str,
    credentials_path: str = "credentials/service_account.json",
    total_processed: int = 0,
    duplicates_removed: int = 0,
):
    """
    Write final insights to Google Sheets.
    """
    if not insights:
        logger.warning("No insights to write.")
        return

    logger.info("Connecting to Google Sheets...")
    client = get_sheets_client(credentials_path)
    spreadsheet = client.open_by_key(spreadsheet_id)

    today = datetime.now().strftime("%Y-%m-%d %H:%M")

    main_ws = ensure_tab(spreadsheet, "Insights", MAIN_HEADERS)
    rows = []
    categories_found = set()

    for item in insights:
        categories_found.add(item.get("category", "Uncategorized"))
        rows.append(_main_row(item, today))

    if rows:
        main_ws.append_rows(rows, value_input_option="RAW")
        logger.info(f"Wrote {len(rows)} insights to 'Insights' tab")

    by_category = defaultdict(list)
    for item in insights:
        by_category[item.get("category", "Uncategorized")].append(item)

    for category, cat_insights in by_category.items():
        tab_name = f"{category}"[:30]
        cat_ws = ensure_tab(spreadsheet, tab_name, CATEGORY_HEADERS)
        cat_rows = [_category_row(item, today) for item in cat_insights]
        cat_ws.append_rows(cat_rows, value_input_option="RAW")
        logger.info(f"Wrote {len(cat_rows)} insights to '{tab_name}' tab")

    run_ws = ensure_tab(spreadsheet, "Run Log", RUN_LOG_HEADERS)
    run_summary = get_run_summary()

    run_ws.append_row(
        [
            today,
            total_processed,
            len(insights),
            duplicates_removed,
            run_summary["failures"],
            ", ".join(sorted(categories_found)),
        ],
        value_input_option="RAW",
    )

    logger.info(
        "\nWrite complete:\n"
        f"   Insights written:    {len(insights)}\n"
        f"   Categories:          {', '.join(sorted(categories_found))}\n"
        f"   Failures this run:   {run_summary['failures']}"
    )
