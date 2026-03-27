"""
drive_reader.py
Connects to Google Drive — lists and downloads files from specified folders.

FIXES:
- Google Docs now exported as plain text (not get_media which crashes)
- Structured logging throughout
- Clear error messages
"""

import os
import io
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from modules.logger import get_logger

logger = get_logger("drive_reader")

SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/spreadsheets"
]

# Supported MIME types
IMAGE_TYPES = {
    "image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"
}
VIDEO_TYPES = {
    "video/mp4", "video/quicktime", "video/x-msvideo",
    "video/mpeg", "video/webm", "video/x-matroska"
}
TEXT_TYPES = {
    "text/plain",
    # NOTE: Google Docs (application/vnd.google-apps.document) is handled
    # separately via export_media — do NOT include in get_media calls
}
GOOGLE_DOC_TYPE = "application/vnd.google-apps.document"


def get_drive_service(credentials_path: str = "credentials/service_account.json"):
    """Build and return a Google Drive API service client."""
    if not Path(credentials_path).exists():
        raise FileNotFoundError(
            f"Service account credentials not found at: {credentials_path}\n"
            "Please follow the README setup guide to create a service account."
        )
    creds = service_account.Credentials.from_service_account_file(
        credentials_path, scopes=SCOPES
    )
    service = build("drive", "v3", credentials=creds)
    logger.info("Google Drive service connected.")
    return service


def list_files(service, folder_id: str, page_size: int = 1000) -> list[dict]:
    """List all files in a Drive folder, handling pagination."""
    files = []
    page_token = None

    while True:
        try:
            response = service.files().list(
                q=f"'{folder_id}' in parents and trashed=false",
                pageSize=page_size,
                fields="nextPageToken, files(id, name, mimeType, size, createdTime)",
                pageToken=page_token
            ).execute()
        except Exception as e:
            logger.error(f"Failed to list files in folder {folder_id}: {e}")
            break

        batch = response.get("files", [])
        files.extend(batch)
        logger.debug(f"Fetched {len(batch)} files from folder {folder_id}")

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    logger.info(f"Total files found in folder {folder_id}: {len(files)}")
    return files


def download_file(service, file_id: str, destination_path: str) -> str:
    """Download a binary file (video, image) from Drive to disk."""
    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)
    request = service.files().get_media(fileId=file_id)

    with open(destination_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

    logger.debug(f"Downloaded file to: {destination_path}")
    return destination_path


def get_file_bytes(service, file_id: str) -> bytes:
    """Download a binary file (image) into memory as bytes."""
    request = service.files().get_media(fileId=file_id)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buffer.getvalue()


def read_text_file(service, file_id: str, mime_type: str = "text/plain") -> str:
    """
    Read a text file from Drive into a string.

    CRITICAL FIX: Native Google Docs cannot use get_media().
    They MUST use export_media() with mimeType='text/plain'.
    Regular .txt files use get_media() as normal.
    """
    if mime_type == GOOGLE_DOC_TYPE:
        # ✅ Correct: Export Google Doc as plain text
        request = service.files().export_media(fileId=file_id, mimeType="text/plain")
        logger.debug(f"Exporting Google Doc {file_id} as plain text")
    else:
        # Regular text file
        request = service.files().get_media(fileId=file_id)
        logger.debug(f"Downloading text file {file_id}")

    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()

    return buffer.getvalue().decode("utf-8", errors="ignore")


def classify_file(file: dict) -> str:
    """Classify a Drive file into a processing category."""
    mime = file.get("mimeType", "")
    if mime in IMAGE_TYPES:
        return "image"
    if mime in VIDEO_TYPES:
        return "video"
    if mime in TEXT_TYPES or mime == GOOGLE_DOC_TYPE:
        return "text"
    return "unknown"
