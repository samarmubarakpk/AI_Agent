"""
logger.py
Structured logging for the Counseling Data Agent.
Logs to both console and a rolling file in /logs directory.
Every failure is tracked with filename + reason.
"""

import logging
import os
import json
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# ── Failure tracker (in-memory for this run) ────────────────────────────────────
_failures: list[dict] = []
_skipped: list[dict] = []


def get_logger(name: str = "agent") -> logging.Logger:
    """Return a configured logger that writes to console + file."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(logging.DEBUG)

    # Console handler — clean output
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))

    # File handler — full detail
    log_file = LOG_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger


def log_failure(filename: str, source: str, reason: str, error: Exception = None):
    """Record a processing failure for the final report."""
    entry = {
        "filename": filename,
        "source": source,
        "reason": reason,
        "error": str(error) if error else None,
        "timestamp": datetime.now().isoformat()
    }
    _failures.append(entry)
    logger = get_logger()
    logger.error(f"FAILED [{source}] {filename} — {reason}" + (f" | {error}" if error else ""))


def log_skipped(filename: str, source: str, reason: str):
    """Record a skipped item (irrelevant content — not an error)."""
    _skipped.append({"filename": filename, "source": source, "reason": reason})
    get_logger().info(f"SKIPPED [{source}] {filename} — {reason}")


def write_failure_report():
    """Write a JSON failure report at end of run."""
    report_path = LOG_DIR / f"failures_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report = {
        "total_failures": len(_failures),
        "total_skipped": len(_skipped),
        "failures": _failures,
        "skipped": _skipped
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    get_logger().info(f"Failure report written → {report_path}")
    return report


def get_run_summary() -> dict:
    return {
        "failures": len(_failures),
        "skipped": len(_skipped),
        "failure_details": _failures
    }
