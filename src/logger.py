import json
import logging
import traceback as _tb
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path("melody_cli.log")


def _get_logger() -> logging.Logger:
    logger = logging.getLogger("melody_cli")
    if not logger.handlers:
        handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger


def _entry(event: str, **fields) -> str:
    return json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "event": event, **fields}, default=str)


def log_query_session(query: str, results: list, low_confidence: bool) -> None:
    songs_log = [
        {"id": song["id"], "title": song["title"], "score": round(sim, 4)}
        for song, sim, _ in results
    ]
    entry = _entry(
        "query",
        query=query,
        results=songs_log,
        best_score=songs_log[0]["score"] if songs_log else None,
        low_confidence=low_confidence,
        warning="LOW CONFIDENCE" if low_confidence else None,
    )
    _get_logger().info(entry)


def log_mode_session(mode: str, profile_name: str, results: list) -> None:
    songs_log = [
        {"id": song["id"], "title": song["title"], "score": round(score, 4)}
        for song, score, _ in results
    ]
    entry = _entry("mode", mode=mode, profile=profile_name, results=songs_log)
    _get_logger().info(entry)


def log_exception(context: str, exc: Exception) -> None:
    entry = _entry(
        "error",
        context=context,
        error=type(exc).__name__,
        message=str(exc),
        traceback=_tb.format_exc(),
    )
    _get_logger().error(entry)
