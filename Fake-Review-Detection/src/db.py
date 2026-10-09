"""Tiny SQLite 'database layer': stores every prediction made in the app."""
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

from src import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT    NOT NULL,
    review_text TEXT    NOT NULL,
    prediction  TEXT    NOT NULL,
    confidence  REAL    NOT NULL
);
"""


def _connect():
    return sqlite3.connect(config.DB_PATH)


def init_db():
    with closing(_connect()) as conn, conn:
        conn.execute(SCHEMA)


def log_prediction(text: str, prediction: str, confidence: float):
    init_db()
    with closing(_connect()) as conn, conn:
        conn.execute(
            "INSERT INTO predictions (created_at, review_text, prediction, confidence) VALUES (?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), text, prediction, float(confidence)),
        )


def recent_predictions(limit: int = 20):
    init_db()
    with closing(_connect()) as conn:
        return conn.execute(
            "SELECT created_at, review_text, prediction, confidence FROM predictions "
            "ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
