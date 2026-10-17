"""SQLite persistence: sessions, message logs, feedback, stats."""
import os
import sqlite3
from contextlib import contextmanager

DB_PATH = os.getenv("CHATBOT_DB", "chatbot.db")


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                role TEXT NOT NULL CHECK (role IN ('user', 'bot')),
                text TEXT NOT NULL,
                intent TEXT,
                matched_faq TEXT,
                confidence REAL,
                sentiment TEXT,
                latency_ms INTEGER,
                helpful INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
            """
        )


def ensure_session(session_id: str):
    with get_conn() as c:
        c.execute("INSERT OR IGNORE INTO sessions (id) VALUES (?)", (session_id,))


def log_message(session_id, role, text, intent=None, matched_faq=None,
                confidence=None, sentiment=None, latency_ms=None) -> int:
    with get_conn() as c:
        cur = c.execute(
            """INSERT INTO messages
               (session_id, role, text, intent, matched_faq, confidence, sentiment, latency_ms)
               VALUES (?,?,?,?,?,?,?,?)""",
            (session_id, role, text, intent, matched_faq, confidence, sentiment, latency_ms),
        )
        return cur.lastrowid


def recent_messages(session_id: str, limit: int = 6):
    with get_conn() as c:
        rows = c.execute(
            "SELECT id, role, text, matched_faq FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def history(session_id: str, limit: int = 100):
    with get_conn() as c:
        rows = c.execute(
            "SELECT id, role, text, created_at FROM messages WHERE session_id=? ORDER BY id LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def set_feedback(message_id: int, helpful: bool) -> bool:
    with get_conn() as c:
        cur = c.execute(
            "UPDATE messages SET helpful=? WHERE id=? AND role='bot'",
            (1 if helpful else 0, message_id),
        )
        return cur.rowcount > 0


def logs(limit: int = 100, offset: int = 0):
    with get_conn() as c:
        rows = c.execute(
            "SELECT * FROM messages ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset)
        ).fetchall()
    return [dict(r) for r in rows]


def stats():
    with get_conn() as c:
        r = c.execute(
            """SELECT
                 (SELECT COUNT(*) FROM sessions) AS sessions,
                 (SELECT COUNT(*) FROM messages WHERE role='user') AS user_messages,
                 (SELECT COUNT(*) FROM messages WHERE role='bot' AND intent='fallback') AS fallbacks,
                 (SELECT ROUND(AVG(confidence),3) FROM messages WHERE role='bot') AS avg_confidence,
                 (SELECT ROUND(AVG(latency_ms),1) FROM messages WHERE role='bot') AS avg_latency_ms,
                 (SELECT COUNT(*) FROM messages WHERE helpful=1) AS helpful_yes,
                 (SELECT COUNT(*) FROM messages WHERE helpful=0) AS helpful_no
            """
        ).fetchone()
        top = c.execute(
            """SELECT matched_faq, COUNT(*) AS n FROM messages
               WHERE role='bot' AND matched_faq IS NOT NULL
               GROUP BY matched_faq ORDER BY n DESC LIMIT 5"""
        ).fetchall()
    out = dict(r)
    out["top_faqs"] = [dict(t) for t in top]
    return out
