import sqlite3
from pathlib import Path
from datetime import datetime
import json

def _utcnow_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS files (
  id TEXT PRIMARY KEY,
  file_name TEXT NOT NULL,
  original_filename TEXT NOT NULL,
  file_type TEXT NOT NULL,
  storage_path TEXT NOT NULL,
  size_bytes INTEGER NOT NULL,
  num_pages INTEGER,
  num_chunks INTEGER NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  question TEXT NOT NULL,
  answer TEXT NOT NULL,
  sources_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
"""

class DB:
    def __init__(self, db_path: Path):
        self.db_path = str(db_path)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init(self):
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    # ---------- Files ----------
    def insert_file(self, *, file_id: str, file_name: str, original_filename: str, file_type: str,
                    storage_path: str, size_bytes: int, num_pages: int | None, num_chunks: int):
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO files (id, file_name, original_filename, file_type, storage_path, size_bytes,
                                   num_pages, num_chunks, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (file_id, file_name, original_filename, file_type, storage_path, size_bytes,
                 num_pages, num_chunks, _utcnow_iso())
            )

    def list_files(self):
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM files ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_file(self, file_id: str):
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
        return dict(row) if row else None

    def delete_file(self, file_id: str):
        with self.connect() as conn:
            conn.execute("DELETE FROM files WHERE id = ?", (file_id,))

    # ---------- History ----------
    def insert_history(self, *, question: str, answer: str, sources: list[dict]):
        sources_json = json.dumps(sources, ensure_ascii=False)
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO history (question, answer, sources_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (question, answer, sources_json, _utcnow_iso())
            )

    def list_history(self):
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT id, question, created_at FROM history ORDER BY id DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_history(self, history_id: int):
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM history WHERE id = ?",
                (history_id,)
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["sources"] = json.loads(d["sources_json"])
        return d

    def delete_history(self, history_id: int):
        with self.connect() as conn:
            conn.execute("DELETE FROM history WHERE id = ?", (history_id,))
