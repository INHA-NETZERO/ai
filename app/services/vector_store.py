import json
import sqlite3
from array import array
from pathlib import Path
from typing import Any

from app.services.embedding import cosine_similarity

try:
    import sqlite_vec
except ImportError:
    sqlite_vec = None


class SemanticCacheRecord:
    def __init__(
        self,
        endpoint: str,
        input_hash: str,
        embedding: list[float],
        response: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        self.endpoint = endpoint
        self.input_hash = input_hash
        self.embedding = embedding
        self.response = response
        self.metadata = metadata


class SQLiteVectorStore:
    """sqlite-vec vector cache with a plain SQLite fallback."""

    def __init__(self, db_path: Path, dimensions: int) -> None:
        self.db_path = db_path
        self.dimensions = dimensions
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._use_sqlite_vec = sqlite_vec is not None
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        if self._use_sqlite_vec and sqlite_vec is not None:
            try:
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)
            except sqlite3.Error:
                self._use_sqlite_vec = False
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    endpoint TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    response TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_semantic_cache_endpoint
                ON semantic_cache(endpoint)
                """
            )
            if self._use_sqlite_vec:
                try:
                    conn.execute(
                        f"""
                        CREATE VIRTUAL TABLE IF NOT EXISTS semantic_cache_vec
                        USING vec0(embedding float[{self.dimensions}])
                        """
                    )
                except sqlite3.Error:
                    self._use_sqlite_vec = False

    def add(self, record: SemanticCacheRecord) -> None:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO semantic_cache(endpoint, input_hash, embedding, response, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    record.endpoint,
                    record.input_hash,
                    json.dumps(record.embedding),
                    json.dumps(record.response, ensure_ascii=False),
                    json.dumps(record.metadata, ensure_ascii=False),
                ),
            )
            if self._use_sqlite_vec:
                try:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO semantic_cache_vec(rowid, embedding)
                        VALUES (?, ?)
                        """,
                        (cursor.lastrowid, _serialize_embedding(record.embedding)),
                    )
                except sqlite3.Error:
                    self._use_sqlite_vec = False

    def search(
        self,
        endpoint: str,
        embedding: list[float],
        threshold: float,
    ) -> tuple[dict[str, Any], dict[str, Any], float] | None:
        if self._use_sqlite_vec:
            result = self._search_with_sqlite_vec(endpoint, embedding, threshold)
            if result is not None:
                return result
        best: tuple[dict[str, Any], dict[str, Any], float] | None = None
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT embedding, response, metadata
                FROM semantic_cache
                WHERE endpoint = ?
                ORDER BY id DESC
                LIMIT 200
                """,
                (endpoint,),
            ).fetchall()

        for row in rows:
            score = cosine_similarity(embedding, json.loads(row["embedding"]))
            if score >= threshold and (best is None or score > best[2]):
                best = (json.loads(row["response"]), json.loads(row["metadata"]), score)
        return best

    def _search_with_sqlite_vec(
        self,
        endpoint: str,
        embedding: list[float],
        threshold: float,
    ) -> tuple[dict[str, Any], dict[str, Any], float] | None:
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT cache.response, cache.metadata, vec.distance
                    FROM semantic_cache_vec vec
                    JOIN semantic_cache cache ON cache.id = vec.rowid
                    WHERE vec.embedding MATCH ? AND k = 20 AND cache.endpoint = ?
                    ORDER BY vec.distance
                    """,
                    (_serialize_embedding(embedding), endpoint),
                ).fetchall()
        except sqlite3.Error:
            self._use_sqlite_vec = False
            return None

        best: tuple[dict[str, Any], dict[str, Any], float] | None = None
        for row in rows:
            score = _cosine_from_l2(float(row["distance"]))
            if score >= threshold and (best is None or score > best[2]):
                best = (json.loads(row["response"]), json.loads(row["metadata"]), score)
        return best


def _serialize_embedding(embedding: list[float]) -> bytes:
    return array("f", embedding).tobytes()


def _cosine_from_l2(distance: float) -> float:
    return max(-1.0, min(1.0, 1 - (distance**2 / 2)))
