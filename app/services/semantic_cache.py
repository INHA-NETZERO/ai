from typing import Any

from app.core.config import Settings
from app.services.cache import cache_key, stable_json
from app.services.embedding import HashEmbeddingService
from app.services.vector_store import SemanticCacheRecord, SQLiteVectorStore


class ChatSemanticCache:
    """Semantic cache reserved for low-power chatbot answers only."""

    def __init__(self, settings: Settings) -> None:
        self.threshold = settings.semantic_cache_threshold
        self.embedding_service = HashEmbeddingService()
        self.vector_store = SQLiteVectorStore(settings.vector_db_path, self.embedding_service.dimensions)

    def get(self, namespace: str, question: str) -> tuple[dict[str, Any], float] | None:
        embedding = self.embedding_service.embed(question)
        result = self.vector_store.search(namespace, embedding, self.threshold)
        if result is None:
            return None
        response, _metadata, score = result
        return response, score

    def set(self, namespace: str, question: str, response: dict[str, Any]) -> None:
        payload = {"namespace": namespace, "question": question}
        self.vector_store.add(
            SemanticCacheRecord(
                endpoint=namespace,
                input_hash=cache_key("chat", payload),
                embedding=self.embedding_service.embed(question),
                response=response,
                metadata={"source": "low_power_chatbot", "payload": stable_json(payload)},
            )
        )

    @property
    def backend(self) -> str:
        return "sqlite_vec_or_sqlite"
