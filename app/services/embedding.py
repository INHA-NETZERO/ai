import hashlib
import re

import numpy as np


TOKEN_PATTERN = re.compile(r"[\w가-힣]+", re.UNICODE)


class HashEmbeddingService:
    """Deterministic local embeddings for semantic cache lookup."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        tokens = TOKEN_PATTERN.findall(text.lower())
        if not tokens:
            return vector.tolist()

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector.tolist()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    left_arr = np.array(left, dtype=np.float32)
    right_arr = np.array(right, dtype=np.float32)
    denom = float(np.linalg.norm(left_arr) * np.linalg.norm(right_arr))
    if denom == 0:
        return 0.0
    return float(np.dot(left_arr, right_arr) / denom)
