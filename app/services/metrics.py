from dataclasses import dataclass


@dataclass
class CacheMetrics:
    exact_hits: int = 0
    exact_misses: int = 0
    semantic_hits: int = 0
    semantic_misses: int = 0

    def exact_hit_rate(self) -> float:
        total = self.exact_hits + self.exact_misses
        return round(self.exact_hits / total, 4) if total else 0.0

    def semantic_hit_rate(self) -> float:
        total = self.semantic_hits + self.semantic_misses
        return round(self.semantic_hits / total, 4) if total else 0.0

    def model_dump(self) -> dict[str, int | float]:
        return {
            "exact_hits": self.exact_hits,
            "exact_misses": self.exact_misses,
            "exact_hit_rate": self.exact_hit_rate(),
            "semantic_hits": self.semantic_hits,
            "semantic_misses": self.semantic_misses,
            "semantic_hit_rate": self.semantic_hit_rate(),
            "estimated_llm_calls_saved": self.semantic_hits,
        }
