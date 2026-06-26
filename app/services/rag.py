import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_KNOWLEDGE_DIR = Path("app/data/knowledge")
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_가-힣]+", re.UNICODE)
MAX_CHUNK_CHARS = 1200


@dataclass(frozen=True)
class RagChunk:
    source: str
    title: str
    content: str

    @property
    def id(self) -> str:
        return f"{self.source}#{self.title}"


@dataclass(frozen=True)
class RetrievedRagContext:
    chunks: list[RagChunk]

    @property
    def source_ids(self) -> list[str]:
        return [chunk.id for chunk in self.chunks]

    def as_prompt_text(self) -> str:
        if not self.chunks:
            return "검색된 RAG 근거 없음"
        sections = []
        for index, chunk in enumerate(self.chunks, start=1):
            sections.append(
                f"[RAG-{index}] 출처: {chunk.id}\n"
                f"{chunk.content.strip()}"
            )
        return "\n\n".join(sections)


class LocalRagRetriever:
    """Small local RAG retriever for policy and explanation grounding.

    This is intentionally dependency-light. It retrieves short markdown chunks
    with keyword overlap so the local 1B model gets concrete ordering rules
    without pulling in a heavy LangChain runtime.
    """

    def __init__(self, knowledge_dir: Path = DEFAULT_KNOWLEDGE_DIR, top_k: int = 4) -> None:
        self.knowledge_dir = knowledge_dir
        self.top_k = top_k
        self._chunks = self._load_chunks()

    def retrieve(self, question: str, grounding: dict[str, Any]) -> RetrievedRagContext:
        query_text = " ".join([question, _grounding_query_text(grounding)])
        query_tokens = _tokenize(query_text)
        if not query_tokens:
            return RetrievedRagContext(chunks=self._chunks[: self.top_k])

        scored = []
        for chunk in self._chunks:
            chunk_tokens = _tokenize(f"{chunk.title} {chunk.content}")
            overlap = query_tokens & chunk_tokens
            if not overlap:
                continue
            score = len(overlap)
            score += _phrase_boost(query_text, chunk.content)
            scored.append((score, chunk))

        scored.sort(key=lambda item: (-item[0], item[1].source, item[1].title))
        return RetrievedRagContext(chunks=[chunk for _score, chunk in scored[: self.top_k]])

    def _load_chunks(self) -> list[RagChunk]:
        if not self.knowledge_dir.exists():
            return []

        chunks: list[RagChunk] = []
        for path in sorted(self.knowledge_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            chunks.extend(_split_markdown(path.name, text))
        return chunks


def _split_markdown(source: str, text: str) -> list[RagChunk]:
    title = Path(source).stem
    current_title = title
    current_lines: list[str] = []
    chunks: list[RagChunk] = []

    for line in text.splitlines():
        if line.startswith("## "):
            _append_chunk(chunks, source, current_title, current_lines)
            current_title = line.removeprefix("## ").strip() or title
            current_lines = []
        elif line.startswith("# "):
            current_title = line.removeprefix("# ").strip() or title
        else:
            current_lines.append(line)
    _append_chunk(chunks, source, current_title, current_lines)
    return chunks


def _append_chunk(chunks: list[RagChunk], source: str, title: str, lines: list[str]) -> None:
    content = "\n".join(lines).strip()
    if not content:
        return
    for index, start in enumerate(range(0, len(content), MAX_CHUNK_CHARS), start=1):
        suffix = f" {index}" if len(content) > MAX_CHUNK_CHARS else ""
        chunks.append(
            RagChunk(
                source=source,
                title=f"{title}{suffix}",
                content=content[start : start + MAX_CHUNK_CHARS],
            )
        )


def _grounding_query_text(grounding: dict[str, Any]) -> str:
    values: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif value is not None:
            values.append(str(value))

    visit(grounding)
    return " ".join(values)


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_PATTERN.findall(text) if len(token) >= 2}


def _phrase_boost(query_text: str, content: str) -> int:
    score = 0
    normalized_query = query_text.lower()
    normalized_content = content.lower()
    for phrase in [
        "발주",
        "추천",
        "수요",
        "매진",
        "결품",
        "탄소",
        "폐기",
        "p10",
        "p50",
        "p90",
        "lightgbm",
        "캐시",
    ]:
        if phrase in normalized_query and phrase in normalized_content:
            score += 2
    return score


def rag_context_hash(context: RetrievedRagContext) -> str:
    import hashlib

    payload = json.dumps(context.source_ids, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
