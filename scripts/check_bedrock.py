#!/usr/bin/env python
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.services.llm import BedrockLlamaClient


def main() -> None:
    settings = get_settings()
    client = BedrockLlamaClient(settings)
    answer = client.generate_text(
        prompt="한국어로 'Bedrock 연결 확인 완료'라고만 답해줘.",
        system_prompt="You are a terse connectivity checker.",
        max_tokens=30,
        temperature=0,
    )
    print(answer)


if __name__ == "__main__":
    main()
