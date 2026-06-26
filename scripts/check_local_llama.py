#!/usr/bin/env python
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.services.llm import LocalLlamaClient


def main() -> None:
    settings = get_settings()
    print(f"provider={settings.llm_provider}")
    print(f"backend={settings.local_llm_backend}")
    print(f"model={settings.local_llm_model}")
    print(f"hf_model={settings.local_hf_model}")
    print(f"gguf_model_path={settings.local_gguf_model_path or ''}")
    print(f"ollama_base_url={settings.ollama_base_url}")
    client = LocalLlamaClient(settings)
    answer = client.generate_text(
        prompt="한국어로 '로컬 Llama 3.2 1B 연결 확인 완료'라고만 답해줘.",
        system_prompt="You are a terse connectivity checker.",
        max_tokens=40,
        temperature=0,
    )
    print(answer)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(
            "Local Llama check failed. "
            "If using Ollama, run `ollama pull llama3.2:1b` and `ollama serve` first. "
            "If using llama.cpp, set LOCAL_LLM_BACKEND=llama_cpp and LOCAL_GGUF_MODEL_PATH. "
            "If using transformers, install transformers/torch and set LOCAL_LLM_BACKEND=transformers. "
            f"Error: {exc}"
        ) from exc
