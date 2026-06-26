import json
from functools import lru_cache
from typing import Any, Literal

import httpx

from app.core.config import Settings


class LocalLlamaClient:
    def __init__(self, settings: Settings) -> None:
        self.backend = settings.local_llm_backend
        self.model = settings.local_llm_model
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.hf_model = settings.local_hf_model
        self.gguf_model_path = settings.local_gguf_model_path

    def generate_text(
        self,
        prompt: str,
        system_prompt: str = "You are a concise assistant for supply-chain operations.",
        max_tokens: int = 700,
        temperature: float = 0,
    ) -> str:
        if self.backend == "transformers":
            return self._generate_with_transformers(prompt, system_prompt, max_tokens, temperature)
        if self.backend == "llama_cpp":
            return self._generate_with_llama_cpp(prompt, system_prompt, max_tokens, temperature)
        return self._generate_with_ollama(prompt, system_prompt, max_tokens, temperature)

    def generate_json(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int = 700,
        temperature: float = 0,
    ) -> dict[str, Any]:
        return _loads_json_object(self.generate_text(prompt, system_prompt, max_tokens, temperature))

    def _generate_with_ollama(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        response = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
            },
            timeout=60.0,
        )
        response.raise_for_status()
        payload = response.json()
        message = payload.get("message", {})
        content = message.get("content") or payload.get("response")
        if not isinstance(content, str):
            return json.dumps(payload, ensure_ascii=False)
        return content

    def _generate_with_transformers(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        generator = _load_transformers_generator(self.hf_model)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        output = generator(
            messages,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 0.01),
            return_full_text=False,
        )
        text = output[0].get("generated_text", "")
        if isinstance(text, list):
            return str(text[-1].get("content", "")) if text else ""
        return str(text)

    def _generate_with_llama_cpp(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        if not self.gguf_model_path:
            raise RuntimeError("LOCAL_GGUF_MODEL_PATH is required when LOCAL_LLM_BACKEND=llama_cpp.")
        llama = _load_llama_cpp(self.gguf_model_path)
        response = llama.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return str(response["choices"][0]["message"]["content"])


@lru_cache(maxsize=1)
def _load_transformers_generator(model_id: str):
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "LOCAL_LLM_BACKEND=transformers requires transformers/torch. "
            "Install them or use LOCAL_LLM_BACKEND=ollama."
        ) from exc
    return pipeline("text-generation", model=model_id, device_map="auto")


@lru_cache(maxsize=1)
def _load_llama_cpp(model_path: str):
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError(
            "LOCAL_LLM_BACKEND=llama_cpp requires llama-cpp-python. "
            "Install it or use LOCAL_LLM_BACKEND=ollama."
        ) from exc
    return Llama(model_path=model_path, n_ctx=4096, verbose=False)


def _loads_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM response did not contain a JSON object")
    return json.loads(text[start : end + 1])


LocalLlmBackend = Literal["ollama", "transformers", "llama_cpp"]
