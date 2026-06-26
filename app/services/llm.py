import json
from typing import Any
from urllib.parse import quote

import httpx
from botocore.exceptions import BotoCoreError, ClientError

from app.core.config import Settings
from app.services.aws_clients import create_aws_client


class BedrockLlamaClient:
    def __init__(self, settings: Settings) -> None:
        self.model_id = settings.bedrock_model_id
        self.region_name = settings.aws_region
        self.bearer_token = settings.aws_bearer_token_bedrock or settings.bedrock_api_key
        self._client = None if self.bearer_token else create_aws_client("bedrock-runtime", settings)

    def generate_text(
        self,
        prompt: str,
        system_prompt: str = "You are a concise assistant for supply-chain operations.",
        max_tokens: int = 700,
        temperature: float = 0,
    ) -> str:
        if self.bearer_token:
            return self._generate_with_api_key(prompt, system_prompt, max_tokens, temperature)

        try:
            if self._client is None:
                raise RuntimeError("Bedrock runtime client is not initialized")
            response = self._client.converse(
                modelId=self.model_id,
                system=[{"text": system_prompt}],
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": prompt}],
                    }
                ],
                inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
            )
            return response["output"]["message"]["content"][0]["text"]
        except (BotoCoreError, ClientError, KeyError, IndexError):
            return self._generate_with_invoke_model(prompt, system_prompt, max_tokens, temperature)

    def generate_json(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int = 700,
        temperature: float = 0,
    ) -> dict[str, Any]:
        return _loads_json_object(self.generate_text(prompt, system_prompt, max_tokens, temperature))

    def _generate_with_invoke_model(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        if self._client is None:
            raise RuntimeError("Bedrock runtime client is not initialized")
        response = self._client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(
                {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                ensure_ascii=False,
            ),
        )
        payload = json.loads(response["body"].read())
        content = payload.get("generation") or payload.get("output") or payload.get("content")
        if isinstance(content, list):
            content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
        if not isinstance(content, str):
            content = json.dumps(payload, ensure_ascii=False)
        return content

    def _generate_with_api_key(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        encoded_model_id = quote(self.model_id, safe="")
        url = f"https://bedrock-runtime.{self.region_name}.amazonaws.com/model/{encoded_model_id}/converse"
        response = httpx.post(
            url,
            headers={
                "Authorization": f"Bearer {self.bearer_token}",
                "Content-Type": "application/json",
            },
            json={
                "system": [{"text": system_prompt}],
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt}],
                    }
                ],
                "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
            },
            timeout=20.0,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["output"]["message"]["content"][0]["text"]


def _loads_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM response did not contain a JSON object")
    return json.loads(text[start : end + 1])
