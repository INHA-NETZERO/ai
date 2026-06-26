import json
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError


class BedrockLlamaClient:
    def __init__(self, region_name: str, model_id: str) -> None:
        self.model_id = model_id
        self._client = boto3.client("bedrock-runtime", region_name=region_name)

    def generate_text(
        self,
        prompt: str,
        system_prompt: str = "You are a concise assistant for supply-chain and carbon operations.",
        max_tokens: int = 700,
        temperature: float = 0,
    ) -> str:
        try:
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


def _loads_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM response did not contain a JSON object")
    return json.loads(text[start : end + 1])
