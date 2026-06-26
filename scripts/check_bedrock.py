#!/usr/bin/env python
import sys
from pathlib import Path

import httpx
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.services.llm import BedrockLlamaClient


def has_aws_credential_settings() -> bool:
    settings = get_settings()
    return bool(
        settings.aws_profile
        or (settings.aws_access_key_id and settings.aws_secret_access_key)
    )


def main() -> None:
    settings = get_settings()
    auth_mode = "bedrock_api_key" if settings.bedrock_api_key else "aws_credentials"
    print(f"auth_mode={auth_mode}")

    if not settings.bedrock_api_key and not has_aws_credential_settings():
        raise SystemExit(
            "No Bedrock credentials found. "
            "Put BEDROCK_API_KEY=... in .env, not .env.example."
        )

    client = BedrockLlamaClient(settings)
    try:
        answer = client.generate_text(
            prompt="한국어로 'Bedrock 연결 확인 완료'라고만 답해줘.",
            system_prompt="You are a terse connectivity checker.",
            max_tokens=30,
            temperature=0,
        )
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        if status_code in {401, 403}:
            raise SystemExit(
                f"Bedrock reached AWS, but authorization failed ({status_code}). "
                "Check API key validity, Bedrock model access, region, and permission for "
                f"{settings.bedrock_model_id} in {settings.aws_region}."
            ) from exc
        raise SystemExit(f"Bedrock HTTP request failed ({status_code}). Check AWS Bedrock status and request settings.") from exc
    except httpx.HTTPError as exc:
        raise SystemExit(
            "Could not reach Bedrock endpoint. Check network/VPN/DNS and AWS region "
            f"({settings.aws_region})."
        ) from exc
    except NoCredentialsError as exc:
        raise SystemExit("AWS credentials were not found. Put BEDROCK_API_KEY=... in .env or configure AWS credentials.") from exc
    except (BotoCoreError, ClientError) as exc:
        raise SystemExit(f"AWS Bedrock SDK call failed: {exc}") from exc
    print(answer)


if __name__ == "__main__":
    main()
