#!/usr/bin/env python
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings


def main() -> None:
    settings = get_settings()
    bearer_token = settings.aws_bearer_token_bedrock or settings.bedrock_api_key
    if bearer_token and not os.getenv("AWS_BEARER_TOKEN_BEDROCK"):
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = bearer_token

    region_name = os.getenv("AWS_REGION") or settings.aws_region or "us-east-1"
    model_id = os.getenv("BEDROCK_MODEL_ID") or settings.bedrock_model_id or "meta.llama3-2-1b-instruct-v1:0"

    print(f"region={region_name}")
    print(f"model_id={model_id}")
    print(f"has_bedrock_bearer_token={bool(os.getenv('AWS_BEARER_TOKEN_BEDROCK'))}")

    client = boto3.client("bedrock-runtime", region_name=region_name)
    response = client.converse(
        modelId=model_id,
        messages=[
            {
                "role": "user",
                "content": [{"text": "한국어로 'Bedrock Llama 3.2 1B 연결 확인 완료'라고만 답해줘."}],
            }
        ],
    )

    print(response["output"]["message"]["content"][0]["text"])


if __name__ == "__main__":
    try:
        main()
    except ClientError as exc:
        status_code = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        error_code = exc.response.get("Error", {}).get("Code", "Unknown")
        raise SystemExit(f"Bedrock ClientError: {status_code or error_code}: {error_code}") from exc
    except NoCredentialsError as exc:
        raise SystemExit("AWS_BEARER_TOKEN_BEDROCK is not set, or your boto3/botocore cannot read it.") from exc
    except BotoCoreError as exc:
        raise SystemExit(f"Bedrock BotoCoreError: {exc}") from exc
