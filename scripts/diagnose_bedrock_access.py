#!/usr/bin/env python
import argparse
import base64
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings


DEFAULT_MODELS = [
    "meta.llama3-2-1b-instruct-v1:0",
    "us.meta.llama3-2-1b-instruct-v1:0",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Bedrock bearer token and model access without printing secrets.")
    parser.add_argument("--model-id", action="append", default=[], help="Model ID to test. Repeatable.")
    args = parser.parse_args()

    settings = get_settings()
    bearer_token = settings.aws_bearer_token_bedrock or settings.bedrock_api_key
    if bearer_token and not os.getenv("AWS_BEARER_TOKEN_BEDROCK"):
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = bearer_token

    region_name = os.getenv("AWS_REGION") or settings.aws_region or "us-east-1"
    models = args.model_id or DEFAULT_MODELS

    print(f"region={region_name}")
    print(f"has_bedrock_bearer_token={bool(os.getenv('AWS_BEARER_TOKEN_BEDROCK'))}")
    _print_token_window(os.getenv("AWS_BEARER_TOKEN_BEDROCK", ""))

    _check_control_plane(region_name)

    client = boto3.client("bedrock-runtime", region_name=region_name)
    for model_id in models:
        print(f"\nmodel_id={model_id}")
        try:
            response = client.converse(
                modelId=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": "Reply with only: OK"}],
                    }
                ],
            )
            text = response["output"]["message"]["content"][0]["text"]
            print(f"result=OK text={text[:80]}")
        except ClientError as exc:
            status_code = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            error_code = exc.response.get("Error", {}).get("Code", "Unknown")
            print(f"result=ERROR status={status_code} code={error_code}")
        except NoCredentialsError:
            print("result=ERROR code=NoCredentialsError")
        except BotoCoreError as exc:
            print(f"result=ERROR code=BotoCoreError message={exc}")


def _print_token_window(token: str) -> None:
    token_payload = _decode_bedrock_api_key_payload(token)
    if not token_payload:
        print("token_window=unavailable")
        return

    parsed = urlparse(token_payload)
    params = parse_qs(parsed.query)
    date_value = _first(params, "X-Amz-Date")
    expires_value = _first(params, "X-Amz-Expires")
    if not date_value or not expires_value:
        print("token_window=unavailable")
        return

    issued_at = datetime.strptime(date_value, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    expires_at = issued_at + timedelta(seconds=int(expires_value))
    now = datetime.now(UTC)
    remaining = expires_at - now
    remaining_minutes = int(remaining.total_seconds() // 60)
    print(f"token_issued_at_utc={issued_at.isoformat()}")
    print(f"token_expires_at_utc={expires_at.isoformat()}")
    print(f"token_remaining_minutes={remaining_minutes}")
    print(f"token_expired={remaining.total_seconds() <= 0}")


def _check_control_plane(region_name: str) -> None:
    try:
        client = boto3.client("bedrock", region_name=region_name)
        client.list_foundation_models()
        print("bedrock_control_plane=OK")
    except ClientError as exc:
        status_code = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        error_code = exc.response.get("Error", {}).get("Code", "Unknown")
        message = exc.response.get("Error", {}).get("Message", "")
        print(f"bedrock_control_plane=ERROR status={status_code} code={error_code}")
        if "bedrock:CallWithBearerToken" in message:
            print("missing_permission=bedrock:CallWithBearerToken")
            print("diagnosis=Add bedrock:CallWithBearerToken to the IAM principal that created the Bedrock API key.")
    except (NoCredentialsError, BotoCoreError) as exc:
        print(f"bedrock_control_plane=ERROR code={exc.__class__.__name__}")


def _decode_bedrock_api_key_payload(token: str) -> str:
    prefix = "bedrock-api-key-"
    if not token.startswith(prefix):
        return ""
    encoded = token[len(prefix) :]
    padding = "=" * (-len(encoded) % 4)
    try:
        return unquote(base64.b64decode(encoded + padding).decode("utf-8"))
    except Exception:
        return ""


def _first(params: dict[str, list[str]], key: str) -> str:
    values = params.get(key) or []
    return values[0] if values else ""


if __name__ == "__main__":
    main()
