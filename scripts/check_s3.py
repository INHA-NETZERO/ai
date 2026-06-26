#!/usr/bin/env python
import argparse
import csv
import io
import sys
from pathlib import Path

import httpx
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.services.demo_data import load_closing_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Check S3 connectivity for Net-Zero AI Server.")
    parser.add_argument(
        "--url",
        help="Optional S3 presigned GET URL. This path does not require AWS credentials.",
    )
    args = parser.parse_args()

    if args.url:
        check_presigned_url(args.url)
        return

    check_direct_s3()


def check_presigned_url(url: str) -> None:
    try:
        response = httpx.get(url, timeout=10.0)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise SystemExit(
            f"Presigned URL reached S3 but failed with HTTP {exc.response.status_code}. "
            "Check whether the URL expired or lacks GetObject permission."
        ) from exc
    except httpx.HTTPError as exc:
        raise SystemExit("Could not download presigned URL. Check network, URL, and expiration.") from exc

    rows, columns = _parse_csv(response.text)
    print("mode=presigned_url")
    print(f"columns={','.join(columns)}")
    print(f"rows={len(rows)}")


def check_direct_s3() -> None:
    settings = get_settings().model_copy(update={"data_source": "s3"})
    if not settings.s3_bucket:
        raise SystemExit(
            "S3_BUCKET is not set. For direct S3 mode, put S3_BUCKET, S3_PREFIX, "
            "S3_*_KEY, and AWS credentials/profile/role in .env. "
            "If Spring gives a presigned URL, use: scripts/check_s3.py --url '<url>'"
        )

    try:
        data = load_closing_data(settings)
    except NoCredentialsError as exc:
        raise SystemExit(
            "AWS credentials were not found for direct S3. "
            "Set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, AWS_PROFILE, or run on an IAM role. "
            "AWS_BEARER_TOKEN_BEDROCK is only for Bedrock and cannot read S3."
        ) from exc
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "Unknown")
        raise SystemExit(
            f"S3 request failed with {code}. Check bucket, key, region, and s3:GetObject permission."
        ) from exc
    except (BotoCoreError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Direct S3 check failed: {exc}") from exc

    print("mode=direct_s3")
    print(f"bucket={settings.s3_bucket}")
    print(f"prefix={settings.s3_prefix}")
    print(f"business_date={data['business_date']}")
    print(f"inventory_flow_rows={len(data['inventory_flow'])}")
    print(f"item_master_rows={len(data['item_master'])}")
    print(f"order_policy_rows={len(data['order_policy'])}")
    print(f"data_version={data['data_version']}")


def _parse_csv(content: str) -> tuple[list[dict[str, str]], list[str]]:
    reader = csv.DictReader(io.StringIO(content.lstrip("\ufeff")))
    rows = list(reader)
    return rows, list(reader.fieldnames or [])


if __name__ == "__main__":
    main()
