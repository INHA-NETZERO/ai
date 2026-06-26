from typing import Any

import boto3

from app.core.config import Settings


def create_aws_client(service_name: str, settings: Settings) -> Any:
    return create_aws_session(settings).client(service_name)


def create_aws_session(settings: Settings) -> boto3.Session:
    session_kwargs: dict[str, Any] = {"region_name": settings.aws_region}
    if settings.aws_profile:
        session_kwargs["profile_name"] = settings.aws_profile
    elif settings.aws_access_key_id and settings.aws_secret_access_key:
        session_kwargs["aws_access_key_id"] = settings.aws_access_key_id
        session_kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
        if settings.aws_session_token:
            session_kwargs["aws_session_token"] = settings.aws_session_token
    return boto3.Session(**session_kwargs)
