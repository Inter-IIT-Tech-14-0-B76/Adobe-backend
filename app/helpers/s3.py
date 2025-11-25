from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional

import boto3

from config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    S3_BUCKET,
    S3_REGION,
)

_S3_CLIENT = boto3.client(
    "s3",
    region_name=S3_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)


def _compute_sha256(b: bytes) -> str:
    """Compute SHA256 hash of bytes"""
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def _s3_get_object_bytes_sync(object_key: str) -> bytes:
    """Download object from S3 synchronously and return bytes."""
    resp = _S3_CLIENT.get_object(Bucket=S3_BUCKET, Key=object_key)
    return resp["Body"].read()


def _s3_put_object_sync(
    object_key: str, data: bytes, mime_type: Optional[str] = None
) -> Dict[str, Any]:
    """Upload object to S3 synchronously."""
    extra: Dict[str, Any] = {}
    if mime_type:
        extra["ContentType"] = mime_type
    extra["ServerSideEncryption"] = "AES256"

    resp = _S3_CLIENT.put_object(Bucket=S3_BUCKET, Key=object_key, Body=data, **extra)
    return resp


def _s3_presign_sync(object_key: str, expires_in: int = 900) -> str:
    return _S3_CLIENT.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": object_key},
        ExpiresIn=expires_in,
    )
