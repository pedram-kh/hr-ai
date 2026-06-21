"""S3-compatible object storage access for hr-ai.

hr-ai READS the uploaded original and WRITES rendered page images. This is
object storage only — hr-ai never writes the database and never migrates
(ADR-0007, ADR-0010). The same bucket is used by hr-backend.
"""

import boto3
from botocore.config import Config

from .config import settings

_client = None


def s3_client():
    global _client
    if _client is None:
        _client = boto3.client(
            "s3",
            endpoint_url=settings.aws_endpoint,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region,
            config=Config(s3={"addressing_style": "path" if settings.aws_use_path_style else "auto"}),
        )
    return _client


def get_object_bytes(key: str) -> bytes:
    resp = s3_client().get_object(Bucket=settings.aws_bucket, Key=key)
    return resp["Body"].read()


def put_object_bytes(key: str, data: bytes, content_type: str) -> None:
    s3_client().put_object(
        Bucket=settings.aws_bucket,
        Key=key,
        Body=data,
        ContentType=content_type,
    )
