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
        # Explicit static credentials for local dev (MinIO, always set). When
        # unset (staging — no static key is issued, by design: a long-lived
        # key is exactly what an EC2 instance profile exists to avoid), omit
        # them so boto3 falls back to its default credential chain, which
        # resolves the EC2 instance profile via IMDS automatically (ADR-0009 —
        # config only, no adapter redesign). endpoint_url/region are unchanged
        # either way: staging sets AWS_ENDPOINT to the real regional S3
        # endpoint via config, same as it always pointed at MinIO in dev.
        client_kwargs = {
            "endpoint_url": settings.aws_endpoint,
            "region_name": settings.aws_region,
            "config": Config(s3={"addressing_style": "path" if settings.aws_use_path_style else "auto"}),
        }
        if settings.aws_access_key_id and settings.aws_secret_access_key:
            client_kwargs["aws_access_key_id"] = settings.aws_access_key_id
            client_kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
        _client = boto3.client("s3", **client_kwargs)
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
