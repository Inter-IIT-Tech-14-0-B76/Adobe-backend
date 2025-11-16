# routers/images.py
from typing import Optional, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime, timezone, timedelta
import hashlib
import io

import anyio
from fastapi import APIRouter, Depends, File, UploadFile, Form, Query, Path, HTTPException, status
from fastapi.responses import JSONResponse
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

# local project imports — adjust paths if you keep models elsewhere
from utils.auth_helpers import verify_firebase_token, upsert_user_from_token
from utils.db import async_session
from utils.models import Image, User

# If you use boto3 (sync), we'll run it in a thread to avoid blocking the event loop.
import boto3
import os

S3_BUCKET = os.getenv("S3_BUCKET", "your-bucket")
S3_REGION = os.getenv("S3_REGION", "us-east-1")
_S3_CLIENT = boto3.client("s3", region_name=S3_REGION)

router = APIRouter(prefix="/images", tags=["images"])


# -----------------------
# Helpers
# -----------------------
def _compute_sha256(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def _s3_put_object_sync(object_key: str, data: bytes, mime_type: Optional[str] = None) -> Dict[str, Any]:
    extra = {}
    if mime_type:
        extra["ContentType"] = mime_type
    # example server-side encryption; change per your policy
    extra["ServerSideEncryption"] = "AES256"
    resp = _S3_CLIENT.put_object(Bucket=S3_BUCKET, Key=object_key, Body=data, **extra)
    return resp


def _s3_presign_sync(object_key: str, expires_in: int = 900) -> str:
    return _S3_CLIENT.generate_presigned_url(
        "get_object", Params={"Bucket": S3_BUCKET, "Key": object_key}, ExpiresIn=expires_in
    )


@router.post("/", status_code=status.HTTP_201_CREATED)
async def upload_image(
    # we require an authenticated user token; dependency returns decoded token payload
    token_payload: Dict = Depends(verify_firebase_token),
    project_id: UUID = Form(...),
    file: UploadFile = File(...),
    uploaded_at: Optional[datetime] = Form(None),
    retention_days: Optional[int] = Form(None),
    session: AsyncSession = Depends(async_session),
):
    """
    Uploads an image for a project. Authentication required.
    The uploader is derived from the Firebase token (upserted if needed).
    """
    # ensure user exists / upsert and get user object attached to async session
    uploader = await upsert_user_from_token(token_payload, session, set_last_login=True)
    if not uploader:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Could not resolve uploader user")

    contents = await file.read()
    if not contents:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty file")

    sha256_hash = _compute_sha256(contents)
    content_algo = "sha256"

    # dedupe check: unique by (project_id, sha256_hash) ignoring soft-deleted
    q = select(Image).where(
        Image.project_id == str(project_id),
        Image.sha256_hash == sha256_hash,
        Image.is_deleted == False,
    )
    result = await session.exec(q)
    existing = result.scalar_one_or_none()
    if existing:
        # idempotent: return existing metadata (200)
        return JSONResponse(status_code=200, content=existing.public_dict())

    # construct object key (do NOT store full URL)
    safe_fname = file.filename or "upload"
    object_key = f"project/{project_id}/{sha256_hash}/{safe_fname}"

    # upload to S3 (sync) inside a thread
    try:
        s3_resp = await anyio.to_thread.run_sync(_s3_put_object_sync, object_key, contents, file.content_type)
    except Exception as exc:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"S3 upload failed: {exc}")

    s3_version_id = s3_resp.get("VersionId") if isinstance(s3_resp, dict) else None

    width = None
    height = None

    retention_expires_at = None
    if retention_days is not None:
        retention_expires_at = datetime.now(timezone.utc) + timedelta(days=int(retention_days))

    image = Image(
        id=str(uuid4()),
        project_id=str(project_id),
        uploader_user_id=str(uploader.id),
        object_key=object_key,
        file_name=safe_fname,
        mime_type=file.content_type,
        width=width,
        height=height,
        size_bytes=len(contents),
        sha256_hash=sha256_hash,
        content_hash_algo=content_algo,
        exif_metadata_encrypted=None,
        s3_version_id=s3_version_id,
        is_deleted=False,
        retention_expires_at=retention_expires_at,
        sensitivity_flags=None,
        created_at=datetime.now(timezone.utc),
        uploaded_at=uploaded_at or datetime.now(timezone.utc),
    )

    session.add(image)
    await session.commit()
    await session.refresh(image)
    return JSONResponse(status_code=201, content=image.public_dict())


@router.get("/", status_code=200)
async def list_images(
    token_payload: Dict = Depends(verify_firebase_token),
    project_id: Optional[UUID] = Query(None),
    include_deleted: bool = Query(False),
    limit: int = Query(25, gt=0, le=200),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(async_session),
):
    """
    List images. Requires auth token (you can later restrict results by user/project membership).
    """
    # we simply validate token here; optionally you could upsert user as in upload
    stmt = select(Image)
    if project_id:
        stmt = stmt.where(Image.project_id == str(project_id))
    if not include_deleted:
        stmt = stmt.where(Image.is_deleted == False)

    total = await session.exec(stmt)
    total_count = total.count()
    rows = await session.exec(stmt.offset(offset).limit(limit))
    items = [r.public_dict() for r in rows.all()]
    return {"total": total_count, "items": items}


@router.get("/{image_id}", status_code=200)
async def get_image(
    token_payload: Dict = Depends(verify_firebase_token),
    image_id: UUID = Path(...),
    presign: bool = Query(False),
    presign_expires: int = Query(300, gt=0, le=3600),
    session: AsyncSession = Depends(async_session),
):
    """
    Get image metadata. If presign=true we return a presigned S3 URL (expires in presign_expires seconds).
    """
    q = select(Image).where(Image.id == str(image_id))
    result = await session.exec(q)
    image = result.scalar_one_or_none()
    if not image or image.is_deleted:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Image not found")

    data = image.public_dict()

    if presign:
        try:
            url = await anyio.to_thread.run_sync(_s3_presign_sync, image.object_key, presign_expires)
            data["presigned_url"] = url
        except Exception as exc:
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to create presign: {exc}")

    return JSONResponse(status_code=200, content=data)


@router.delete("/{image_id}", status_code=status.HTTP_204_NO_CONTENT)
async def soft_delete_image(
    token_payload: Dict = Depends(verify_firebase_token),
    image_id: UUID = Path(...),
    session: AsyncSession = Depends(async_session),
):
    """
    Soft-delete: only the uploader can delete their image (basic ownership check).
    Extend this to admins or project owners as needed.
    """
    # ensure uploader user exists (and to get user id)
    user = await upsert_user_from_token(token_payload, session, set_last_login=False)
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid user")

    q = select(Image).where(Image.id == str(image_id))
    result = await session.exec(q)
    image = result.scalar_one_or_none()
    if not image:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Image not found")

    # basic authorization: only uploader can delete
    if str(image.uploader_user_id) != str(user.id):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Not allowed to delete this image")

    image.is_deleted = True
    session.add(image)
    await session.commit()
    return JSONResponse(status_code=204, content=None)
