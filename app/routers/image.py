import json
from typing import Any, Dict, Optional
from uuid import uuid4

import anyio
from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from sqlmodel import desc, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.helpers.auth import upsert_user_from_token, verify_firebase_token
from app.helpers.s3 import (
    _compute_sha256,
    _s3_presign_sync,
    _s3_put_object_sync,
)
from app.utils.db import async_session
from app.utils.models import Image, ImageActionType, Project

image_router = APIRouter(tags=["Images"])


def _merge_transformations(
    parent_t: Dict[str, Any], delta_t: Dict[str, Any]
) -> Dict[str, Any]:
    merged = dict(parent_t)
    for k, v in delta_t.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _merge_transformations(merged[k], v)
        else:
            merged[k] = v
    return merged


@image_router.post(
    "/images/upload",
    status_code=201,
    summary="Upload an image and create a new project",
    description="Creates a new project and sets the uploaded image as the root + active image.",
)
async def upload_physical_image(
    token_payload: Dict = Depends(verify_firebase_token),
    project_id: Optional[str] = Form(None),
    parent_image_id: Optional[str] = Form(None),
    file: UploadFile = File(...),
    action_type: ImageActionType = Form(ImageActionType.UPLOAD),
    generation_params_str: Optional[str] = Form(None),
    session: AsyncSession = Depends(async_session),
):
    uploader = await upsert_user_from_token(token_payload, session, set_last_login=True)
    if not uploader:
        raise HTTPException(status_code=401, detail="User invalid")

    if parent_image_id:
        raise HTTPException(
            status_code=400,
            detail="Root uploads cannot set a parent_image_id.",
        )

    contents = await file.read()
    sha256_hash = _compute_sha256(contents)

    project = Project(
        id=str(uuid4()),
        user_id=uploader.id,
        name=file.filename or "Untitled Project",
    )
    session.add(project)
    await session.flush()

    object_key = f"project/{project.id}/{sha256_hash}/{file.filename or 'image'}"
    await anyio.to_thread.run_sync(
        _s3_put_object_sync, object_key, contents, file.content_type
    )

    gen_params = json.loads(generation_params_str) if generation_params_str else {}

    image = Image(
        id=str(uuid4()),
        project_id=project.id,
        object_key=object_key,
        file_name=file.filename,
        mime_type=file.content_type,
        sha256_hash=sha256_hash,
        parent_image_id=None,
        action_type=action_type,
        is_virtual=False,
        transformations={},
        generation_params=gen_params,
    )
    session.add(image)
    await session.flush()

    project.active_image_id = image.id
    session.add(project)

    await session.commit()
    await session.refresh(image)

    resp = image.public_dict()
    resp["project_id"] = str(project.id)
    return resp


@image_router.post(
    "/images/edit",
    status_code=201,
    summary="Create a virtual image edit",
    description="Adds a derived image that reuses the parent image file and stores transform metadata.",
)
async def create_virtual_edit(
    token_payload: Dict = Depends(verify_firebase_token),
    project_id: str = Body(...),
    parent_image_id: str = Body(...),
    transformations: Dict = Body(...),
    session: AsyncSession = Depends(async_session),
):
    uploader = await upsert_user_from_token(token_payload, session, set_last_login=True)
    if not uploader:
        raise HTTPException(status_code=401, detail="User invalid")

    project = await session.get(Project, project_id)
    if not project or project.user_id != uploader.id:
        raise HTTPException(status_code=403, detail="Unauthorized or project not found")

    parent = await session.get(Image, parent_image_id)
    if not parent or parent.project_id != project_id:
        raise HTTPException(status_code=404, detail="Invalid parent image")

    incoming = transformations or {}

    new_image = Image(
        id=str(uuid4()),
        project_id=project_id,
        parent_image_id=parent_image_id,
        object_key=parent.object_key,
        file_name=parent.file_name,
        mime_type=parent.mime_type,
        sha256_hash=parent.sha256_hash,
        transformations=incoming,
        action_type=ImageActionType.EDIT,
        is_virtual=True,
    )
    session.add(new_image)
    await session.flush()

    project.active_image_id = new_image.id
    session.add(project)

    await session.commit()
    await session.refresh(new_image)

    return new_image.public_dict()


@image_router.post(
    "/projects/{project_id}/undo",
    status_code=200,
    summary="Undo last edit",
    description="Resets active image pointer to the previous image.",
)
async def undo_action(
    project_id: str,
    token_payload: Dict = Depends(verify_firebase_token),
    session: AsyncSession = Depends(async_session),
):
    uploader = await upsert_user_from_token(token_payload, session)
    project = await session.get(Project, project_id)

    if not uploader or not project or project.user_id != uploader.id:
        raise HTTPException(status_code=403, detail="Unauthorized")

    current_img = await session.get(Image, project.active_image_id)
    if not current_img or not current_img.parent_image_id:
        raise HTTPException(status_code=400, detail="Nothing to undo")

    project.active_image_id = current_img.parent_image_id
    session.add(project)
    await session.commit()

    new_current = await session.get(Image, project.active_image_id)
    return new_current.public_dict()


@image_router.post(
    "/projects/{project_id}/redo",
    status_code=200,
    summary="Redo last undone edit",
    description="Moves active image pointer to the latest child image.",
)
async def redo_action(
    project_id: str,
    token_payload: Dict = Depends(verify_firebase_token),
    session: AsyncSession = Depends(async_session),
):
    uploader = await upsert_user_from_token(token_payload, session)
    project = await session.get(Project, project_id)

    if not uploader or not project or project.user_id != uploader.id:
        raise HTTPException(status_code=403, detail="Unauthorized")

    statement = (
        select(Image)
        .where(Image.parent_image_id == project.active_image_id)
        .order_by(desc(Image.created_at))
        .limit(1)
    )
    results = await session.execute(statement)
    child = results.scalar_one_or_none()
    if not child:
        raise HTTPException(status_code=400, detail="Nothing to redo")

    project.active_image_id = child.id
    session.add(project)
    await session.commit()
    await session.refresh(child)

    return child.public_dict()


@image_router.get(
    "/projects/{project_id}/current",
    status_code=200,
    summary="Get active image",
    description="Returns the active image for a project including a presigned S3 URL.",
)
async def get_current_project_state(
    project_id: str,
    token_payload: Dict = Depends(verify_firebase_token),
    session: AsyncSession = Depends(async_session),
):
    uploader = await upsert_user_from_token(token_payload, session)
    project = await session.get(Project, project_id)

    if not uploader or not project or project.user_id != uploader.id:
        raise HTTPException(status_code=403, detail="Unauthorized")

    if not project.active_image_id:
        return {"message": "No active image"}

    image = await session.get(Image, project.active_image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Active image missing")

    data = image.public_dict()
    data["presigned_url"] = _s3_presign_sync(image.object_key)
    return data


@image_router.get(
    "/projects",
    status_code=200,
    summary="List user's projects",
    description="Returns all projects owned by the authenticated user.",
)
async def list_user_projects(
    token_payload: Dict = Depends(verify_firebase_token),
    session: AsyncSession = Depends(async_session),
):
    uploader = await upsert_user_from_token(token_payload, session)
    if not uploader:
        raise HTTPException(status_code=401, detail="User invalid")

    statement = (
        select(Project)
        .where(Project.user_id == uploader.id)
        .order_by(desc(Project.created_at))
    )
    projects = (await session.execute(statement)).scalars().all()
    return [p.public_dict() for p in projects]


@image_router.get(
    "/projects/{project_id}",
    status_code=200,
    summary="Get project details",
    description="Returns full metadata for a specific project.",
)
async def get_project(
    project_id: str,
    token_payload: Dict = Depends(verify_firebase_token),
    session: AsyncSession = Depends(async_session),
):
    uploader = await upsert_user_from_token(token_payload, session)
    project = await session.get(Project, project_id)

    if not uploader or not project or project.user_id != uploader.id:
        raise HTTPException(status_code=403, detail="Unauthorized")

    return project.public_dict()


@image_router.delete(
    "/projects/{project_id}",
    status_code=200,
    summary="Delete a project",
    description="Deletes the project and all associated images.",
)
async def delete_project(
    project_id: str,
    token_payload: Dict = Depends(verify_firebase_token),
    session: AsyncSession = Depends(async_session),
):
    uploader = await upsert_user_from_token(token_payload, session)
    project = await session.get(Project, project_id)

    if not uploader or not project or project.user_id != uploader.id:
        raise HTTPException(status_code=403, detail="Unauthorized")

    await session.delete(project)
    await session.commit()
    return {"message": "Project deleted successfully"}


@image_router.get(
    "/projects/{project_id}/images",
    status_code=200,
    summary="Get all images in a project",
    description="Returns all images in a project ordered by creation time.",
)
async def get_project_images(
    project_id: str,
    token_payload: Dict = Depends(verify_firebase_token),
    session: AsyncSession = Depends(async_session),
):
    uploader = await upsert_user_from_token(token_payload, session)
    project = await session.get(Project, project_id)

    if not uploader or not project or project.user_id != uploader.id:
        raise HTTPException(status_code=403, detail="Unauthorized")

    statement = (
        select(Image).where(Image.project_id == project_id).order_by(Image.created_at)
    )
    images = (await session.execute(statement)).scalars().all()
    return [img.public_dict() for img in images]
