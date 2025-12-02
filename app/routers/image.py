import json
from typing import Dict, List, Optional
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
from sqlmodel import col, select, desc
from sqlmodel.ext.asyncio.session import AsyncSession

from app.helpers.auth import upsert_user_from_token, verify_firebase_token
from app.helpers.s3 import (
    _compute_sha256,
    _s3_presign_sync,
    _s3_put_object_sync,
)
from app.utils.db import async_session
from app.utils.models import (
    Image, 
    ImageActionType, 
    Project, 
    VersionHistory
)

async def _delete_future_versions(session: AsyncSession, parent_id: str):
    """
    Recursively finds and deletes all child versions (the future history)
    originating from the given parent_id.
    """
    statement = select(VersionHistory).where(VersionHistory.parent_id == parent_id)
    result = await session.execute(statement)
    children = result.scalars().all()

    for child in children:
        await _delete_future_versions(session, child.id)
        await session.delete(child)

image_router = APIRouter(tags=["Images and Versions"])

async def _fetch_images_by_ids(session: AsyncSession, image_ids: List[str]) -> List[Image]:
    """
    Since we store IDs in a JSON blob, we must manually query the Image table
    to get the actual objects. We rely on the order of IDs in the blob.
    """
    if not image_ids:
        return []
    
    statement = select(Image).where(col(Image.id).in_(image_ids))
    results = (await session.execute(statement)).scalars().all()
    
    image_map = {img.id: img for img in results}
    ordered_images = [image_map[uid] for uid in image_ids if uid in image_map]
    
    return ordered_images


@image_router.post(
    "/images/upload",
    status_code=201,
    summary="Start Project & Upload Root",
    description="Creates a project and the first version containing this single image."
)
async def upload_physical_image(
    token_payload: Dict = Depends(verify_firebase_token),
    project_id: Optional[str] = Form(None), 
    parent_version_id: Optional[str] = Form(None),
    file: UploadFile = File(...),
    action_type: ImageActionType = Form(ImageActionType.UPLOAD),
    prompt: Optional[str] = Form(None),
    generation_params_str: Optional[str] = Form(None),
    session: AsyncSession = Depends(async_session),
):
    uploader = await upsert_user_from_token(token_payload, session, set_last_login=True)
    if not uploader:
        raise HTTPException(status_code=401, detail="User invalid")

    contents = await file.read()
    sha256_hash = _compute_sha256(contents)
    gen_params = json.loads(generation_params_str) if generation_params_str else {}

    if project_id:
        project = await session.get(Project, project_id)
        if not project or project.user_id != uploader.id:
            raise HTTPException(status_code=403, detail="Unauthorized")
    else:
        project = Project(
            id=str(uuid4()),
            user_id=uploader.id,
            name=file.filename or "Untitled Project",
        )
        session.add(project)
        await session.flush() 

    object_key = f"project/{project.id}/assets/{sha256_hash}/{file.filename or 'image'}"
    
    await anyio.to_thread.run_sync(
        _s3_put_object_sync, object_key, contents, file.content_type
    )

    new_image = Image(
        id=str(uuid4()),
        project_id=project.id,
        object_key=object_key,
        file_name=file.filename,
        mime_type=file.content_type,
        sha256_hash=sha256_hash,
        action_type=action_type,
        is_virtual=False,
        transformations={},
        generation_params=gen_params,
    )
    session.add(new_image)
    await session.flush() 

    new_version = VersionHistory(
        id=str(uuid4()),
        project_id=project.id,
        parent_id=parent_version_id, 
        image_ids=[new_image.id],
        prompt=prompt,
        output_logs="Initial upload",
    )
    session.add(new_version)
    
    await session.flush() 

    project.current_version_id = new_version.id
    session.add(project)

    await session.commit()
    await session.refresh(new_version)

    resp = new_version.public_dict()
    resp["images"] = [new_image.public_dict()]
    resp["images"][0]["presigned_url"] = _s3_presign_sync(new_image.object_key)
    return resp


@image_router.post(
    "/images/upload/{project_id}",
    status_code=201,
    summary="Add image to project (New Version)",
    description="Adds a new image to the current sequence. Does NOT duplicate existing images, just appends the ID."
)
async def add_image_to_project(
    project_id: str,
    token_payload: Dict = Depends(verify_firebase_token),
    file: UploadFile = File(...),
    action_type: ImageActionType = Form(ImageActionType.UPLOAD),
    session: AsyncSession = Depends(async_session),
):
    uploader = await upsert_user_from_token(token_payload, session)
    project = await session.get(Project, project_id)
    if not project or project.user_id != uploader.id:
        raise HTTPException(status_code=403, detail="Unauthorized")

    parent_version = None
    previous_image_ids = []
    
    if project.current_version_id:
        parent_version = await session.get(VersionHistory, project.current_version_id)
        if parent_version:
            previous_image_ids = parent_version.image_ids.copy() 

    contents = await file.read()
    sha256_hash = _compute_sha256(contents)
    object_key = f"project/{project.id}/assets/{sha256_hash}/{file.filename}"
    
    await anyio.to_thread.run_sync(
        _s3_put_object_sync, object_key, contents, file.content_type
    )

    new_image = Image(
        id=str(uuid4()),
        project_id=project.id,
        object_key=object_key,
        file_name=file.filename,
        mime_type=file.content_type,
        sha256_hash=sha256_hash,
        action_type=action_type,
        is_virtual=False,
    )
    session.add(new_image)
    await session.flush() 

    new_state_ids = previous_image_ids + [new_image.id] 
    
    new_version = VersionHistory(
        id=str(uuid4()),
        project_id=project.id,
        parent_id=parent_version.id if parent_version else None,
        image_ids=new_state_ids, 
        prompt=parent_version.prompt if parent_version else None,
        output_logs="Added new image asset",
    )
    session.add(new_version)
    
    await session.flush()

    project.current_version_id = new_version.id
    session.add(project)
    
    await session.commit()

    full_images = await _fetch_images_by_ids(session, new_state_ids)
    
    resp = new_version.public_dict()
    resp["images"] = []
    for img in full_images:
        d = img.public_dict()
        d["presigned_url"] = _s3_presign_sync(img.object_key)
        resp["images"].append(d)
        
    return resp


@image_router.post(
    "/images/edit",
    status_code=201,
    summary="Create a new version (Edit)",
    description="Edits a specific version. WARNING: This action deletes all existing child versions (future history) of the selected version before creating the new one.",
)
async def create_virtual_edit(
    token_payload: Dict = Depends(verify_firebase_token),
    project_id: str = Body(...),
    version_id: str = Body(..., description="The ID of the version to edit"),
    transformations: Dict = Body(...),
    prompt: Optional[str] = Body(None),
    session: AsyncSession = Depends(async_session),
):
    uploader = await upsert_user_from_token(token_payload, session, set_last_login=True)
    project = await session.get(Project, project_id)
    if not project or project.user_id != uploader.id:
        raise HTTPException(status_code=403, detail="Unauthorized")

    parent_version = await session.get(VersionHistory, version_id)
    if not parent_version:
        raise HTTPException(status_code=404, detail="Version not found")
    if not parent_version.image_ids:
        raise HTTPException(status_code=400, detail="The selected version contains no images to edit")
    
    source_image_id = parent_version.image_ids[-1]

    source_image = await session.get(Image, source_image_id)
    if not source_image:
        raise HTTPException(status_code=404, detail="Source image asset missing")

    await _delete_future_versions(session, version_id)

    new_edited_image = Image(
        id=str(uuid4()),
        project_id=project_id,
        parent_image_id=source_image.id, 
        object_key=source_image.object_key, 
        file_name=source_image.file_name,
        mime_type=source_image.mime_type,
        sha256_hash=source_image.sha256_hash,
        transformations=transformations, 
        action_type=ImageActionType.EDIT,
        is_virtual=True,
    )
    session.add(new_edited_image)
    await session.flush() 

    new_image_ids = [
        uid if uid != source_image_id else new_edited_image.id 
        for uid in parent_version.image_ids
    ]

    new_version = VersionHistory(
        id=str(uuid4()),
        project_id=project_id,
        parent_id=version_id, 
        image_ids=new_image_ids, 
        prompt=prompt if prompt else parent_version.prompt,
        output_logs=f"Edited version {version_id}",
    )
    session.add(new_version)
    
    await session.flush() 
    
    project.current_version_id = new_version.id
    session.add(project)
    
    await session.commit()
    await session.refresh(new_version)

    full_images = await _fetch_images_by_ids(session, new_image_ids)
    
    resp = new_version.public_dict()
    resp["images"] = []
    for img in full_images:
        d = img.public_dict()
        d["presigned_url"] = _s3_presign_sync(img.object_key)
        resp["images"].append(d)
        
    return resp


@image_router.get(
    "/projects/{project_id}/current",
    status_code=200,
    summary="Get current state",
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

    if not project.current_version_id:
        return {"message": "No active history"}

    version = await session.get(VersionHistory, project.current_version_id)
    if not version:
        raise HTTPException(status_code=404, detail="Version data missing")

    images = await _fetch_images_by_ids(session, version.image_ids)

    data = version.public_dict()
    image_list = []
    for img in images:
        img_dict = img.public_dict()
        img_dict["presigned_url"] = _s3_presign_sync(img.object_key)
        image_list.append(img_dict)
        
    data["images"] = image_list
    return data


@image_router.post("/projects/{project_id}/undo", status_code=200)
async def undo_action(project_id: str, token_payload: Dict = Depends(verify_firebase_token), session: AsyncSession = Depends(async_session)):
    uploader = await upsert_user_from_token(token_payload, session)
    project = await session.get(Project, project_id)
    if not project or project.user_id != uploader.id: raise HTTPException(status_code=403, detail="Auth Error")

    current = await session.get(VersionHistory, project.current_version_id)
    if not current or not current.parent_id:
        raise HTTPException(status_code=400, detail="Cannot undo")
    
    project.current_version_id = current.parent_id
    session.add(project)
    await session.commit()

    parent = await session.get(VersionHistory, current.parent_id)
    images = await _fetch_images_by_ids(session, parent.image_ids)
    
    data = parent.public_dict()
    data["images"] = [i.public_dict() for i in images]
    return data

@image_router.post(
    "/projects/{project_id}/redo",
    status_code=200,
    summary="Redo (Go to latest child version)",
    description="Moves the project state forward to the most recently created child version."
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

    if not project.current_version_id:
        raise HTTPException(status_code=400, detail="Current version is null")

    statement = (
        select(VersionHistory)
        .where(VersionHistory.parent_id == project.current_version_id)
        .order_by(desc(VersionHistory.created_at))
        .limit(1)
    )
    results = await session.execute(statement)
    child_version = results.scalar_one_or_none()
    
    if not child_version:
        raise HTTPException(status_code=400, detail="Nothing to redo (no child version found)")

    project.current_version_id = child_version.id
    session.add(project)
    await session.commit()

    full_images = await _fetch_images_by_ids(session, child_version.image_ids)

    resp = child_version.public_dict()
    resp["images"] = []
    for img in full_images:
        d = img.public_dict()
        d["presigned_url"] = _s3_presign_sync(img.object_key)
        resp["images"].append(d)

    return resp

@image_router.delete(
    "/projects/{project_id}",
    status_code=200,
    summary="Delete a project",
    description="Deletes the project, its entire version history, and all image asset records.",
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
    return {"message": "Project and all history deleted successfully"}


@image_router.get(
    "/projects/{project_id}/versions",
    status_code=200,
    summary="Get all versions of a project",
    description="Returns a list of all history versions for a project, ordered by creation date (newest first)."
)
async def get_project_versions(
    project_id: str,
    token_payload: Dict = Depends(verify_firebase_token),
    session: AsyncSession = Depends(async_session),
):
    uploader = await upsert_user_from_token(token_payload, session)
    project = await session.get(Project, project_id)
    if not project or project.user_id != uploader.id:
        raise HTTPException(status_code=403, detail="Unauthorized")
    statement = (
        select(VersionHistory)
        .where(VersionHistory.project_id == project_id)
        .order_by(desc(VersionHistory.created_at))
    )
    versions = (await session.exec(statement)).all()
    return [v.public_dict() for v in versions]


@image_router.get(
    "/versions/{version_id}/images",
    status_code=200,
    summary="Get all images of a version",
    description="Returns the full list of image objects (with presigned URLs) for a specific version state."
)
async def get_version_images(
    version_id: str,
    token_payload: Dict = Depends(verify_firebase_token),
    session: AsyncSession = Depends(async_session),
):
    uploader = await upsert_user_from_token(token_payload, session)
    version = await session.get(VersionHistory, version_id)
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")
    project = await session.get(Project, version.project_id)
    if not project or project.user_id != uploader.id:
        raise HTTPException(status_code=403, detail="Unauthorized")

    full_images = await _fetch_images_by_ids(session, version.image_ids)
    image_list = []
    for img in full_images:
        d = img.public_dict()
        d["presigned_url"] = _s3_presign_sync(img.object_key)
        image_list.append(d)

    return image_list


@image_router.get(
    "/images/{image_id}",
    status_code=200,
    summary="Get image details",
    description="Returns the metadata and presigned URL for a specific image asset."
)
async def get_image_details(
    image_id: str,
    token_payload: Dict = Depends(verify_firebase_token),
    session: AsyncSession = Depends(async_session),
):
    uploader = await upsert_user_from_token(token_payload, session)
    image = await session.get(Image, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    project = await session.get(Project, image.project_id)
    if not project or project.user_id != uploader.id:
        raise HTTPException(status_code=403, detail="Unauthorized")

    resp = image.public_dict()
    resp["presigned_url"] = _s3_presign_sync(image.object_key)
    
    return resp

@image_router.get(
    "/users/{user_id}/projects",
    status_code=200,
    summary="List projects for a specific user",
    description="Returns a list of all projects belonging to the specified user ID. Enforces security so users can only view their own projects."
)
async def list_projects_by_user(
    user_id: str,
    token_payload: Dict = Depends(verify_firebase_token),
    session: AsyncSession = Depends(async_session),
):
    requester = await upsert_user_from_token(token_payload, session)
    if not requester:
        raise HTTPException(status_code=401, detail="User invalid")
    if requester.id != user_id:
        raise HTTPException(status_code=403, detail="Unauthorized to view projects for this user")

    statement = (
        select(Project)
        .where(Project.user_id == user_id)
        .order_by(desc(Project.created_at))
    )

    result = await session.execute(statement)
    projects = result.scalars().all()

    return [p.public_dict() for p in projects]

@image_router.get("/", summary="Root endpoint", response_model=dict)
async def root():
    """Root endpoint that returns a welcome message."""
    return {"message": "Go For Gold!!!"}