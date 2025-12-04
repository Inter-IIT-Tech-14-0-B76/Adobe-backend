"""
Module for 3D object visualization routers and endpoints.

This module provides API endpoints for a conversational 3D workflow:

1. Initial Image Submission (POST /surround/create)
   - User uploads 4 images
   - System generates initial video via todo_generate_video()
   - Returns video + persistent ID

2. Prompt-Based Update (POST /surround/{id}/prompt)
   - User sends prompt (optionally with new images)
   - System generates updated video via todo_generate_video_from_prompt()
   - Uses original images if no new ones provided

3. GLB File Retrieval (GET /surround/{id}/glb)
   - Generates and returns GLB file via todo_generate_glb()
   - Uses latest state associated with the ID

The ID is consistent across all operations for conversational context.
"""

from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

import anyio
from fastapi import APIRouter, Body, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.helpers.auth import upsert_user_from_token, verify_firebase_token
from app.helpers.surround import (
    todo_generate_video,
    todo_generate_video_from_prompt,
    todo_generate_glb,
)
from app.utils.db import async_session
from app.utils.models import Project, Project3D

surround_router = APIRouter(tags=["3D Visualization"])


# Temporary storage for uploaded images during processing
TEMP_UPLOAD_DIR = Path("/tmp/surround_uploads")


async def _save_uploaded_images(
    files: List[UploadFile],
    project_3d_id: str,
) -> List[str]:
    """
    Save uploaded images to temporary storage and return their paths.

    Args:
        files: List of uploaded image files.
        project_3d_id: The Project3D ID for organizing files.

    Returns:
        List of local file paths where images were saved.
    """
    TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    local_paths = []
    for idx, file in enumerate(files):
        contents = await file.read()
        filename = file.filename or f"image_{idx}"
        local_path = TEMP_UPLOAD_DIR / f"{project_3d_id}_{idx}_{filename}"

        def write_file(path=local_path, data=contents):
            path.write_bytes(data)

        await anyio.to_thread.run_sync(write_file)
        local_paths.append(str(local_path))

    return local_paths


async def _cleanup_temp_files(paths: List[str]) -> None:
    """
    Clean up temporary files after processing.

    Args:
        paths: List of file paths to delete.
    """
    for path in paths:
        try:
            path_obj = Path(path)
            if path_obj.exists():

                def delete_file(p=path_obj):
                    p.unlink()

                await anyio.to_thread.run_sync(delete_file)
        except Exception:
            pass


@surround_router.post(
    "/surround/create",
    status_code=201,
    summary="Create 3D project with initial images",
    description="""
    Step 1 of the conversational workflow: Initial Image Submission.

    Upload 4 images to create a new 3D project. The system will:
    1. Store the images
    2. Generate an initial video via todo_generate_video()
    3. Return the video path along with a persistent ID

    The returned ID should be used for all subsequent operations
    (prompt updates, GLB generation).
    """,
)
async def create_surround_project(
    token_payload: Dict = Depends(verify_firebase_token),
    files: List[UploadFile] = File(..., description="4 images for 3D"),
    p_id: Optional[str] = None,
    session: AsyncSession = Depends(async_session),
) -> Dict:
    """
    Create a new 3D project with 4 initial images.

    Args:
        token_payload: Firebase authentication token payload.
        files: List of exactly 4 uploaded image files.
        p_id: Optional parent project ID to associate with.
        session: Database session.

    Returns:
        Dict containing:
            - id: The persistent Project3D ID for future operations
            - video_path: Path to the generated initial video
            - images: List of stored image paths
            - generation_count: Number of generations (1)
            - project_3d: Full Project3D record

    Raises:
        HTTPException: If auth fails, wrong file count, or processing error.
    """
    # Authenticate user
    user = await upsert_user_from_token(
        token_payload, session, set_last_login=True
    )
    if not user:
        raise HTTPException(status_code=401, detail="User invalid")

    # Validate file count
    if len(files) != 4:
        raise HTTPException(
            status_code=400,
            detail=f"Exactly 4 images are required, received {len(files)}",
        )

    # Validate parent project if provided
    if p_id:
        project = await session.get(Project, p_id)
        if not project or project.user_id != user.id:
            raise HTTPException(
                status_code=403, detail="Unauthorized or project not found"
            )

    # Create the Project3D record first to get the ID
    project_3d = Project3D(
        id=str(uuid4()),
        p_id=p_id,
    )

    local_image_paths = []

    try:
        # Save uploaded images temporarily
        local_image_paths = await _save_uploaded_images(files, project_3d.id)

        # Store image paths in the model
        project_3d.set_images(local_image_paths)

        # Generate initial video using todo_generate_video
        result = await anyio.to_thread.run_sync(
            lambda: todo_generate_video(local_image_paths, project_3d.id)
        )

        if not result.get("success"):
            raise HTTPException(
                status_code=500,
                detail=f"Video generation failed: {result.get('error')}",
            )

        # Update the project with video info
        video_path = result.get("video_path")
        project_3d.add_to_history(prompt=None, video_path=video_path)

        # Save to database
        session.add(project_3d)
        await session.commit()
        await session.refresh(project_3d)

        return {
            "id": project_3d.id,
            "video_path": video_path,
            "images": project_3d.get_images(),
            "generation_count": project_3d.generation_count,
            "project_3d": project_3d.public_dict(),
            "message": "3D project created successfully with initial video",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create 3D project: {str(e)}",
        )


@surround_router.post(
    "/surround/{surround_id}/prompt",
    status_code=200,
    summary="Update video with prompt",
    description="""
    Step 2 of the conversational workflow: Prompt-Based Update.

    Send a text prompt (and optionally new images) to generate an updated video.
    The system will:
    1. Use the original images stored under this ID (or new images if provided)
    2. Apply the prompt via todo_generate_video_from_prompt()
    3. Return the new video, maintaining the same persistent ID

    If no new images are provided, the original 4 images are automatically used.
    """,
)
async def update_with_prompt(
    surround_id: str,
    token_payload: Dict = Depends(verify_firebase_token),
    prompt: str = Body(..., embed=True, description="Text prompt"),
    files: Optional[List[UploadFile]] = File(None, description="New images"),
    session: AsyncSession = Depends(async_session),
) -> Dict:
    """
    Update the video for a 3D project using a text prompt.

    Args:
        surround_id: The persistent Project3D ID.
        token_payload: Firebase authentication token payload.
        prompt: Text prompt describing desired changes.
        files: Optional new images to replace/augment originals.
        session: Database session.

    Returns:
        Dict containing:
            - id: The persistent Project3D ID
            - video_path: Path to the newly generated video
            - images_used: Images that were used for generation
            - prompt_applied: The prompt that was applied
            - generation_count: Updated generation count
            - project_3d: Updated Project3D record

    Raises:
        HTTPException: If ID not found, auth fails, or processing error.
    """
    # Authenticate user
    user = await upsert_user_from_token(
        token_payload, session, set_last_login=True
    )
    if not user:
        raise HTTPException(status_code=401, detail="User invalid")

    # Find the Project3D record
    project_3d = await session.get(Project3D, surround_id)
    if not project_3d:
        raise HTTPException(
            status_code=404,
            detail=f"3D project with ID {surround_id} not found",
        )

    # Verify ownership if linked to a project
    if project_3d.p_id:
        project = await session.get(Project, project_3d.p_id)
        if project and project.user_id != user.id:
            raise HTTPException(status_code=403, detail="Unauthorized")

    new_image_paths = []

    try:
        # If new images provided, save them
        if files and len(files) > 0:
            new_image_paths = await _save_uploaded_images(files, project_3d.id)
            # Update stored images if new ones provided
            project_3d.set_images(new_image_paths)

        # Get the images to use (new or original)
        original_images = project_3d.get_images()

        # Generate updated video using todo_generate_video_from_prompt
        result = await anyio.to_thread.run_sync(
            lambda: todo_generate_video_from_prompt(
                original_images=original_images,
                prompt=prompt,
                new_images=new_image_paths if new_image_paths else None,
                project_id=project_3d.id,
                generation_number=int(project_3d.generation_count),
            )
        )

        if not result.get("success"):
            raise HTTPException(
                status_code=500,
                detail=f"Video generation failed: {result.get('error')}",
            )

        # Update the project with new video info
        video_path = result.get("video_path")
        project_3d.add_to_history(prompt=prompt, video_path=video_path)

        # Save to database
        session.add(project_3d)
        await session.commit()
        await session.refresh(project_3d)

        return {
            "id": project_3d.id,
            "video_path": video_path,
            "images_used": result.get("images_used"),
            "prompt_applied": result.get("prompt_applied"),
            "generation_count": int(project_3d.generation_count),
            "project_3d": project_3d.public_dict(),
            "message": "Video updated successfully with prompt",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update video: {str(e)}",
        )


@surround_router.get(
    "/surround/{surround_id}/glb",
    summary="Get GLB file for project",
    description="""
    Step 3 of the conversational workflow: GLB File Retrieval.

    Generate and retrieve the GLB (3D model) file for a project.
    Uses the latest state (images + prompt) associated with the ID
    via todo_generate_glb().

    The ID must match a previously created 3D project.
    """,
    response_class=FileResponse,
)
async def get_glb_file(
    surround_id: str,
    token_payload: Dict = Depends(verify_firebase_token),
    session: AsyncSession = Depends(async_session),
):
    """
    Get or generate the GLB file for a 3D project.

    Args:
        surround_id: The persistent Project3D ID.
        token_payload: Firebase authentication token payload.
        session: Database session.

    Returns:
        FileResponse: The GLB file as a binary download.

    Raises:
        HTTPException: If ID not found, auth fails, or generation error.
    """
    # Authenticate user
    user = await upsert_user_from_token(
        token_payload, session, set_last_login=True
    )
    if not user:
        raise HTTPException(status_code=401, detail="User invalid")

    # Find the Project3D record
    project_3d = await session.get(Project3D, surround_id)
    if not project_3d:
        raise HTTPException(
            status_code=404,
            detail=f"3D project with ID {surround_id} not found",
        )

    # Verify ownership if linked to a project
    if project_3d.p_id:
        project = await session.get(Project, project_3d.p_id)
        if project and project.user_id != user.id:
            raise HTTPException(status_code=403, detail="Unauthorized")

    # Check if GLB already exists
    if project_3d.glb_file_path:
        glb_path = Path(project_3d.glb_file_path)
        if glb_path.exists():
            return FileResponse(
                path=str(glb_path),
                filename=f"{surround_id}.glb",
                media_type="model/gltf-binary",
            )

    # Generate GLB if it doesn't exist
    image_paths = project_3d.get_images()
    if not image_paths:
        raise HTTPException(
            status_code=400,
            detail="No images available for GLB generation",
        )

    # Generate GLB using todo_generate_glb
    result = await anyio.to_thread.run_sync(
        lambda: todo_generate_glb(
            project_id=project_3d.id,
            image_paths=image_paths,
            latest_prompt=project_3d.latest_prompt,
        )
    )

    if not result.get("success"):
        raise HTTPException(
            status_code=500,
            detail=f"GLB generation failed: {result.get('error')}",
        )

    # Update the project with GLB path
    glb_path = result.get("glb_path")
    project_3d.glb_file_path = glb_path
    session.add(project_3d)
    await session.commit()

    return FileResponse(
        path=glb_path,
        filename=f"{surround_id}.glb",
        media_type="model/gltf-binary",
    )


@surround_router.get(
    "/surround/{surround_id}",
    summary="Get 3D project details",
    description="""
    Retrieve the full details of a 3D project including:
    - Current images
    - All generated videos (history)
    - All prompts used (history)
    - GLB file path (if generated)
    - Generation count
    """,
)
async def get_surround_project(
    surround_id: str,
    token_payload: Dict = Depends(verify_firebase_token),
    session: AsyncSession = Depends(async_session),
) -> Dict:
    """
    Get full details of a 3D project.

    Args:
        surround_id: The persistent Project3D ID.
        token_payload: Firebase authentication token payload.
        session: Database session.

    Returns:
        Dict containing the full Project3D record.

    Raises:
        HTTPException: If ID not found or authentication fails.
    """
    # Authenticate user
    user = await upsert_user_from_token(
        token_payload, session, set_last_login=True
    )
    if not user:
        raise HTTPException(status_code=401, detail="User invalid")

    # Find the Project3D record
    project_3d = await session.get(Project3D, surround_id)
    if not project_3d:
        raise HTTPException(
            status_code=404,
            detail=f"3D project with ID {surround_id} not found",
        )

    # Verify ownership if linked to a project
    if project_3d.p_id:
        project = await session.get(Project, project_3d.p_id)
        if project and project.user_id != user.id:
            raise HTTPException(status_code=403, detail="Unauthorized")

    return {
        "id": project_3d.id,
        "project_3d": project_3d.public_dict(),
    }


@surround_router.get(
    "/surround/{surround_id}/video",
    summary="Get latest video file",
    description="Download the latest generated video for a 3D project.",
    response_class=FileResponse,
)
async def get_latest_video(
    surround_id: str,
    token_payload: Dict = Depends(verify_firebase_token),
    session: AsyncSession = Depends(async_session),
):
    """
    Get the latest video file for a 3D project.

    Args:
        surround_id: The persistent Project3D ID.
        token_payload: Firebase authentication token payload.
        session: Database session.

    Returns:
        FileResponse: The video file as a binary download.

    Raises:
        HTTPException: If ID not found, no video, or auth fails.
    """
    # Authenticate user
    user = await upsert_user_from_token(
        token_payload, session, set_last_login=True
    )
    if not user:
        raise HTTPException(status_code=401, detail="User invalid")

    # Find the Project3D record
    project_3d = await session.get(Project3D, surround_id)
    if not project_3d:
        raise HTTPException(
            status_code=404,
            detail=f"3D project with ID {surround_id} not found",
        )

    # Verify ownership if linked to a project
    if project_3d.p_id:
        project = await session.get(Project, project_3d.p_id)
        if project and project.user_id != user.id:
            raise HTTPException(status_code=403, detail="Unauthorized")

    if not project_3d.demo_video_path:
        raise HTTPException(
            status_code=404,
            detail="No video has been generated for this project",
        )

    video_path = Path(project_3d.demo_video_path)
    if not video_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Video file not found on disk",
        )

    return FileResponse(
        path=str(video_path),
        filename=f"{surround_id}_video.mp4",
        media_type="video/mp4",
    )


# Legacy endpoints for backward compatibility


@surround_router.post(
    "/surround/generate-video",
    status_code=201,
    summary="[Legacy] Generate video from images",
    description="Legacy endpoint. Use POST /surround/create instead.",
)
async def generate_video_from_images(
    token_payload: Dict = Depends(verify_firebase_token),
    project_id: Optional[str] = None,
    files: List[UploadFile] = File(...),
    session: AsyncSession = Depends(async_session),
) -> Dict:
    """Legacy endpoint - redirects to create_surround_project."""
    # For backward compatibility, just call the new endpoint logic
    return await create_surround_project(
        token_payload=token_payload,
        files=files,
        p_id=project_id,
        session=session,
    )


@surround_router.get(
    "/surround/glb/{project_id}",
    summary="[Legacy] Get GLB by project ID",
    description="Legacy endpoint. Use GET /surround/{id}/glb instead.",
    response_class=FileResponse,
)
async def get_glb_by_project_id(
    project_id: str,
    token_payload: Dict = Depends(verify_firebase_token),
    session: AsyncSession = Depends(async_session),
):
    """
    Legacy endpoint - get GLB by parent project ID.

    Finds the Project3D associated with the given project ID.
    """
    # Authenticate user
    user = await upsert_user_from_token(
        token_payload, session, set_last_login=True
    )
    if not user:
        raise HTTPException(status_code=401, detail="User invalid")

    # Verify project ownership
    project = await session.get(Project, project_id)
    if not project or project.user_id != user.id:
        raise HTTPException(status_code=403, detail="Unauthorized")

    # Find Project3D by p_id
    statement = select(Project3D).where(Project3D.p_id == project_id)
    result = await session.execute(statement)
    project_3d = result.scalar_one_or_none()

    if not project_3d:
        raise HTTPException(
            status_code=404,
            detail="No 3D data found for this project",
        )

    # Delegate to the main endpoint
    return await get_glb_file(
        surround_id=project_3d.id,
        token_payload=token_payload,
        session=session,
    )
