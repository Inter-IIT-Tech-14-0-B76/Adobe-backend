from __future__ import annotations

import hashlib
import os
import tempfile
from typing import Any, Dict, Optional
from uuid import uuid4

import anyio
from fastapi import (
    APIRouter,
    Body,
    Depends,
    HTTPException,
    status,
)
from fastapi import Path as FastAPIPath
from fastapi.responses import JSONResponse
from sqlmodel.ext.asyncio.session import AsyncSession

from app.helpers.auth import get_image, verify_firebase_token
from app.helpers.comfy import (
    ComfyError,
    inject_image_filename,
    inject_prompt_text,
    load_workflow_template,
    queue_prompt,
    upload_image_to_comfy,
    wait_for_images,
)
from app.helpers.s3 import (
    _s3_get_object_bytes_sync,
    _s3_presign_sync,
    _s3_put_object_sync,
)
from app.utils.db import async_session
from app.utils.models import Image, ImageActionType
from config import DEFAULT_WORKSPACE

router = APIRouter()


def _compute_sha256(b: bytes) -> str:
    """Compute SHA256 hash of bytes."""
    return hashlib.sha256(b).hexdigest()


def choose_framework(framework: Optional[str] = None) -> str:
    """Normalize and default the CV framework selection."""
    if framework is None:
        return "comfyui"
    return framework.lower()


def choose_workspace(framework: str, workspace: Optional[str] = None) -> str:
    """Select workspace based on framework and optional override."""
    if framework == "comfyui":
        return workspace or DEFAULT_WORKSPACE or "default"
    return workspace or "default"


def _comfy_workflow_worker(
    image_bytes: bytes,
    upload_filename: str,
    workspace_name: str,
    prompt_text: str,
    timeout_seconds: float = 60.0,
) -> bytes:
    """
    Execute ComfyUI workflow synchronously (for use in thread pool).

    1. Write bytes to a temporary file
    2. Upload to ComfyUI
    3. Load and configure workflow
    4. Queue and wait for result
    5. Return first output image bytes
    """
    tmp_path = None
    try:
        # Create temporary file with appropriate extension
        _, ext = os.path.splitext(upload_filename)
        with tempfile.NamedTemporaryFile(
            prefix="comfy_upload_",
            suffix=ext or ".png",
            delete=False,
        ) as tf:
            tmp_path = tf.name
            tf.write(image_bytes)
            tf.flush()

        # Upload to ComfyUI
        upload_info = upload_image_to_comfy(tmp_path)
        image_name = upload_info.get("name")
        if not image_name:
            raise ComfyError(f"ComfyUI upload response missing 'name': {upload_info}")

        # Load and configure workflow
        workflow = load_workflow_template(workspace_name)
        inject_image_filename(workflow, [image_name])

        if prompt_text:
            inject_prompt_text(workflow, prompt_text)

        # Execute workflow
        prompt_id = queue_prompt(workflow)
        images = wait_for_images(prompt_id, timeout_seconds=timeout_seconds)

        if not images:
            raise ComfyError(
                f"No images returned from ComfyUI for prompt_id={prompt_id}"
            )

        return images[0]

    finally:
        # Clean up temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass  # Swallow cleanup errors


@router.post("/projects/{project_id}/cv/run", status_code=status.HTTP_201_CREATED)
async def run_cv_pipeline(
    project_id: str = FastAPIPath(..., description="ID of the project to process."),
    token_payload: Dict[str, Any] = Depends(verify_firebase_token),
    framework: Optional[str] = Body(
        None, embed=True, description="Which CV framework to use (default: comfyui)."
    ),
    workspace: Optional[str] = Body(
        None,
        embed=True,
        description="Workspace/pipeline identifier for workflow selection.",
    ),
    prompt: Optional[str] = Body(
        None,
        embed=True,
        description="Optional prompt for text-based CV/generation.",
    ),
    session: AsyncSession = Depends(async_session),
):
    """
    Run a CV pipeline on the project's active image using ComfyUI.

    This endpoint orchestrates:
    1. Fetching the active image from S3
    2. Processing through ComfyUI workflow
    3. Storing result back to S3
    4. Creating new Image record linked to original
    """
    # Get and validate project/image
    print(f"input params: {project_id}, {token_payload}, {session}")
    uploader, active_image, project = await get_image(
        project_id, token_payload, session
    )

    # Download source image from S3
    try:
        image_bytes: bytes = await anyio.to_thread.run_sync(
            _s3_get_object_bytes_sync, active_image.object_key
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to download image from S3: {e}"
        )

    # Prepare workflow parameters
    original_name = active_image.file_name or "input.png"
    selected_framework = choose_framework(framework)
    # selected_workspace = choose_workspace(selected_framework, workspace)
    selected_workspace = "check"
    effective_prompt = prompt or ""

    if selected_framework != "comfyui":
        raise HTTPException(
            status_code=400, detail=f"Unsupported framework: {selected_framework}"
        )

    # Run ComfyUI workflow in thread pool
    try:
        result_bytes: bytes = await anyio.to_thread.run_sync(
            _comfy_workflow_worker,
            image_bytes,
            original_name,
            selected_workspace,
            effective_prompt,
            60.0,
        )
    except ComfyError as ce:
        raise HTTPException(status_code=502, detail=f"ComfyUI error: {ce}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Unexpected error processing image: {e}"
        )

    # Store result to S3
    sha256_hash = _compute_sha256(result_bytes)
    result_filename = f"cv_{uuid4().hex}.png"
    object_key_out = f"project/{project_id}/{sha256_hash}/{result_filename}"

    try:
        await anyio.to_thread.run_sync(
            _s3_put_object_sync, object_key_out, result_bytes, "image/png"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to upload processed image to S3: {e}"
        )

    # Create new Image record
    gen_params: Dict[str, Any] = {
        "framework": selected_framework,
        "workspace": selected_workspace,
        "prompt": effective_prompt,
    }

    new_image = Image(
        id=str(uuid4()),
        project_id=str(project_id),
        parent_image_id=str(active_image.id),
        object_key=object_key_out,
        file_name=result_filename,
        mime_type="image/png",
        sha256_hash=sha256_hash,
        action_type=ImageActionType.EDIT,
        is_virtual=False,
        transformations={},
        generation_params=gen_params,
    )

    session.add(new_image)
    await session.flush()

    # Update project's active image
    project.active_image_id = new_image.id
    session.add(project)

    await session.commit()
    await session.refresh(new_image)

    # Build response
    resp = new_image.public_dict()
    resp["project_id"] = str(project.id)
    resp["presigned_url"] = _s3_presign_sync(new_image.object_key)

    return JSONResponse(status_code=201, content=resp)
