from __future__ import annotations

import os
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
from fastapi import (
    Path as FastAPIPath,
)
from fastapi.responses import JSONResponse
from sqlmodel.ext.asyncio.session import AsyncSession

from app.helpers.auth import upsert_user_from_token, verify_firebase_token
from app.helpers.comfy import ComfyError, process_image_with_comfy
from app.helpers.s3 import (
    _s3_get_object_bytes_sync,
    _s3_presign_sync,
    _s3_put_object_sync,
)
from app.utils.db import async_session
from app.utils.models import Image, ImageActionType, Project
from config import LOCAL_TMP_DIR

router = APIRouter()


def _ensure_tmp_dir() -> None:
    LOCAL_TMP_DIR.mkdir(parents=True, exist_ok=True)


def choose_framework(framework: Optional[str] = None) -> str:
    """
    Decide which CV framework to use.
    Currently only 'comfyui' is implemented, but this is the extension point
    for adding more later (e.g. 'opencv', 'torch-model', etc.).
    """
    if framework is None:
        return "comfyui"
    return framework.lower()


def choose_workspace(
    framework: str,
    workspace: Optional[str] = None,
) -> str:
    """
    Decide which workspace / pipeline to use, given a framework.

    For now this is a simple default, but it's written in a way that later you
    can route based on:
      - project metadata
      - user selection
      - image properties
      - etc.

    For 'comfyui', this maps to a workflow JSON name, e.g. 'default' => default.json.
    """
    if framework == "comfyui":
        # TODO: Replace this with smarter logic later (e.g. project-specific workspace)
        return workspace or "default"
    # Fallback for other frameworks (future)
    return workspace or "default"


@router.post(
    "/projects/{project_id}/cv/run",
    status_code=status.HTTP_201_CREATED,
)
async def run_cv_pipeline(
    project_id: str = FastAPIPath(..., description="ID of the project to process."),
    token_payload: Dict[str, Any] = Depends(verify_firebase_token),
    framework: Optional[str] = Body(
        None,
        embed=True,
        description="Which CV framework to use (default: comfyui).",
    ),
    workspace: Optional[str] = Body(
        None,
        embed=True,
        description="Workspace/pipeline identifier. Currently defaulted; will be dynamic later.",
    ),
    prompt: Optional[str] = Body(
        None,
        embed=True,
        description="Optional prompt for text-based CV/generation. Currently unused or fixed.",
    ),
    session: AsyncSession = Depends(async_session),
):
    """
    Run a CV pipeline on the project's active image:

    1. Auth user.
    2. Fetch project & ensure ownership.
    3. Fetch the project's active image.
    4. Download the image from S3.
    5. Save it locally (temporary file).
    6. Send it through the selected framework (currently ComfyUI).
    7. Upload the resulting image back to S3.
    8. Create a new Image node and mark it as the new active image.
    9. Return the resulting image metadata + presigned URL.
    """
    uploader = await upsert_user_from_token(token_payload, session, set_last_login=True)
    if not uploader:
        raise HTTPException(status_code=401, detail="User invalid")

    project = await session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.user_id != uploader.id:
        raise HTTPException(
            status_code=403, detail="Not authorized to process this project"
        )

    if not project.active_image_id:
        raise HTTPException(
            status_code=400, detail="Project has no active image to process"
        )

    active_image = await session.get(Image, project.active_image_id)
    if not active_image:
        raise HTTPException(status_code=404, detail="Active image not found")

    if not active_image.object_key:
        raise HTTPException(status_code=400, detail="Active image has no S3 object_key")

    try:
        image_bytes: bytes = await anyio.to_thread.run_sync(
            _s3_get_object_bytes_sync, active_image.object_key
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to download image from S3: {e}"
        )

    _ensure_tmp_dir()
    original_name = active_image.file_name or "input.png"
    _, ext = os.path.splitext(original_name)
    if not ext:
        ext = ".png"
    local_filename = f"{active_image.id}{ext}"
    local_path = LOCAL_TMP_DIR / local_filename

    try:
        local_path.write_bytes(image_bytes)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to write local temp file: {e}"
        )

    selected_framework = choose_framework(framework)
    selected_workspace = choose_workspace(selected_framework, workspace)

    effective_prompt = prompt or "Default CV processing prompt"

    try:
        if selected_framework == "comfyui":
            # process_image_with_comfy is synchronous; run it in a worker thread
            result_bytes: bytes = await anyio.to_thread.run_sync(
                process_image_with_comfy,
                str(local_path),
                selected_workspace,
                effective_prompt,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported framework: {selected_framework}",
            )
    except ComfyError as ce:
        raise HTTPException(
            status_code=502,
            detail=f"ComfyUI error while processing image: {ce}",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error while processing image: {e}",
        )

    sha256_hash = _compute_sha256(result_bytes)
    result_ext = ".png"  # Comfy typically outputs PNG; adjust if needed
    result_filename = f"cv_{uuid4().hex}{result_ext}"

    object_key_out = f"project/{project_id}/{sha256_hash}/{result_filename}"

    try:
        await anyio.to_thread.run_sync(
            _s3_put_object_sync,
            object_key_out,
            result_bytes,
            "image/png",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload processed image to S3: {e}",
        )

    # 9) Create new Image row
    #    This is a *physical* node since we generated a new file.
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

    project.active_image_id = new_image.id
    session.add(project)

    await session.commit()
    await session.refresh(new_image)

    resp = new_image.public_dict()
    resp["project_id"] = str(project.id)
    resp["presigned_url"] = _s3_presign_sync(new_image.object_key)

    return JSONResponse(status_code=201, content=resp)
