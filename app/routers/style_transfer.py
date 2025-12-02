"""
Style Transfer API Router
Provides endpoints for AI-powered style transfer operations.
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Literal
import os
import sys
import tempfile
import shutil
import json
from pathlib import Path
import asyncio
import time

router = APIRouter(prefix="/api/style-transfer", tags=["Style Transfer"])

WORKSPACE_DIR = Path("/workspace/AIP/workspace")
OUTPUTS_DIR = WORKSPACE_DIR / "outputs"
VENV_DIR = Path("/workspace/AIP/.venv")

if os.name == "nt":
    PYTHON_EXEC = VENV_DIR / "Scripts" / "python.exe"
else:
    PYTHON_EXEC = VENV_DIR / "bin" / "python"

TASK_SCRIPTS = {
    "style_with_prompt": WORKSPACE_DIR / "tasks" / "style_transfer_text.py",
    "style_with_ref": WORKSPACE_DIR / "tasks" / "style_transfer_ref.py"
}

class TaskStatusResponse(BaseModel):
    """Response model for task status"""
    status: Literal["success", "error", "processing"]
    message: str
    task_id: Optional[str] = None
    output_paths: Optional[dict] = None
    metadata: Optional[dict] = None


async def save_upload_file(upload_file: UploadFile, destination: Path) -> Path:
    """Save an uploaded file to a destination path."""
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return destination
    finally:
        upload_file.file.close()

async def run_style_transfer_pipeline(
    task_key: str,
    content_path: Path,
    style_path: Optional[Path] = None,
    style_text: Optional[str] = None,
    prompt: str = "",
    negative_prompt: str = "",
    steps: int = 50,
    style_steps: int = 25,
    max_side: int = 1024
) -> dict:
    """
    Constructs the command and runs the style transfer script via subprocess.
    """
    output_dir_images = OUTPUTS_DIR / "images"
    output_dir_data = OUTPUTS_DIR / "data"
    output_dir_images.mkdir(parents=True, exist_ok=True)
    output_dir_data.mkdir(parents=True, exist_ok=True)
    
    script_path = TASK_SCRIPTS.get(task_key)
    if not script_path or not script_path.exists():
        raise FileNotFoundError(f"Task script not found: {script_path}")
    
    if not PYTHON_EXEC.exists():
         print(f"Warning: Venv python not found at {PYTHON_EXEC}, using system python.")
         python_cmd = sys.executable
    else:
         python_cmd = str(PYTHON_EXEC)

    cmd = [python_cmd, str(script_path)]
    
    if task_key == "style_with_prompt":
        cmd.extend([
            "--content", str(content_path),
            "--style_text", style_text or "",
            "--prompt", prompt,
            "--output_dir", str(output_dir_images),
            "--negative_prompt", negative_prompt,
            "--steps", str(steps),
            "--style_steps", str(style_steps),
            "--max_side", str(max_side)
        ])
    elif task_key == "style_with_ref":
        if not style_path:
            raise ValueError("style_path is required for style_with_ref task")
        cmd.extend([
            "--content", str(content_path),
            "--style", str(style_path),
            "--prompt", prompt,
            "--output_dir", str(output_dir_images),
            "--negative_prompt", negative_prompt,
            "--steps", str(steps),
            "--max_side", str(max_side)
        ])
    
    print(f"\n{'='*70}")
    print(f"Running Style Transfer: {task_key}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(WORKSPACE_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        stdout_text = stdout.decode() if stdout else ""
        stderr_text = stderr.decode() if stderr else ""
        
        if process.returncode != 0:
            print(f"Error output:\n{stderr_text}")
            raise RuntimeError(f"Pipeline execution failed: {stderr_text}")
        
        result = {
            "status": "success",
            "message": "Style transfer completed successfully",
            "stdout": stdout_text,
            "output_dir_images": str(output_dir_images),
        }
        
        if output_dir_images.exists():
            json_files = sorted(
                output_dir_images.glob(f"style_transfer_*.json"),
                key=lambda f: f.stat().st_mtime,
                reverse=True
            )
            
            if json_files:
                with open(json_files[0], 'r') as f:
                    metadata = json.load(f)
                    result["metadata"] = metadata
                    result["output_paths"] = {
                        "composite": metadata.get("output_composite"),
                        "styled_only": metadata.get("output_styled_only"),
                        "metadata_file": str(json_files[0])
                    }
                    if task_key == "style_with_prompt":
                        result["output_paths"]["generated_style_reference"] = metadata.get("generated_style_reference")
        
        return result
        
    except Exception as e:
        raise RuntimeError(f"Pipeline execution error: {str(e)}")


@router.post("/style-with-text", response_model=TaskStatusResponse)
async def style_transfer_with_text(
    content_image: UploadFile = File(..., description="Content image"),
    style_text: str = Form(..., description="Style description"),
    prompt: str = Form(..., description="Generation prompt"),
    negative_prompt: str = Form("", description="Negative prompt"),
    steps: int = Form(50, ge=10, le=150),
    style_steps: int = Form(25, ge=10, le=100),
    max_side: int = Form(1024, ge=512, le=2048)
):
    temp_dir = None
    try:
        temp_dir = Path(tempfile.mkdtemp(prefix="style_transfer_"))
        content_path = temp_dir / f"content_{int(time.time())}_{content_image.filename}"
        await save_upload_file(content_image, content_path)
        
        result = await run_style_transfer_pipeline(
            task_key="style_with_prompt",
            content_path=content_path,
            style_text=style_text,
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            style_steps=style_steps,
            max_side=max_side
        )
        
        return TaskStatusResponse(
            status="success",
            message="Completed successfully",
            output_paths=result.get("output_paths"),
            metadata=result
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


@router.post("/style-with-reference", response_model=TaskStatusResponse)
async def style_transfer_with_reference(
    content_image: UploadFile = File(..., description="Content image"),
    style_image: UploadFile = File(..., description="Style reference image"),
    prompt: str = Form(..., description="Generation prompt"),
    negative_prompt: str = Form("", description="Negative prompt"),
    steps: int = Form(50, ge=10, le=150),
    max_side: int = Form(1024, ge=512, le=2048)
):
    temp_dir = None
    try:
        temp_dir = Path(tempfile.mkdtemp(prefix="style_transfer_"))
        content_path = temp_dir / f"content_{int(time.time())}_{content_image.filename}"
        style_path = temp_dir / f"style_{int(time.time())}_{style_image.filename}"
        
        await save_upload_file(content_image, content_path)
        await save_upload_file(style_image, style_path)
        
        result = await run_style_transfer_pipeline(
            task_key="style_with_ref",
            content_path=content_path,
            style_path=style_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            max_side=max_side
        )
        
        return TaskStatusResponse(
            status="success",
            message="Completed successfully",
            output_paths=result.get("output_paths"),
            metadata=result
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

from fastapi.responses import FileResponse

@router.get("/get-style-image/{filename}")
async def get_style_image(filename: str):
    """
    Get a generated style image by filename.
    
    Args:
        filename: The name of the style image file to retrieve
        
    Returns:
        The image file if found, or 404 if not found
    """
    image_path = WORKSPACE_DIR / "outputs" / "images" / filename
    
    if not image_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Image {filename} not found"
        )
    
    return FileResponse(
        image_path,
        media_type="image/png",
        filename=filename
    )

from fastapi.responses import Response
from fastapi import status

@router.post("/echo-image", response_class=Response, status_code=status.HTTP_200_OK)
async def echo_image(
    image: UploadFile = File(..., description="Image to be echoed back")
):
    contents = await image.read()
    return Response(
        content=contents,
        media_type=image.content_type,
        headers={"Content-Disposition": f"inline; filename={image.filename}"}
    )