"""
Style Transfer API Router
Provides endpoints for AI-powered style transfer operations.
Located at: app/routers/style_transfer.py

The API automatically handles:
- Virtual environment creation for the specific task
- Installing task-specific requirements
- Running the style transfer pipeline
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Literal
import os
import sys
import subprocess
import tempfile
import shutil
import json
from pathlib import Path
import asyncio
import time

router = APIRouter(prefix="/api/style-transfer", tags=["Style Transfer"])


WORKSPACE_DIR = Path(__file__).parent.parent.parent / "workspace"
OUTPUTS_DIR = WORKSPACE_DIR / "outputs"


TASK_CONFIG = {
    "style_with_prompt": {
        "venv_dir": WORKSPACE_DIR / ".venv_style_transfer",
        "requirements": WORKSPACE_DIR / "requirements_style_transfer.txt",
        "script": WORKSPACE_DIR / "tasks" / "style_transfer_text.py",
        "task_name": "style_with_prompt"
    },
    "style_with_ref": {
        "venv_dir": WORKSPACE_DIR / ".venv_style_transfer",
        "requirements": WORKSPACE_DIR / "requirements_style_transfer.txt",
        "script": WORKSPACE_DIR / "tasks" / "style_transfer_ref.py",
        "task_name": "style_with_ref"
    }
}


class TaskStatusResponse(BaseModel):
    """Response model for task status"""
    status: Literal["success", "error", "processing"]
    message: str
    task_id: Optional[str] = None
    output_paths: Optional[dict] = None
    metadata: Optional[dict] = None


def get_python_executable(venv_dir: Path) -> Path:
    """Get the python executable path for the given venv."""
    if os.name == "nt":  # Windows
        return venv_dir / "Scripts" / "python.exe"
    else:  # Linux/Mac
        return venv_dir / "bin" / "python"


def get_pip_executable(venv_dir: Path) -> Path:
    """Get the pip executable path for the given venv."""
    if os.name == "nt":  # Windows
        return venv_dir / "Scripts" / "pip.exe"
    else:  # Linux/Mac
        return venv_dir / "bin" / "pip"


async def ensure_task_environment(task_key: str) -> Path:
    """
    Ensure the virtual environment for a specific task is ready.
    Creates venv and installs requirements if needed.
    
    Args:
        task_key: Key from TASK_CONFIG
        
    Returns:
        Path to python executable in the venv
    """
    config = TASK_CONFIG[task_key]
    venv_dir = config["venv_dir"]
    requirements_file = config["requirements"]
    
    python_exe = get_python_executable(venv_dir)
    pip_exe = get_pip_executable(venv_dir)
    
    if not python_exe.exists():
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m", "venv",
                str(venv_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise RuntimeError(f"Failed to create venv: {error_msg}")
            
            print("✓ Virtual environment created")
            
        except Exception as e:
            raise RuntimeError(f"Failed to create virtual environment: {str(e)}")
    else:
        print(f"✓ Virtual environment already exists at: {venv_dir}")
    
    print("\nUpgrading pip, wheel, setuptools...")
    try:
        process = await asyncio.create_subprocess_exec(
            str(python_exe),
            "-m", "pip", "install",
            "--upgrade", "pip", "wheel", "setuptools",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        await process.communicate()
        print("✓ Core packages upgraded")
        
    except Exception as e:
        print(f"⚠️  Warning: Failed to upgrade pip: {str(e)}")
    
    if requirements_file.exists():
        print(f"\nInstalling requirements from: {requirements_file}")
        
        try:
            process = await asyncio.create_subprocess_exec(
                str(pip_exe),
                "install", "-r", str(requirements_file),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                print(f"⚠️  Warning during requirements installation: {error_msg}")
            else:
                print("✓ Requirements installed successfully")
                
        except Exception as e:
            print(f"⚠️  Warning: Failed to install requirements: {str(e)}")
    else:
        print(f"⚠️  Warning: Requirements file not found: {requirements_file}")
    
    print(f"{'='*70}\n")
    
    return python_exe


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
    Run the complete style transfer pipeline.
    
    This function:
    1. Sets up the virtual environment
    2. Installs required dependencies
    3. Runs the style transfer script
    4. Returns the results
    
    Args:
        task_key: Task configuration key
        content_path: Path to content image
        style_path: Path to style reference image (for style_with_ref)
        style_text: Text description of style (for style_with_prompt)
        prompt: Generation prompt
        negative_prompt: Negative prompt
        steps: Inference steps
        style_steps: Style generation steps
        max_side: Maximum image side length
        
    Returns:
        dict: Results including output paths and metadata
    """
    python_exe = await ensure_task_environment(task_key)
    
    output_dir_images = OUTPUTS_DIR / "images"
    output_dir_data = OUTPUTS_DIR / "data"
    output_dir_images.mkdir(parents=True, exist_ok=True)
    output_dir_data.mkdir(parents=True, exist_ok=True)
    
    config = TASK_CONFIG[task_key]
    script_path = config["script"]
    
    if not script_path.exists():
        raise FileNotFoundError(f"Task script not found: {script_path}")
    
    cmd = [str(python_exe), str(script_path)]
    
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
    print(f"Running Style Transfer Pipeline: {task_key}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")
    
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
            raise RuntimeError(f"Pipeline execution failed with code {process.returncode}: {stderr_text}")
        
        print(f"\n{'='*70}")
        print(f"✓ Pipeline completed successfully!")
        print(f"{'='*70}\n")
        
        result = {
            "status": "success",
            "message": "Style transfer completed successfully",
            "stdout": stdout_text,
            "output_dir_images": str(output_dir_images),
            "output_dir_data": str(output_dir_data),
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
    content_image: UploadFile = File(..., description="Content image to be styled"),
    style_text: str = Form(..., description="Text description of desired style (e.g., 'oil painting, impressionist style')"),
    prompt: str = Form(..., description="Generation prompt for final pass"),
    negative_prompt: str = Form("", description="Negative prompt for generation"),
    steps: int = Form(50, ge=10, le=150, description="Inference steps for final pass"),
    style_steps: int = Form(25, ge=10, le=100, description="Steps for generating style reference"),
    max_side: int = Form(1024, ge=512, le=2048, description="Max side length for image resize")
):
    """
    Apply style transfer using a text description to generate the style reference.
    
    **Pipeline Process:**
    1. Automatically creates/updates virtual environment with required dependencies
    2. Generates a style reference image from your text description
    3. Applies the style to your content image using ControlNet and IP-Adapter
    4. Returns styled image with original background preserved
    
    **Example Parameters:**
    - style_text: "oil painting, impressionist style, vibrant colors"
    - prompt: "a beautiful portrait in artistic style"
    - steps: 50 (higher = better quality, slower)
    - style_steps: 25 (steps for generating style reference)
    
    **Returns:**
    - Composite image (styled object on original background)
    - Styled-only image (fully styled)
    - Generated style reference image
    - Metadata JSON file
    """
    temp_dir = None
    
    try:
        temp_dir = Path(tempfile.mkdtemp(prefix="style_transfer_"))
        
        content_path = temp_dir / f"content_{int(time.time())}_{content_image.filename}"
        await save_upload_file(content_image, content_path)
        
        print(f"Processing style transfer with text for: {content_image.filename}")
        print(f"Style description: {style_text}")
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
            message="Style transfer with text completed successfully",
            output_paths=result.get("output_paths"),
            metadata=result
        )
        
    except Exception as e:
        print(f"Error in style_transfer_with_text: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Style transfer failed: {str(e)}"
        )
    finally:
        if temp_dir and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Failed to cleanup temp directory: {str(e)}")


@router.post("/style-with-reference", response_model=TaskStatusResponse)
async def style_transfer_with_reference(
    content_image: UploadFile = File(..., description="Content image to be styled"),
    style_image: UploadFile = File(..., description="Style reference image"),
    prompt: str = Form(..., description="Generation prompt"),
    negative_prompt: str = Form("", description="Negative prompt for generation"),
    steps: int = Form(50, ge=10, le=150, description="Inference steps"),
    max_side: int = Form(1024, ge=512, le=2048, description="Max side length for image resize")
):
    """
    Apply style transfer using a reference image as the style source.
    
    **Pipeline Process:**
    1. Automatically creates/updates virtual environment with required dependencies
    2. Uses your provided style reference image
    3. Applies the style to your content image using ControlNet and IP-Adapter
    4. Returns styled image with original background preserved
    
    **Example Parameters:**
    - content_image: Your photo/image to be styled
    - style_image: An example image with the desired artistic style
    - prompt: "portrait in the style of the reference"
    - steps: 50 (higher = better quality, slower)
    
    **Returns:**
    - Composite image (styled object on original background)
    - Styled-only image (fully styled)
    - Metadata JSON file
    """
    temp_dir = None
    
    try:
        temp_dir = Path(tempfile.mkdtemp(prefix="style_transfer_"))
        
        content_path = temp_dir / f"content_{int(time.time())}_{content_image.filename}"
        style_path = temp_dir / f"style_{int(time.time())}_{style_image.filename}"
        
        await save_upload_file(content_image, content_path)
        await save_upload_file(style_image, style_path)
        
        print(f"Processing style transfer with reference for: {content_image.filename}")
        print(f"Style reference: {style_image.filename}")
        
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
            message="Style transfer with reference completed successfully",
            output_paths=result.get("output_paths"),
            metadata=result
        )
        
    except Exception as e:
        print(f"Error in style_transfer_with_reference: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Style transfer failed: {str(e)}"
        )
    finally:
        if temp_dir and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Failed to cleanup temp directory: {str(e)}")


@router.get("/outputs")
async def list_outputs():
    """
    List all output files from previous style transfer operations.
    
    Returns:
    - List of generated images (composite and styled-only versions)
    - List of metadata JSON files
    - List of generated style reference images (for text-based transfers)
    """
    try:
        outputs_images = OUTPUTS_DIR / "images"
        
        result = {
            "images": [],
            "metadata": [],
            "style_references": []
        }
        
        if outputs_images.exists():
            for pattern in ["style_transfer_text_*.png", "style_transfer_ref_*.png"]:
                result["images"].extend([
                    {
                        "path": str(f.relative_to(WORKSPACE_DIR)),
                        "filename": f.name,
                        "created": f.stat().st_mtime
                    }
                    for f in outputs_images.glob(pattern)
                ])
            
            result["metadata"] = [
                {
                    "path": str(f.relative_to(WORKSPACE_DIR)),
                    "filename": f.name,
                    "created": f.stat().st_mtime
                }
                for f in outputs_images.glob("style_transfer_*.json")
            ]
            
            result["style_references"] = [
                {
                    "path": str(f.relative_to(WORKSPACE_DIR)),
                    "filename": f.name,
                    "created": f.stat().st_mtime
                }
                for f in outputs_images.glob("style_transfer_text_generated_style_*.png")
            ]
            
            for key in result:
                result[key] = sorted(result[key], key=lambda x: x["created"], reverse=True)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list outputs: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    Verifies that the workspace and required directories exist.
    """
    try:
        workspace_exists = WORKSPACE_DIR.exists()
        
        return {
            "status": "healthy" if workspace_exists else "unhealthy",
            "workspace_dir": str(WORKSPACE_DIR),
            "workspace_exists": workspace_exists,
            "outputs_dir": str(OUTPUTS_DIR)
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }