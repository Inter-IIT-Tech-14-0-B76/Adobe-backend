"""
Helper functions for 3D object visualization and surround view processing.

This module provides utility functions for the conversational 3D workflow:
- todo_generate_video: Initial video generation from 4 images via /infer API
- todo_generate_video_from_prompt: Update video based on prompt and images
- todo_generate_glb: Generate GLB file via /convert API

Storage locations:
- S3: Videos and GLB files are uploaded to S3 for frontend access
- Local temp: /workspace/backend/demo_videos/ (temporary before S3 upload)

External APIs:
- INFER_HOST: Generates MP4 video from 4 input images
- CONVERT_HOST: Converts to GLB format using the job key
- WORKSPACE_SERVER: Style transfer endpoints for prompt-based styling
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from app.helpers.s3 import (
    _s3_put_object_sync,
    _s3_presign_sync,
)
from config import WORKSPACE_SERVER

# Storage directories for 3D assets (local temp storage)
GLB_STORAGE_DIR = Path("/workspace/backend/glb")
DEMO_VIDEO_STORAGE_DIR = Path("/workspace/backend/demo_videos")

# External API endpoints for 3D generation
INFER_HOST = os.getenv(
    "LGM_INFER_URL", "https://unpj0fg9si7yet-7860.proxy.runpod.net/infer"
)
CONVERT_HOST = os.getenv(
    "LGM_CONVERT_URL", "https://unpj0fg9si7yet-8888.proxy.runpod.net/convert"
)

# Timeout settings (in seconds)
INFER_TIMEOUT = 1200  # 20 minutes for video generation
CONVERT_TIMEOUT = 1800  # 30 minutes for GLB conversion
STYLE_TRANSFER_TIMEOUT = 120  # 2 minutes per image


class SurroundProcessingError(Exception):
    """Exception raised for errors during 3D/surround processing."""

    pass


def _upload_video_to_s3(
    local_path: str,
    project_id: str,
    generation_number: int = 1,
) -> Dict[str, Any]:
    """
    Upload a video file to S3 and return the object key and presigned URL.

    Args:
        local_path: Local path to the video file.
        project_id: Project ID for organizing in S3.
        generation_number: Generation number for the video.

    Returns:
        Dict containing:
            - success: Boolean indicating if upload succeeded
            - object_key: S3 object key
            - presigned_url: Presigned URL for frontend access
            - error: Error message if failed
    """
    try:
        path_obj = Path(local_path)
        if not path_obj.exists():
            return {
                "success": False,
                "object_key": None,
                "presigned_url": None,
                "error": f"Local video file not found: {local_path}",
            }

        video_bytes = path_obj.read_bytes()
        filename = path_obj.name
        object_key = f"surround/{project_id}/videos/gen{generation_number}_{filename}"

        print(f"[S3] Uploading video to {object_key}")
        _s3_put_object_sync(object_key, video_bytes, "video/mp4")

        presigned_url = _s3_presign_sync(object_key, expires_in=3600)
        print(f"[S3] Video uploaded successfully: {object_key}")

        return {
            "success": True,
            "object_key": object_key,
            "presigned_url": presigned_url,
            "error": None,
        }

    except Exception as e:
        error_msg = f"S3 upload failed: {str(e)}"
        print(f"[S3] {error_msg}")
        return {
            "success": False,
            "object_key": None,
            "presigned_url": None,
            "error": error_msg,
        }


def _upload_glb_to_s3(
    local_path: str,
    project_id: str,
) -> Dict[str, Any]:
    """
    Upload a GLB file to S3 and return the object key and presigned URL.

    Args:
        local_path: Local path to the GLB file.
        project_id: Project ID for organizing in S3.

    Returns:
        Dict containing:
            - success: Boolean indicating if upload succeeded
            - object_key: S3 object key
            - presigned_url: Presigned URL for frontend access
            - error: Error message if failed
    """
    try:
        path_obj = Path(local_path)
        if not path_obj.exists():
            return {
                "success": False,
                "object_key": None,
                "presigned_url": None,
                "error": f"Local GLB file not found: {local_path}",
            }

        glb_bytes = path_obj.read_bytes()
        filename = path_obj.name
        object_key = f"surround/{project_id}/glb/{filename}"

        print(f"[S3] Uploading GLB to {object_key}")
        _s3_put_object_sync(object_key, glb_bytes, "model/gltf-binary")

        presigned_url = _s3_presign_sync(object_key, expires_in=3600)
        print(f"[S3] GLB uploaded successfully: {object_key}")

        return {
            "success": True,
            "object_key": object_key,
            "presigned_url": presigned_url,
            "error": None,
        }

    except Exception as e:
        error_msg = f"S3 GLB upload failed: {str(e)}"
        print(f"[S3] {error_msg}")
        return {
            "success": False,
            "object_key": None,
            "presigned_url": None,
            "error": error_msg,
        }


def _apply_style_transfer_to_image(
    image_path: str,
    prompt: str,
    output_dir: str,
) -> Dict[str, Any]:
    """
    Apply style transfer to a single image using the style-transfer/text endpoint.

    Args:
        image_path: Path to the input image.
        prompt: Style prompt to apply.
        output_dir: Directory for output files.

    Returns:
        Dict containing:
            - success: Boolean
            - output_path: Path to styled image
            - error: Error message if failed
    """
    try:
        url = f"{WORKSPACE_SERVER}/style-transfer/text"
        payload = {
            "content": image_path,
            "style_text": prompt,
            "prompt": prompt,
            "output_dir": output_dir,
            "steps": 40,
            "style_steps": 20,
            "max_side": 1024,
        }

        print(f"[STYLE] POST {url}")
        print(f"[STYLE] Image: {image_path}, Prompt: {prompt[:50]}...")

        response = requests.post(url, json=payload, timeout=STYLE_TRANSFER_TIMEOUT)

        if response.status_code != 200:
            error_msg = f"Style transfer failed: {response.status_code}"
            try:
                error_detail = response.json()
                error_msg += f": {error_detail}"
            except Exception:
                error_msg += f": {response.text[:200]}"
            return {
                "success": False,
                "output_path": None,
                "error": error_msg,
            }

        result = response.json()

        # Extract output path from response
        output_paths = result.get("output_paths", {})
        output_path = (
            output_paths.get("composite")
            or output_paths.get("styled_only")
            or result.get("output_composite")
            or result.get("output_styled_only")
        )

        if not output_path:
            return {
                "success": False,
                "output_path": None,
                "error": "Style transfer did not return output path",
            }

        print(f"[STYLE] Output: {output_path}")
        return {
            "success": True,
            "output_path": output_path,
            "error": None,
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "output_path": None,
            "error": f"Style transfer timed out after {STYLE_TRANSFER_TIMEOUT}s",
        }
    except Exception as e:
        return {
            "success": False,
            "output_path": None,
            "error": f"Style transfer error: {str(e)}",
        }


def apply_style_transfer_to_images(
    image_paths: List[str],
    prompt: str,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Apply style transfer to all images using the given prompt.

    Args:
        image_paths: List of paths to input images.
        prompt: Style prompt to apply to all images.
        output_dir: Optional output directory.

    Returns:
        Dict containing:
            - success: Boolean
            - styled_paths: List of paths to styled images
            - error: Error message if any image failed
    """
    if output_dir is None:
        output_dir = str(DEMO_VIDEO_STORAGE_DIR / "styled")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    styled_paths = []
    print(f"[STYLE] Applying style transfer to {len(image_paths)} images")
    print(f"[STYLE] Prompt: {prompt}")

    for i, img_path in enumerate(image_paths):
        print(f"[STYLE] Processing image {i + 1}/{len(image_paths)}: {img_path}")

        result = _apply_style_transfer_to_image(
            image_path=img_path,
            prompt=prompt,
            output_dir=output_dir,
        )

        if not result["success"]:
            return {
                "success": False,
                "styled_paths": styled_paths,
                "error": f"Image {i + 1} failed: {result['error']}",
            }

        styled_paths.append(result["output_path"])

    print(f"[STYLE] All {len(styled_paths)} images styled successfully")
    return {
        "success": True,
        "styled_paths": styled_paths,
        "error": None,
    }


def ensure_storage_directories() -> None:
    """
    Ensure that the storage directories for GLB files and demo videos exist.

    Creates the directories if they don't already exist.
    """
    GLB_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_VIDEO_STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def generate_glb_path(project_id: str, filename: Optional[str] = None) -> str:
    """
    Generate a unique file path for storing a GLB file.

    Args:
        project_id: The project ID to associate with the GLB file.
        filename: Optional custom filename. If not provided, a UUID-based name.

    Returns:
        The full path where the GLB file should be stored.
    """
    ensure_storage_directories()

    if filename is None:
        filename = f"{project_id}_{uuid.uuid4().hex[:8]}.glb"
    elif not filename.endswith(".glb"):
        filename = f"{filename}.glb"

    return str(GLB_STORAGE_DIR / filename)


def generate_video_path(
    project_id: str,
    generation_number: Optional[int] = None,
    filename: Optional[str] = None,
) -> str:
    """
    Generate a unique file path for storing a demo video (MP4).

    Args:
        project_id: The project ID to associate with the video file.
        generation_number: Optional generation number for versioning.
        filename: Optional custom filename. If not provided, a UUID-based name.

    Returns:
        The full path where the video file should be stored.
    """
    ensure_storage_directories()

    if filename is None:
        gen_suffix = f"_gen{generation_number}" if generation_number else ""
        filename = f"{project_id}{gen_suffix}_{uuid.uuid4().hex[:8]}.mp4"
    elif not filename.endswith(".mp4"):
        filename = f"{filename}.mp4"

    return str(DEMO_VIDEO_STORAGE_DIR / filename)


def _call_infer_api(
    image_paths: List[str],
    key: str,
    output_mp4_path: str,
) -> Dict[str, Any]:
    """
    Call the /infer API to generate a video from 4 input images.

    Args:
        image_paths: List of 4 image file paths (view0, view1, view2, view3).
        key: Unique key for this job (used later for /convert).
        output_mp4_path: Path where the output MP4 should be saved.

    Returns:
        Dict containing:
            - success: Boolean indicating if the API call succeeded
            - error: Error message if failed (None on success)
            - video_path: Path to saved video (same as output_mp4_path)
    """
    # Validate we have exactly 4 images
    if len(image_paths) < 4:
        return {
            "success": False,
            "error": f"4 images required, got {len(image_paths)}",
            "video_path": None,
        }

    # Verify all image files exist
    for i, img_path in enumerate(image_paths[:4]):
        if not Path(img_path).exists():
            return {
                "success": False,
                "error": f"Image file not found: {img_path}",
                "video_path": None,
            }

    # Prepare multipart files for the request
    files = []
    file_handles = []
    try:
        for i in range(4):
            img_path = image_paths[i]
            # Open file and keep handle for cleanup
            fh = open(img_path, "rb")
            file_handles.append(fh)
            # Use view0, view1, view2, view3 as field names
            files.append((f"view{i}", (f"view{i}.png", fh, "image/png")))

        print(f"[INFER] POST {INFER_HOST}")
        print(f"[INFER] key = {key}")
        print(f"[INFER] images = {image_paths[:4]}")

        # Make the API request
        response = requests.post(
            INFER_HOST,
            data={"key": key},
            files=files,
            timeout=INFER_TIMEOUT,
        )

        if response.status_code != 200:
            error_msg = f"/infer returned status {response.status_code}"
            try:
                error_detail = response.json()
                error_msg += f": {error_detail}"
            except Exception:
                error_msg += f": {response.text[:500]}"
            return {
                "success": False,
                "error": error_msg,
                "video_path": None,
            }

        # Save the MP4 response
        ensure_storage_directories()
        with open(output_mp4_path, "wb") as f:
            f.write(response.content)

        print(f"[INFER] Saved MP4 to {output_mp4_path}")

        return {
            "success": True,
            "error": None,
            "video_path": output_mp4_path,
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": f"/infer request timed out after {INFER_TIMEOUT}s",
            "video_path": None,
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"/infer HTTP error: {str(e)}",
            "video_path": None,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"/infer unexpected error: {str(e)}",
            "video_path": None,
        }
    finally:
        # Close all file handles
        for fh in file_handles:
            try:
                fh.close()
            except Exception:
                pass


def _call_convert_api(
    key: str,
    output_glb_path: str,
) -> Dict[str, Any]:
    """
    Call the /convert API to generate a GLB from a previously processed job.

    The /convert endpoint uses the key from a prior /infer call to retrieve
    the PLY data and convert it to GLB format.

    Args:
        key: The unique key used in the prior /infer call.
        output_glb_path: Path where the output GLB should be saved.

    Returns:
        Dict containing:
            - success: Boolean indicating if the API call succeeded
            - error: Error message if failed (None on success)
            - glb_path: Path to saved GLB (same as output_glb_path)
    """
    try:
        print(f"[CONVERT] POST {CONVERT_HOST}")
        print(f"[CONVERT] key = {key}")

        response = requests.post(
            CONVERT_HOST,
            data={"key": key},
            timeout=CONVERT_TIMEOUT,
        )

        if response.status_code != 200:
            error_msg = f"/convert returned status {response.status_code}"
            try:
                error_detail = response.json()
                error_msg += f": {error_detail}"
            except Exception:
                error_msg += f": {response.text[:500]}"
            return {
                "success": False,
                "error": error_msg,
                "glb_path": None,
            }

        # Save the GLB response
        ensure_storage_directories()
        with open(output_glb_path, "wb") as f:
            f.write(response.content)

        print(f"[CONVERT] Saved GLB to {output_glb_path}")

        return {
            "success": True,
            "error": None,
            "glb_path": output_glb_path,
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": f"/convert request timed out after {CONVERT_TIMEOUT}s",
            "glb_path": None,
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"/convert HTTP error: {str(e)}",
            "glb_path": None,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"/convert unexpected error: {str(e)}",
            "glb_path": None,
        }


def todo_generate_video(
    image_paths: List[str],
    project_id: str,
) -> Dict[str, Any]:
    """
    Generate initial video from 4 input images by calling the /infer API.

    This is the first step in the conversational workflow. User uploads 4 images
    and an initial video is generated via the external LGM inference service.
    The video is then uploaded to S3 for frontend access.

    Args:
        image_paths: List of paths to input images (expects 4 images).
        project_id: The project ID for file naming and tracking.

    Returns:
        Dict containing:
            - video_path: Local path to the generated video file
            - video_url: S3 presigned URL for frontend access
            - video_object_key: S3 object key
            - success: Boolean indicating if processing succeeded
            - error: Error message if processing failed (None on success)
            - generation_number: The generation number (1 for initial)
            - key: The job key used (needed for GLB conversion)
    """
    if len(image_paths) < 4:
        return {
            "video_path": None,
            "video_url": None,
            "video_object_key": None,
            "success": False,
            "error": f"4 images required, got {len(image_paths)}",
            "generation_number": 0,
            "key": None,
        }

    try:
        # Generate unique key for this job (used for both /infer and /convert)
        job_key = f"{project_id}_gen1_{uuid.uuid4().hex[:8]}"

        # Generate the output video path
        video_path = generate_video_path(project_id, generation_number=1)

        print("[todo_generate_video] Starting video generation")
        print(f"[todo_generate_video] Project ID: {project_id}")
        print(f"[todo_generate_video] Job key: {job_key}")
        print(f"[todo_generate_video] Image paths: {image_paths[:4]}")

        # Call the /infer API
        result = _call_infer_api(
            image_paths=image_paths[:4],
            key=job_key,
            output_mp4_path=video_path,
        )

        if not result["success"]:
            return {
                "video_path": None,
                "video_url": None,
                "video_object_key": None,
                "success": False,
                "error": result["error"],
                "generation_number": 0,
                "key": job_key,
            }

        # Upload video to S3
        print("[todo_generate_video] Uploading video to S3...")
        s3_result = _upload_video_to_s3(
            local_path=result["video_path"],
            project_id=project_id,
            generation_number=1,
        )

        if not s3_result["success"]:
            return {
                "video_path": result["video_path"],
                "video_url": None,
                "video_object_key": None,
                "success": False,
                "error": f"S3 upload failed: {s3_result['error']}",
                "generation_number": 1,
                "key": job_key,
            }

        print("[todo_generate_video] Video generation complete")
        return {
            "video_path": result["video_path"],
            "video_url": s3_result["presigned_url"],
            "video_object_key": s3_result["object_key"],
            "success": True,
            "error": None,
            "generation_number": 1,
            "key": job_key,
        }

    except Exception as e:
        return {
            "video_path": None,
            "video_url": None,
            "video_object_key": None,
            "success": False,
            "error": f"Video generation failed: {str(e)}",
            "generation_number": 0,
            "key": None,
        }


def todo_generate_video_from_prompt(
    original_images: List[str],
    prompt: str,
    new_images: Optional[List[str]] = None,
    project_id: Optional[str] = None,
    generation_number: int = 1,
) -> Dict[str, Any]:
    """
    Generate updated video based on prompt and optionally new images.

    This function applies style transfer to all 4 images using the prompt,
    then generates a video from the styled images. The video is uploaded to S3.

    Flow:
    1. Apply style_transfer_with_text to all 4 images using the prompt
    2. Use the 4 styled images to call /infer API
    3. Upload the resulting video to S3
    4. Return S3 URL and new job_key (for GLB conversion)

    Args:
        original_images: List of original image paths stored for this project.
        prompt: Text prompt describing the desired style/changes.
        new_images: Optional list of new image paths to replace originals.
        project_id: The project ID for file naming.
        generation_number: The current generation number for versioning.

    Returns:
        Dict containing:
            - video_path: Local path to the generated video file
            - video_url: S3 presigned URL for frontend access
            - video_object_key: S3 object key
            - success: Boolean indicating if processing succeeded
            - error: Error message if processing failed (None on success)
            - generation_number: The new generation number
            - images_used: List of image paths that were used (styled images)
            - prompt_applied: The prompt that was applied
            - key: The job key used (needed for GLB conversion)
    """
    # Determine which images to use
    images_to_use = (
        new_images if new_images and len(new_images) >= 4 else original_images
    )

    base_error_response = {
        "video_path": None,
        "video_url": None,
        "video_object_key": None,
        "success": False,
        "generation_number": generation_number,
        "images_used": images_to_use[:4] if images_to_use else [],
        "prompt_applied": prompt,
        "key": None,
    }

    if not images_to_use or len(images_to_use) < 4:
        base_error_response["error"] = "4 images required for video generation"
        return base_error_response

    if not prompt:
        base_error_response["error"] = "Prompt is required for video update"
        base_error_response["prompt_applied"] = None
        return base_error_response

    try:
        new_gen_number = generation_number + 1
        effective_project_id = project_id or uuid.uuid4().hex

        print("[prompt_video] Starting prompt-based video generation")
        print(f"[prompt_video] Prompt: {prompt}")
        print(f"[prompt_video] Images: {images_to_use[:4]}")
        print(f"[prompt_video] Generation: {new_gen_number}")

        # Step 1: Apply style transfer to all 4 images
        print("[prompt_video] Step 1: Applying style transfer to images...")
        style_result = apply_style_transfer_to_images(
            image_paths=images_to_use[:4],
            prompt=prompt,
        )

        if not style_result["success"]:
            base_error_response["error"] = (
                f"Style transfer failed: {style_result['error']}"
            )
            return base_error_response

        styled_images = style_result["styled_paths"]
        print(f"[prompt_video] Styled images: {styled_images}")

        # Step 2: Generate unique key for this job (NEW key for styled images)
        job_key = f"{effective_project_id}_gen{new_gen_number}_{uuid.uuid4().hex[:8]}"

        # Generate the output video path
        video_path = generate_video_path(
            effective_project_id, generation_number=new_gen_number
        )

        print(f"[prompt_video] Step 2: Calling /infer with styled images...")
        print(f"[prompt_video] Job key: {job_key}")

        # Step 3: Call the /infer API with styled images
        result = _call_infer_api(
            image_paths=styled_images,
            key=job_key,
            output_mp4_path=video_path,
        )

        if not result["success"]:
            base_error_response["error"] = result["error"]
            base_error_response["key"] = job_key
            base_error_response["images_used"] = styled_images
            return base_error_response

        # Step 4: Upload video to S3
        print("[prompt_video] Step 3: Uploading video to S3...")
        s3_result = _upload_video_to_s3(
            local_path=result["video_path"],
            project_id=effective_project_id,
            generation_number=new_gen_number,
        )

        if not s3_result["success"]:
            return {
                "video_path": result["video_path"],
                "video_url": None,
                "video_object_key": None,
                "success": False,
                "error": f"S3 upload failed: {s3_result['error']}",
                "generation_number": new_gen_number,
                "images_used": styled_images,
                "prompt_applied": prompt,
                "key": job_key,
            }

        print("[prompt_video] Video generation with prompt complete")
        return {
            "video_path": result["video_path"],
            "video_url": s3_result["presigned_url"],
            "video_object_key": s3_result["object_key"],
            "success": True,
            "error": None,
            "generation_number": new_gen_number,
            "images_used": styled_images,
            "prompt_applied": prompt,
            "key": job_key,
        }

    except Exception as e:
        base_error_response["error"] = f"Prompt-based generation failed: {str(e)}"
        return base_error_response


def todo_generate_glb(
    project_id: str,
    image_paths: List[str],
    latest_prompt: Optional[str] = None,
    job_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate GLB 3D model file by calling the /convert API.

    If a job_key is provided (from a prior video generation), it uses that key
    to convert the existing PLY to GLB. Otherwise, it first calls /infer to
    generate the 3D data, then calls /convert. The GLB is uploaded to S3.

    Args:
        project_id: The project ID to generate GLB for.
        image_paths: List of image paths to use for GLB generation.
        latest_prompt: Optional latest prompt (for reference/logging).
        job_key: Optional key from prior /infer call (skips re-inference).

    Returns:
        Dict containing:
            - glb_path: Local path to the generated GLB file
            - glb_url: S3 presigned URL for frontend access
            - glb_object_key: S3 object key
            - success: Boolean indicating if processing succeeded
            - error: Error message if processing failed (None on success)
            - key: The job key used
    """
    if not image_paths or len(image_paths) < 4:
        # If we have a job_key, we can still try to convert
        if not job_key:
            return {
                "glb_path": None,
                "glb_url": None,
                "glb_object_key": None,
                "success": False,
                "error": "4 images required for GLB (or provide job_key)",
                "key": None,
            }

    try:
        # Generate the output GLB path
        glb_path = generate_glb_path(project_id)

        print("[todo_generate_glb] Starting GLB generation")
        print(f"[todo_generate_glb] Project ID: {project_id}")
        print(f"[todo_generate_glb] Latest prompt: {latest_prompt}")

        # Determine the key to use for /convert
        effective_key = job_key

        # If no job_key provided, we need to run /infer first
        if not effective_key:
            effective_key = f"{project_id}_glb_{uuid.uuid4().hex[:8]}"

            print("[todo_generate_glb] No job_key, running /infer first")
            print(f"[todo_generate_glb] New key: {effective_key}")

            # Generate a temporary video path (we don't necessarily need it)
            temp_video_path = generate_video_path(
                project_id, filename=f"temp_{effective_key}"
            )

            infer_result = _call_infer_api(
                image_paths=image_paths[:4],
                key=effective_key,
                output_mp4_path=temp_video_path,
            )

            if not infer_result["success"]:
                return {
                    "glb_path": None,
                    "glb_url": None,
                    "glb_object_key": None,
                    "success": False,
                    "error": f"Inference failed: {infer_result['error']}",
                    "key": effective_key,
                }

        print(f"[todo_generate_glb] Calling /convert with key: {effective_key}")

        # Call the /convert API
        convert_result = _call_convert_api(
            key=effective_key,
            output_glb_path=glb_path,
        )

        if not convert_result["success"]:
            return {
                "glb_path": None,
                "glb_url": None,
                "glb_object_key": None,
                "success": False,
                "error": convert_result["error"],
                "key": effective_key,
            }

        # Upload GLB to S3
        print("[todo_generate_glb] Uploading GLB to S3...")
        s3_result = _upload_glb_to_s3(
            local_path=convert_result["glb_path"],
            project_id=project_id,
        )

        if not s3_result["success"]:
            return {
                "glb_path": convert_result["glb_path"],
                "glb_url": None,
                "glb_object_key": None,
                "success": False,
                "error": f"S3 upload failed: {s3_result['error']}",
                "key": effective_key,
            }

        print("[todo_generate_glb] GLB generation complete")
        return {
            "glb_path": convert_result["glb_path"],
            "glb_url": s3_result["presigned_url"],
            "glb_object_key": s3_result["object_key"],
            "success": True,
            "error": None,
            "key": effective_key,
        }

    except Exception as e:
        return {
            "glb_path": None,
            "glb_url": None,
            "glb_object_key": None,
            "success": False,
            "error": f"GLB generation failed: {str(e)}",
            "key": job_key,
        }


# Legacy functions for backward compatibility


def process_images_to_video(
    image_paths: List[str],
    project_id: str,
) -> Dict[str, Any]:
    """
    Legacy function - wraps todo_generate_video for backward compatibility.

    Args:
        image_paths: List of paths to input images.
        project_id: The project ID for file naming.

    Returns:
        Dict with video_path, success, and error fields.
    """
    result = todo_generate_video(image_paths, project_id)
    return {
        "video_path": result.get("video_path"),
        "success": result.get("success"),
        "error": result.get("error"),
    }


def process_images_to_glb(
    image_paths: List[str],
    project_id: str,
) -> Dict[str, Any]:
    """
    Legacy function - wraps todo_generate_glb for backward compatibility.

    Args:
        image_paths: List of paths to input images.
        project_id: The project ID for file naming.

    Returns:
        Dict with glb_path, success, and error fields.
    """
    return todo_generate_glb(project_id, image_paths)


def get_glb_file_path(project_3d_id: str) -> Optional[str]:
    """
    Retrieve the GLB file path for a given Project3D ID.

    Args:
        project_3d_id: The ID of the Project3D record.

    Returns:
        The path to the GLB file if it exists, None otherwise.
    """
    potential_path = GLB_STORAGE_DIR / f"{project_3d_id}.glb"

    if potential_path.exists():
        return str(potential_path)

    # Also check for pattern-based paths
    for glb_file in GLB_STORAGE_DIR.glob(f"{project_3d_id}_*.glb"):
        return str(glb_file)

    return None


def validate_image_paths(image_paths: List[str]) -> Dict[str, Any]:
    """
    Validate that the provided image paths exist and are valid image files.

    Args:
        image_paths: List of image file paths to validate.

    Returns:
        Dict containing:
            - valid: Boolean indicating if all paths are valid
            - invalid_paths: List of paths that failed validation
            - error: Error message if validation failed (None on success)
    """
    invalid_paths = []

    for path in image_paths:
        if not path:
            invalid_paths.append(path)
            continue

        file_path = Path(path)
        if not file_path.exists():
            invalid_paths.append(path)
            continue

        # Check for common image extensions
        valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
        if file_path.suffix.lower() not in valid_extensions:
            invalid_paths.append(path)

    if invalid_paths:
        return {
            "valid": False,
            "invalid_paths": invalid_paths,
            "error": f"Invalid or missing image files: {invalid_paths}",
        }

    return {
        "valid": True,
        "invalid_paths": [],
        "error": None,
    }
