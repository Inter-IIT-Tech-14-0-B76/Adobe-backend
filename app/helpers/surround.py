"""
Helper functions for 3D object visualization and surround view processing.

This module provides utility functions for the conversational 3D workflow:
- todo_generate_video: Initial video generation from 4 images
- todo_generate_video_from_prompt: Update video based on prompt and images
- todo_generate_glb: Generate GLB file from current state

Storage locations:
- GLB files: /workspace/backend/glb/
- Demo videos: /workspace/backend/demo_videos/
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# Storage directories for 3D assets
GLB_STORAGE_DIR = Path("/workspace/backend/glb")
DEMO_VIDEO_STORAGE_DIR = Path("/workspace/backend/demo_videos")


class SurroundProcessingError(Exception):
    """Exception raised for errors during 3D/surround processing."""

    pass


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
        filename: Optional custom filename. If not provided, a UUID-based name is used.

    Returns:
        The full path where the GLB file should be stored.
    """
    ensure_storage_directories()

    if filename is None:
        filename = f"{project_id}_{uuid.uuid4().hex}.glb"
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
        filename: Optional custom filename. If not provided, a UUID-based name is used.

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


def todo_generate_video(
    image_paths: List[str],
    project_id: str,
) -> Dict[str, Any]:
    """
    Generate initial video from 4 input images.

    This is the first step in the conversational workflow. User uploads 4 images
    and an initial video is generated.

    Args:
        image_paths: List of paths to input images (expects 4 images).
        project_id: The project ID for file naming and tracking.

    Returns:
        Dict containing:
            - video_path: Path to the generated video file
            - success: Boolean indicating if processing succeeded
            - error: Error message if processing failed (None on success)
            - generation_number: The generation number (1 for initial)

    Example:
        >>> result = todo_generate_video(
        ...     image_paths=["/path/img1.jpg", "/path/img2.jpg", "/path/img3.jpg", "/path/img4.jpg"],
        ...     project_id="abc123"
        ... )
        >>> print(result["video_path"])
        /workspace/backend/demo_videos/abc123_gen1_a1b2c3d4.mp4
    """
    if len(image_paths) < 4:
        return {
            "video_path": None,
            "success": False,
            "error": f"4 images are required for initial video generation, got {len(image_paths)}",
            "generation_number": 0,
        }

    try:
        # Generate the output video path
        video_path = generate_video_path(project_id, generation_number=1)

        # TODO: Replace with actual video generation logic
        # This is a dummy implementation that creates an empty file as placeholder
        ensure_storage_directories()

        # Create a placeholder file (to be replaced with actual processing)
        # In production, this would call your video generation pipeline
        Path(video_path).touch()

        print(f"[TODO] todo_generate_video called with {len(image_paths)} images")
        print(f"[TODO] Image paths: {image_paths}")
        print(f"[TODO] Generated video path: {video_path}")

        return {
            "video_path": video_path,
            "success": True,
            "error": None,
            "generation_number": 1,
        }

    except Exception as e:
        return {
            "video_path": None,
            "success": False,
            "error": f"Video generation failed: {str(e)}",
            "generation_number": 0,
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

    This is used for subsequent updates in the conversational workflow.
    If no new images are provided, the original images are used.

    Args:
        original_images: List of original image paths stored for this project.
        prompt: Text prompt describing the desired changes/style.
        new_images: Optional list of new image paths to replace/augment originals.
        project_id: The project ID for file naming.
        generation_number: The current generation number for versioning.

    Returns:
        Dict containing:
            - video_path: Path to the generated video file
            - success: Boolean indicating if processing succeeded
            - error: Error message if processing failed (None on success)
            - generation_number: The new generation number
            - images_used: List of image paths that were used
            - prompt_applied: The prompt that was applied

    Example:
        >>> result = todo_generate_video_from_prompt(
        ...     original_images=["/path/img1.jpg", "/path/img2.jpg", "/path/img3.jpg", "/path/img4.jpg"],
        ...     prompt="Make it look more futuristic",
        ...     project_id="abc123",
        ...     generation_number=2
        ... )
    """
    # Determine which images to use
    images_to_use = new_images if new_images and len(new_images) > 0 else original_images

    if not images_to_use:
        return {
            "video_path": None,
            "success": False,
            "error": "No images available for video generation",
            "generation_number": generation_number,
            "images_used": [],
            "prompt_applied": prompt,
        }

    if not prompt:
        return {
            "video_path": None,
            "success": False,
            "error": "Prompt is required for video update",
            "generation_number": generation_number,
            "images_used": images_to_use,
            "prompt_applied": None,
        }

    try:
        new_gen_number = generation_number + 1
        effective_project_id = project_id or uuid.uuid4().hex

        # Generate the output video path
        video_path = generate_video_path(effective_project_id, generation_number=new_gen_number)

        # TODO: Replace with actual prompt-based video generation logic
        # This is a dummy implementation
        ensure_storage_directories()

        # Create a placeholder file (to be replaced with actual processing)
        Path(video_path).touch()

        print(f"[TODO] todo_generate_video_from_prompt called")
        print(f"[TODO] Prompt: {prompt}")
        print(f"[TODO] Images used: {images_to_use}")
        print(f"[TODO] Generation number: {new_gen_number}")
        print(f"[TODO] Generated video path: {video_path}")

        return {
            "video_path": video_path,
            "success": True,
            "error": None,
            "generation_number": new_gen_number,
            "images_used": images_to_use,
            "prompt_applied": prompt,
        }

    except Exception as e:
        return {
            "video_path": None,
            "success": False,
            "error": f"Prompt-based video generation failed: {str(e)}",
            "generation_number": generation_number,
            "images_used": images_to_use,
            "prompt_applied": prompt,
        }


def todo_generate_glb(
    project_id: str,
    image_paths: List[str],
    latest_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate GLB 3D model file from the current project state.

    Uses the latest images and prompt associated with the project ID
    to generate a GLB file for 3D visualization.

    Args:
        project_id: The project ID to generate GLB for.
        image_paths: List of image paths to use for GLB generation.
        latest_prompt: Optional latest prompt to influence GLB generation.

    Returns:
        Dict containing:
            - glb_path: Path to the generated GLB file
            - success: Boolean indicating if processing succeeded
            - error: Error message if processing failed (None on success)

    Example:
        >>> result = todo_generate_glb(
        ...     project_id="abc123",
        ...     image_paths=["/path/img1.jpg", "/path/img2.jpg", "/path/img3.jpg", "/path/img4.jpg"],
        ...     latest_prompt="Futuristic style"
        ... )
        >>> print(result["glb_path"])
        /workspace/backend/glb/abc123_a1b2c3d4.glb
    """
    if not image_paths:
        return {
            "glb_path": None,
            "success": False,
            "error": "No images available for GLB generation",
        }

    try:
        # Generate the output GLB path
        glb_path = generate_glb_path(project_id)

        # TODO: Replace with actual GLB generation logic
        # This is a dummy implementation
        ensure_storage_directories()

        # Create a placeholder file (to be replaced with actual processing)
        Path(glb_path).touch()

        print(f"[TODO] todo_generate_glb called")
        print(f"[TODO] Project ID: {project_id}")
        print(f"[TODO] Image paths: {image_paths}")
        print(f"[TODO] Latest prompt: {latest_prompt}")
        print(f"[TODO] Generated GLB path: {glb_path}")

        return {
            "glb_path": glb_path,
            "success": True,
            "error": None,
        }

    except Exception as e:
        return {
            "glb_path": None,
            "success": False,
            "error": f"GLB generation failed: {str(e)}",
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
