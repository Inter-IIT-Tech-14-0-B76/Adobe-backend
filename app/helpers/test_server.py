"""
AI Editing Flow Helper Functions
Implements the complete flow: Analyze -> Classify -> Execute Tool
Includes comprehensive debugging and individual endpoint testing
"""

import json
import time
import traceback
from typing import Any, Dict, Optional

import requests

from config import WORKSPACE_OUTPUT_DIR, WORKSPACE_SERVER

timings = {}
DEBUG = True  # Set to False to reduce verbosity

# Default test image paths (adjust as needed)
DEFAULT_CONTENT_IMAGE = "/workspace/AIP/workspace/outputs/images/main.png"
DEFAULT_STYLE_IMAGE = "/workspace/AIP/workspace/outputs/images/reference.png"


def _determine_image_roles_local(user_prompt: str, num_images: int) -> Dict[str, Any]:
    """
    Local helper to determine which image is style/reference and which is content.
    Used when the server's /classify endpoint doesn't support num_images parameter.
    """
    prompt_lower = user_prompt.lower()

    # Default: first image is style, second is content
    style_index = 0
    content_index = 1
    confidence = "low"

    # Patterns where first image is style reference
    first_is_style_patterns = [
        "style of first",
        "first image style",
        "style from first",
        "like the first",
        "first image's style",
        "copy first",
        "transfer from first",
        "first one's style",
        "style of image 1",
        "image 1 style",
        "style from image 1",
    ]

    # Patterns where second image is style reference
    second_is_style_patterns = [
        "style of second",
        "second image style",
        "style from second",
        "like the second",
        "second image's style",
        "copy second",
        "transfer from second",
        "second one's style",
        "style of image 2",
        "image 2 style",
        "style from image 2",
        "starry night",  # specific case: Van Gogh's painting is likely the style reference
    ]

    # Patterns where first is content (to be styled)
    first_is_content_patterns = [
        "apply to first",
        "first image to",
        "style the first",
        "edit first",
        "transform first",
        "first one to",
        "apply to image 1",
        "image 1 to",
        "to the couple",
        "to the person",
        "to the photo",  # content is often described by subject
    ]

    # Patterns where second is content (to be styled)
    second_is_content_patterns = [
        "apply to second",
        "second image to",
        "style the second",
        "edit second",
        "transform second",
        "second one to",
        "apply to image 2",
        "image 2 to",
    ]

    for pattern in first_is_style_patterns:
        if pattern in prompt_lower:
            style_index = 0
            content_index = 1
            confidence = "high"
            break

    for pattern in second_is_style_patterns:
        if pattern in prompt_lower:
            style_index = 1
            content_index = 0
            confidence = "high"
            break

    for pattern in first_is_content_patterns:
        if pattern in prompt_lower:
            content_index = 0
            style_index = 1
            confidence = "high"
            break

    for pattern in second_is_content_patterns:
        if pattern in prompt_lower:
            content_index = 1
            style_index = 0
            confidence = "high"
            break

    # Ensure indices are within bounds
    if style_index >= num_images:
        style_index = 0
    if content_index >= num_images:
        content_index = num_images - 1

    return {
        "style_index": style_index,
        "content_index": content_index,
        "confidence": confidence,
    }


def debug_print(message: str, level: str = "INFO"):
    """Print debug messages with formatting."""
    if DEBUG:
        prefix = {
            "INFO": "[INFO]",
            "DEBUG": "[DEBUG]",
            "ERROR": "[ERROR]",
            "WARN": "[WARN]",
            "REQUEST": "[REQUEST]",
            "RESPONSE": "[RESPONSE]",
        }.get(level, "[LOG]")
        print(f"{prefix} {message}")


def timed(name: str, func, *args, **kwargs):
    """Wrapper to time function execution."""
    debug_print(f"Starting timed function: {name}", "DEBUG")
    start = time.time()
    try:
        result = func(*args, **kwargs)
        duration = round(time.time() - start, 3)
        timings[name] = duration
        debug_print(f"Completed {name} in {duration}s", "DEBUG")
        return result
    except Exception as e:
        duration = round(time.time() - start, 3)
        timings[name] = duration
        debug_print(f"Error in {name} after {duration}s: {str(e)}", "ERROR")
        raise


def pretty_print(resp: requests.Response, title: str = ""):
    """Pretty print response with error handling."""
    if title:
        print(f"\n{'=' * 60}")
        print(f"{title}")
        print("=" * 60)

    debug_print(f"Response status: {resp.status_code}", "RESPONSE")
    debug_print(f"Response headers: {dict(resp.headers)}", "RESPONSE")

    try:
        data = resp.json()
        print(json.dumps(data, indent=2))
        debug_print("Response parsed successfully", "RESPONSE")
        return data
    except json.JSONDecodeError as e:
        debug_print(f"JSON decode error: {str(e)}", "ERROR")
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.text[:500]}")  # Limit text output
        return None
    except Exception as e:
        debug_print(f"Unexpected error parsing response: {str(e)}", "ERROR")
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.text[:500]}")
        return None


# =============================================================================
# HEALTH & STATUS CHECKS
# =============================================================================


def check_health():
    """Check server health endpoint."""
    debug_print("Checking server health...", "INFO")
    try:
        url = f"{WORKSPACE_SERVER}/health"
        debug_print(f"GET {url}", "REQUEST")
        r = requests.get(url, timeout=10)
        debug_print(f"Health check response: {r.status_code}", "RESPONSE")
        return pretty_print(r)
    except requests.exceptions.Timeout:
        debug_print("Health check timed out", "ERROR")
        print("ERROR: Health check timed out")
        return None
    except requests.exceptions.ConnectionError as e:
        debug_print(f"Connection error: {str(e)}", "ERROR")
        print(f"ERROR: Cannot connect to {WORKSPACE_SERVER}")
        return None
    except Exception as e:
        debug_print(f"Unexpected error: {str(e)}", "ERROR")
        print(f"ERROR: {str(e)}")
        traceback.print_exc()
        return None


def check_status():
    """Check server status endpoint."""
    debug_print("Checking server status...", "INFO")
    try:
        url = f"{WORKSPACE_SERVER}/status"
        debug_print(f"GET {url}", "REQUEST")
        r = requests.get(url, timeout=10)
        debug_print(f"Status check response: {r.status_code}", "RESPONSE")
        return pretty_print(r)
    except Exception as e:
        debug_print(f"Status check error: {str(e)}", "ERROR")
        print(f"ERROR: {str(e)}")
        traceback.print_exc()
        return None


# =============================================================================
# INDIVIDUAL ENDPOINT TESTS
# =============================================================================


def test_style_transfer_text(
    content_image: str = None, style_text: str = None, prompt: str = None
):
    """Test style transfer with text endpoint."""
    print("\n--- /style-transfer/text ---")

    if content_image is None:
        content_image = DEFAULT_CONTENT_IMAGE
    if style_text is None:
        style_text = "soft pastel art style with gentle tones"
    if prompt is None:
        prompt = "apply a smooth dreamy pastel look"

    debug_print(f"Content image: {content_image}", "REQUEST")
    debug_print(f"Style text: {style_text}", "REQUEST")
    debug_print(f"Prompt: {prompt}", "REQUEST")

    payload = {
        "content": content_image,
        "style_text": style_text,
        "prompt": prompt,
        "steps": 40,
        "style_steps": 20,
    }

    debug_print(f"Payload: {json.dumps(payload, indent=2)}", "REQUEST")

    try:
        url = f"{WORKSPACE_SERVER}/style-transfer/text"
        debug_print(f"POST {url}", "REQUEST")
        r = requests.post(url, json=payload, timeout=120)
        debug_print(f"Response status: {r.status_code}", "RESPONSE")
        return pretty_print(r)
    except requests.exceptions.Timeout:
        debug_print("Request timed out after 120s", "ERROR")
        print("ERROR: Request timed out")
        return None
    except Exception as e:
        debug_print(f"Error: {str(e)}", "ERROR")
        traceback.print_exc()
        return None


def test_style_transfer_ref(
    content_image: str = None, style_image: str = None, prompt: str = None
):
    """Test style transfer with reference image endpoint."""
    print("\n--- /style-transfer/ref ---")

    if content_image is None:
        content_image = DEFAULT_CONTENT_IMAGE
    if style_image is None:
        style_image = DEFAULT_STYLE_IMAGE
    if prompt is None:
        prompt = "match lighting and texture from the reference"

    debug_print(f"Content image: {content_image}", "REQUEST")
    debug_print(f"Style image: {style_image}", "REQUEST")
    debug_print(f"Prompt: {prompt}", "REQUEST")

    payload = {
        "content": content_image,
        "style": style_image,
        "prompt": prompt,
        "steps": 50,
        "negative_prompt": "cluttered, complex background, dark background",
    }

    debug_print(f"Payload: {json.dumps(payload, indent=2)}", "REQUEST")

    try:
        url = f"{WORKSPACE_SERVER}/style-transfer/ref"
        debug_print(f"POST {url}", "REQUEST")
        r = requests.post(url, json=payload, timeout=120)
        debug_print(f"Response status: {r.status_code}", "RESPONSE")
        return pretty_print(r)
    except Exception as e:
        debug_print(f"Error: {str(e)}", "ERROR")
        traceback.print_exc()
        return None


def test_style_transfer_ref_pipeline(
    content_image: Optional[str] = None,
    style_image: Optional[str] = None,
    prompt: Optional[str] = None,
):
    """
    Test style transfer with reference image using the full classification pipeline.
    This tests the classifier's ability to detect style_transfer_ref and assign image roles.
    """
    print("\n--- Style Transfer Ref Pipeline Test ---")

    if content_image is None:
        content_image = DEFAULT_CONTENT_IMAGE
    if style_image is None:
        style_image = DEFAULT_STYLE_IMAGE
    if prompt is None:
        prompt = "apply the style of the first image to the second image"

    debug_print(f"Content image: {content_image}", "REQUEST")
    debug_print(f"Style image: {style_image}", "REQUEST")
    debug_print(f"Prompt: {prompt}", "REQUEST")

    # Provide image analyses with role hints
    image_analyses = [
        {"role": "style", "type": "reference"},
        {"role": "content", "type": "target"},
    ]

    result = run_ai_editing_pipeline(
        image_path=content_image,
        user_prompt=prompt,
        image_paths=[style_image, content_image],
        image_analyses=image_analyses,
    )

    if result.get("error"):
        debug_print(f"Pipeline error: {result.get('error')}", "ERROR")
    else:
        debug_print(f"Pipeline result: {json.dumps(result, indent=2)}", "RESPONSE")

    return result


def test_color_grading(image_path: str = None, prompt: str = None, mode: str = "both"):
    """Test color grading endpoint."""
    print("\n--- /color-grading ---")

    if image_path is None:
        image_path = DEFAULT_CONTENT_IMAGE
    if prompt is None:
        prompt = "cinematic teal shadows and warm highlights"

    debug_print(f"Image: {image_path}", "REQUEST")
    debug_print(f"Prompt: {prompt}", "REQUEST")
    debug_print(f"Mode: {mode}", "REQUEST")

    payload = {
        "image": image_path,
        "prompt": prompt,
        "mode": mode,
    }

    debug_print(f"Payload: {json.dumps(payload, indent=2)}", "REQUEST")

    try:
        url = f"{WORKSPACE_SERVER}/color-grading"
        debug_print(f"POST {url}", "REQUEST")
        r = requests.post(url, json=payload, timeout=120)
        debug_print(f"Response status: {r.status_code}", "RESPONSE")
        return pretty_print(r)
    except Exception as e:
        debug_print(f"Error: {str(e)}", "ERROR")
        traceback.print_exc()
        return None


def test_ai_suggestions(image_path: str = None):
    """Test AI suggestions endpoint."""
    print("\n--- /ai-suggestions ---")

    if image_path is None:
        image_path = DEFAULT_CONTENT_IMAGE

    debug_print(f"Image: {image_path}", "REQUEST")

    payload = {"image": image_path}
    debug_print(f"Payload: {json.dumps(payload, indent=2)}", "REQUEST")

    try:
        url = f"{WORKSPACE_SERVER}/ai-suggestions"
        debug_print(f"POST {url}", "REQUEST")
        r = requests.post(url, json=payload, timeout=60)
        debug_print(f"Response status: {r.status_code}", "RESPONSE")
        return pretty_print(r)
    except Exception as e:
        debug_print(f"Error: {str(e)}", "ERROR")
        traceback.print_exc()
        return None


def test_classify_quick(prompt: str = None):
    """Test quick classification endpoint."""
    print("\n--- /classify (quick) ---")

    if prompt is None:
        prompt = "increase sharpness and reduce noise"

    debug_print(f"Prompt: {prompt}", "REQUEST")
    debug_print("Quick mode: True", "REQUEST")

    payload = {"prompt": prompt, "quick": True}

    debug_print(f"Payload: {json.dumps(payload, indent=2)}", "REQUEST")

    try:
        url = f"{WORKSPACE_SERVER}/classify"
        debug_print(f"POST {url}", "REQUEST")
        r = requests.post(url, json=payload, timeout=30)
        debug_print(f"Response status: {r.status_code}", "RESPONSE")
        return pretty_print(r)
    except Exception as e:
        debug_print(f"Error: {str(e)}", "ERROR")
        traceback.print_exc()
        return None


def test_classify_full(prompt: str = None):
    """Test full classification endpoint."""
    print("\n--- /classify (full) ---")

    if prompt is None:
        prompt = "convert to monochrome film look"

    debug_print(f"Prompt: {prompt}", "REQUEST")
    debug_print("Quick mode: False (full)", "REQUEST")

    payload = {"prompt": prompt, "quick": False}

    debug_print(f"Payload: {json.dumps(payload, indent=2)}", "REQUEST")

    try:
        url = f"{WORKSPACE_SERVER}/classify"
        debug_print(f"POST {url}", "REQUEST")
        r = requests.post(url, json=payload, timeout=30)
        debug_print(f"Response status: {r.status_code}", "RESPONSE")
        return pretty_print(r)
    except Exception as e:
        debug_print(f"Error: {str(e)}", "ERROR")
        traceback.print_exc()
        return None


def test_sam_segment(
    image_path: str = None, x: int = 150, y: int = 200, output_dir: str = None
):
    """Test SAM segmentation endpoint."""
    print("\n--- /sam/segment ---")

    if image_path is None:
        image_path = DEFAULT_CONTENT_IMAGE
    if output_dir is None:
        output_dir = "outputs/segmentation"

    debug_print(f"Image: {image_path}", "REQUEST")
    debug_print(f"Click point: ({x}, {y})", "REQUEST")
    debug_print(f"Output dir: {output_dir}", "REQUEST")

    payload = {
        "image": image_path,
        "x": x,
        "y": y,
        "output_dir": output_dir,
    }

    debug_print(f"Payload: {json.dumps(payload, indent=2)}", "REQUEST")

    try:
        url = f"{WORKSPACE_SERVER}/sam/segment"
        debug_print(f"POST {url}", "REQUEST")
        r = requests.post(url, json=payload, timeout=120)
        debug_print(f"Response status: {r.status_code}", "RESPONSE")
        return pretty_print(r)
    except Exception as e:
        debug_print(f"Error: {str(e)}", "ERROR")
        traceback.print_exc()
        return None


# =============================================================================
# PIPELINE FLOW FUNCTIONS
# =============================================================================


def _is_remix_prompt(prompt: str) -> tuple[bool, str]:
    """
    Check if prompt starts with 'Remix:' prefix.
    Returns (is_remix, cleaned_prompt).
    """
    if not prompt:
        return False, prompt
    prompt_stripped = prompt.strip()
    if prompt_stripped.lower().startswith("remix:"):
        # Extract the actual prompt after "Remix:"
        cleaned = prompt_stripped[6:].strip()
        return True, cleaned
    return False, prompt


def _run_comfy_pipeline(
    image_paths: list,
    prompt: str,
    num_images: int,
) -> Dict[str, Any]:
    """
    Run ComfyUI pipeline based on number of images.
    - 2+ images: comfy-remix (combine images)
    - 1 image: comfy-edit (edit single image with prompt)

    Args:
        image_paths: List of image paths
        prompt: The editing prompt (already cleaned of "Remix:" prefix if applicable)
        num_images: Number of images

    Returns:
        Dict with result or error
    """
    debug_print(f"Running ComfyUI pipeline with {num_images} image(s)", "INFO")

    if num_images >= 2:
        # Use comfy-remix for 2+ images
        tool = "comfy-remix"
        params = {
            "image1": image_paths[0],
            "image2": image_paths[1],
        }
        primary_image = image_paths[0]
        debug_print(f"Using comfy-remix: image1={image_paths[0]}, image2={image_paths[1]}", "INFO")
    else:
        # Use comfy-edit for single image
        tool = "comfy-edit"
        params = {}
        primary_image = image_paths[0]
        debug_print(f"Using comfy-edit: image={image_paths[0]}", "INFO")

    try:
        result = execute_tool(
            tool=tool,
            image_path=primary_image,
            prompt=prompt,
            params=params,
            ai_suggestions={},  # Not needed for ComfyUI
        )

        if result.get("error"):
            debug_print(f"ComfyUI tool error: {result.get('error')}", "ERROR")
            return result

        debug_print("ComfyUI pipeline completed successfully", "INFO")
        debug_print(f"Result: {json.dumps(result, indent=2)}", "RESPONSE")
        return result

    except Exception as e:
        error_msg = f"ComfyUI pipeline error: {str(e)}"
        debug_print(error_msg, "ERROR")
        traceback.print_exc()
        return {"error": error_msg}


def run_ai_editing_pipeline(
    image_path: str,
    user_prompt: str,
    image_paths: Optional[list] = None,
    image_analyses: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Run the complete AI editing pipeline: Analyze -> Classify -> Execute Tool.

    Special handling:
    - If prompt starts with "Remix:", uses ComfyUI endpoints:
      - 2+ images: comfy-remix
      - 1 image: comfy-edit
    - If classifier returns "default", uses ComfyUI endpoints based on image count

    Args:
        image_path: Path to the input image file (primary/first image)
        user_prompt: User's editing prompt/request
        image_paths: Optional list of all image paths (for multi-image scenarios)
        image_analyses: Optional list of image analysis dicts (one per image)

    Returns:
        Dict with 'error' key if failed, or result dict with output paths
    """
    # Build list of image paths
    if image_paths is None:
        image_paths = [image_path]
    num_images = len(image_paths)
    debug_print("=" * 60, "INFO")
    debug_print("STARTING AI EDITING PIPELINE", "INFO")
    debug_print("=" * 60, "INFO")
    debug_print(f"Image(s): {image_paths}", "INFO")
    debug_print(f"Number of images: {num_images}", "INFO")
    debug_print(f"Prompt: {user_prompt}", "INFO")

    # Check for "Remix:" prefix - bypass classification entirely
    is_remix, cleaned_prompt = _is_remix_prompt(user_prompt)
    if is_remix:
        debug_print("Detected 'Remix:' prefix - using ComfyUI pipeline", "INFO")
        return _run_comfy_pipeline(
            image_paths=image_paths,
            prompt=cleaned_prompt,
            num_images=num_images,
        )

    # Step 1: Analyze image
    debug_print("\n[STEP 1/3] Analyzing image...", "INFO")
    try:
        analyze_url = f"{WORKSPACE_SERVER}/ai-suggestions"
        analyze_payload = {"image": image_path}
        debug_print(f"POST {analyze_url}", "REQUEST")
        debug_print(f"Payload: {json.dumps(analyze_payload, indent=2)}", "REQUEST")

        analyze_resp = requests.post(analyze_url, json=analyze_payload, timeout=60)

        debug_print(f"Analysis response status: {analyze_resp.status_code}", "RESPONSE")

        if analyze_resp.status_code != 200:
            error_msg = f"Image analysis failed with status {analyze_resp.status_code}"
            debug_print(error_msg, "ERROR")
            debug_print(f"Response: {analyze_resp.text}", "ERROR")
            return {"error": error_msg, "response": analyze_resp.text}

        analysis_data = analyze_resp.json()
        debug_print(
            f"Analysis data received: {json.dumps(analysis_data, indent=2)}", "RESPONSE"
        )
        ai_suggestions = analysis_data

    except requests.exceptions.Timeout:
        error_msg = "Analysis request timed out"
        debug_print(error_msg, "ERROR")
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Analysis error: {str(e)}"
        debug_print(error_msg, "ERROR")
        traceback.print_exc()
        return {"error": error_msg}

    # Step 2: Classify prompt
    debug_print("\n[STEP 2/3] Classifying prompt...", "INFO")
    try:
        classify_url = f"{WORKSPACE_SERVER}/classify"
        classify_payload = {
            "prompt": user_prompt,
            "quick": False,
            "num_images": num_images,
            "image_analyses": image_analyses,
        }
        debug_print(f"POST {classify_url}", "REQUEST")
        debug_print(f"Payload: {json.dumps(classify_payload, indent=2)}", "REQUEST")

        classify_resp = requests.post(classify_url, json=classify_payload, timeout=30)

        debug_print(
            f"Classification response status: {classify_resp.status_code}", "RESPONSE"
        )

        if classify_resp.status_code != 200:
            error_msg = f"Classification failed with status {classify_resp.status_code}"
            debug_print(error_msg, "ERROR")
            debug_print(f"Response: {classify_resp.text}", "ERROR")
            return {"error": error_msg, "response": classify_resp.text}

        classification_data = classify_resp.json()
        debug_print(
            f"Classification data: {json.dumps(classification_data, indent=2)}",
            "RESPONSE",
        )

        # Handle different response formats
        if "classification" in classification_data:
            classification = classification_data.get("classification", "")
        else:
            print("[WARN]: Classification not found in response")
            classification = classification_data

        # Extract image_roles if present (for style_transfer_ref)
        image_roles = classification_data.get("image_roles")

        # Local override: detect style_transfer_ref when server doesn't support num_images
        # Check if we have multiple images and the prompt suggests style transfer between them
        if num_images >= 2 and not image_roles:
            prompt_lower = user_prompt.lower()
            style_ref_keywords = [
                "copy style",
                "match style",
                "like this image",
                "transfer style",
                "style from",
                "same style as",
                "style of the",
                "apply the style",
                "reference image",
                "style reference",
                "transfer the style",
                "style to",
                "make it look like",
            ]
            if any(kw in prompt_lower for kw in style_ref_keywords):
                debug_print(
                    "Local override: Detected style_transfer_ref scenario", "INFO"
                )
                classification = "style_transfer_ref"
                # Determine image roles from prompt
                image_roles = _determine_image_roles_local(user_prompt, num_images)

        tool = classification.strip().lower() if isinstance(classification, str) else ""
        params = {}

        # Map classification to tool endpoint
        if tool == "style_transfer_ref" and num_images >= 2:
            tool = "style-transfer-ref"
            # Use image_roles to determine which image is style and which is content
            if image_roles:
                style_idx = image_roles.get("style_index", 0)
                content_idx = image_roles.get("content_index", 1)
                params["style_image"] = image_paths[style_idx]
                params["content_image"] = image_paths[content_idx]
                debug_print(
                    f"Image roles - Style: {image_paths[style_idx]}, Content: {image_paths[content_idx]}",
                    "INFO",
                )
            else:
                # Default: first is style, second is content
                params["style_image"] = image_paths[0]
                params["content_image"] = (
                    image_paths[1] if len(image_paths) > 1 else image_paths[0]
                )
        elif tool == "default" or tool == "default_mode":
            # "default" classification -> use ComfyUI pipeline based on image count
            debug_print("Classification is 'default' - routing to ComfyUI pipeline", "INFO")
            return _run_comfy_pipeline(
                image_paths=image_paths,
                prompt=user_prompt,
                num_images=num_images,
            )
        elif tool == "style_transfer":
            tool = "style-transfer-text"
        elif tool == "color_grading":
            tool = "color-grading"

        debug_print(f"Detected tool: {tool}", "INFO")
        debug_print(f"Parameters: {json.dumps(params, indent=2)}", "INFO")

    except requests.exceptions.Timeout:
        error_msg = "Classification request timed out"
        debug_print(error_msg, "ERROR")
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Classification error: {str(e)}"
        debug_print(error_msg, "ERROR")
        traceback.print_exc()
        return {"error": error_msg}

    # Step 3: Execute tool
    debug_print("\n[STEP 3/3] Executing tool...", "INFO")
    try:
        # For style-transfer-ref, use the content image as the primary image_path
        primary_image = (
            params.get("content_image", image_path)
            if tool == "style-transfer-ref"
            else image_path
        )

        result = execute_tool(
            tool=tool,
            image_path=primary_image,
            prompt=user_prompt,
            params=params,
            ai_suggestions=ai_suggestions,
        )

        if result.get("error"):
            debug_print(f"Tool execution error: {result.get('error')}", "ERROR")
            return result

        debug_print("Tool execution completed successfully", "INFO")
        debug_print(f"Result: {json.dumps(result, indent=2)}", "RESPONSE")
        return result

    except Exception as e:
        error_msg = f"Tool execution error: {str(e)}"
        debug_print(error_msg, "ERROR")
        traceback.print_exc()
        return {"error": error_msg}


# =============================================================================
# OBJECT REMOVAL PIPELINE
# =============================================================================


def run_sam_segmentation(
    image_path: str,
    x: int,
    y: int,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run SAM2 segmentation on an image at a specific point.

    Args:
        image_path: Path to the input image
        x: X coordinate of the point to segment
        y: Y coordinate of the point to segment
        output_dir: Optional output directory for segmentation results

    Returns:
        Dict with 'mask' and 'output' paths, or 'error' key if failed
    """
    debug_print("=" * 60, "INFO")
    debug_print("RUNNING SAM SEGMENTATION", "INFO")
    debug_print("=" * 60, "INFO")
    debug_print(f"Image: {image_path}", "INFO")
    debug_print(f"Point: ({x}, {y})", "INFO")

    if output_dir is None:
        output_dir = str(WORKSPACE_OUTPUT_DIR / "segmentation")

    try:
        url = f"{WORKSPACE_SERVER}/sam/segment"
        payload = {
            "image": image_path,
            "x": x,
            "y": y,
            "output_dir": output_dir,
        }

        debug_print(f"POST {url}", "REQUEST")
        debug_print(f"Payload: {json.dumps(payload, indent=2)}", "REQUEST")

        response = requests.post(url, json=payload, timeout=120)

        debug_print(f"Response status: {response.status_code}", "RESPONSE")

        if response.status_code != 200:
            error_msg = f"SAM segmentation failed with status {response.status_code}"
            debug_print(error_msg, "ERROR")
            debug_print(f"Response: {response.text}", "ERROR")
            return {"error": error_msg, "response": response.text}

        result = response.json()
        debug_print(f"SAM result: {json.dumps(result, indent=2)}", "RESPONSE")

        if result.get("error"):
            return {"error": result.get("error")}

        return result

    except requests.exceptions.Timeout:
        error_msg = "SAM segmentation request timed out"
        debug_print(error_msg, "ERROR")
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"SAM segmentation error: {str(e)}"
        debug_print(error_msg, "ERROR")
        traceback.print_exc()
        return {"error": error_msg}


def run_inpaint_removal(
    image_path: str,
    mask_path: str,
) -> Dict[str, Any]:
    """
    Run LaMa inpainting to remove objects from an image using a mask.

    Args:
        image_path: Path to the input image
        mask_path: Path to the mask image (white areas will be inpainted)

    Returns:
        Dict with 'output_images' list, or 'error' key if failed
    """
    debug_print("=" * 60, "INFO")
    debug_print("RUNNING INPAINT REMOVAL", "INFO")
    debug_print("=" * 60, "INFO")
    debug_print(f"Image: {image_path}", "INFO")
    debug_print(f"Mask: {mask_path}", "INFO")

    try:
        url = f"{WORKSPACE_SERVER}/comfy/inpaint"
        payload = {
            "image": image_path,
            "mask": mask_path,
        }

        debug_print(f"POST {url}", "REQUEST")
        debug_print(f"Payload: {json.dumps(payload, indent=2)}", "REQUEST")

        response = requests.post(url, json=payload, timeout=300)

        debug_print(f"Response status: {response.status_code}", "RESPONSE")

        if response.status_code != 200:
            error_msg = f"Inpaint removal failed with status {response.status_code}"
            debug_print(error_msg, "ERROR")
            debug_print(f"Response: {response.text}", "ERROR")
            return {"error": error_msg, "response": response.text}

        result = response.json()
        debug_print(f"Inpaint result: {json.dumps(result, indent=2)}", "RESPONSE")

        if result.get("error"):
            return {"error": result.get("error")}

        # Extract output path
        output_images = result.get("output_images", [])
        if output_images:
            result["output_image_path"] = output_images[0]
            debug_print(f"Output image: {output_images[0]}", "INFO")

        return result

    except requests.exceptions.Timeout:
        error_msg = "Inpaint removal request timed out"
        debug_print(error_msg, "ERROR")
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Inpaint removal error: {str(e)}"
        debug_print(error_msg, "ERROR")
        traceback.print_exc()
        return {"error": error_msg}


def run_object_removal_pipeline(
    image_path: str,
    x: int,
    y: int,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Complete object removal pipeline: SAM Segmentation -> LaMa Inpainting.

    Takes an image and a point (x, y), segments the object at that point
    using SAM2, then removes the segmented object using LaMa inpainting.

    Args:
        image_path: Path to the input image
        x: X coordinate of the object to remove
        y: Y coordinate of the object to remove
        output_dir: Optional output directory for intermediate files

    Returns:
        Dict with:
            - 'output_image_path': Path to the final inpainted image
            - 'mask_path': Path to the segmentation mask
            - 'segmentation_output': Path to the segmentation visualization
            - 'error': Error message if pipeline failed
    """
    debug_print("=" * 60, "INFO")
    debug_print("STARTING OBJECT REMOVAL PIPELINE", "INFO")
    debug_print("=" * 60, "INFO")
    debug_print(f"Image: {image_path}", "INFO")
    debug_print(f"Target point: ({x}, {y})", "INFO")

    if output_dir is None:
        output_dir = str(WORKSPACE_OUTPUT_DIR / "object_removal")

    # Step 1: SAM Segmentation
    debug_print("\n[STEP 1/2] Running SAM segmentation...", "INFO")
    sam_result = run_sam_segmentation(
        image_path=image_path,
        x=x,
        y=y,
        output_dir=output_dir,
    )

    if sam_result.get("error"):
        error_msg = f"SAM segmentation failed: {sam_result.get('error')}"
        debug_print(error_msg, "ERROR")
        return {"error": error_msg}

    mask_path = sam_result.get("mask")
    segmentation_output = sam_result.get("output")

    if not mask_path:
        error_msg = "SAM segmentation did not produce a mask"
        debug_print(error_msg, "ERROR")
        return {"error": error_msg}

    debug_print(f"Segmentation mask: {mask_path}", "INFO")
    debug_print(f"Segmentation output: {segmentation_output}", "INFO")

    # Step 2: LaMa Inpainting
    debug_print("\n[STEP 2/2] Running LaMa inpainting...", "INFO")
    inpaint_result = run_inpaint_removal(
        image_path=image_path,
        mask_path=mask_path,
    )

    if inpaint_result.get("error"):
        error_msg = f"Inpainting failed: {inpaint_result.get('error')}"
        debug_print(error_msg, "ERROR")
        return {
            "error": error_msg,
            "mask_path": mask_path,
            "segmentation_output": segmentation_output,
        }

    output_image_path = inpaint_result.get("output_image_path")

    if not output_image_path:
        error_msg = "Inpainting did not produce an output image"
        debug_print(error_msg, "ERROR")
        return {
            "error": error_msg,
            "mask_path": mask_path,
            "segmentation_output": segmentation_output,
        }

    debug_print("=" * 60, "INFO")
    debug_print("OBJECT REMOVAL PIPELINE COMPLETED", "INFO")
    debug_print("=" * 60, "INFO")
    debug_print(f"Output image: {output_image_path}", "INFO")

    return {
        "output_image_path": output_image_path,
        "mask_path": mask_path,
        "segmentation_output": segmentation_output,
        "sam_result": sam_result,
        "inpaint_result": inpaint_result,
    }


def test_object_removal(
    image_path: str = None,
    x: int = 150,
    y: int = 200,
):
    """Test the complete object removal pipeline."""
    print("\n--- Object Removal Pipeline Test ---")

    if image_path is None:
        image_path = DEFAULT_CONTENT_IMAGE

    debug_print(f"Image: {image_path}", "REQUEST")
    debug_print(f"Target point: ({x}, {y})", "REQUEST")

    result = run_object_removal_pipeline(
        image_path=image_path,
        x=x,
        y=y,
    )

    if result.get("error"):
        debug_print(f"Pipeline error: {result.get('error')}", "ERROR")
    else:
        debug_print(f"Pipeline result: {json.dumps(result, indent=2)}", "RESPONSE")

    return result


def ai_editing_flow(image_path, user_prompt):
    """Legacy test function - wraps run_ai_editing_pipeline with logging."""
    print("\n" + "=" * 60)
    print("STARTING AI EDITING FLOW")
    print("=" * 60)
    print(f"Image: {image_path}")
    print(f"Prompt: {user_prompt}")

    result = run_ai_editing_pipeline(image_path, user_prompt)

    if result.get("error"):
        print(f"\nERROR: {result.get('error')}")
        if result.get("response"):
            print(f"Response: {result.get('response')}")
    else:
        print("\n" + "=" * 60)
        print("FLOW COMPLETE")
        print("=" * 60)
        output_path = result.get("output_image_path") or result.get("output", "N/A")
        print(f"Output: {output_path}")

    return result


def execute_tool(tool, image_path, prompt, params, ai_suggestions):
    """Execute a specific tool based on classification."""
    debug_print(f"Executing tool: {tool}", "INFO")

    tool_endpoints = {
        "style-transfer-text": "/style-transfer/text",
        "style-transfer-ref": "/style-transfer/ref",
        "color-grading": "/color-grading",
        "segmentation": "/sam/segment",
        "comfy-edit": "/comfy/edit",
        "comfy-remix": "/comfy/remix",
        "comfy-inpaint": "/comfy/inpaint",
    }

    endpoint = tool_endpoints.get(tool)

    if not endpoint:
        error_msg = f"Unknown tool: {tool}"
        debug_print(error_msg, "ERROR")
        return {"error": error_msg}

    debug_print(f"Using endpoint: {endpoint}", "INFO")

    payload = build_tool_payload(tool, image_path, prompt, params)
    debug_print(f"Tool payload: {json.dumps(payload, indent=2)}", "REQUEST")

    try:
        url = f"{WORKSPACE_SERVER}{endpoint}"
        debug_print(f"POST {url}", "REQUEST")

        response = requests.post(url, json=payload, timeout=120)

        debug_print(f"Tool response status: {response.status_code}", "RESPONSE")

        if response.status_code != 200:
            error_msg = f"Tool API returned {response.status_code}"
            debug_print(error_msg, "ERROR")
            debug_print(f"Response: {response.text}", "ERROR")
            return {"error": error_msg, "response": response.text}

        result = response.json()
        debug_print(f"Tool result: {json.dumps(result, indent=2)}", "RESPONSE")

        # Extract output path based on tool type
        output_path = None
        if tool == "style-transfer-text":
            output_paths = result.get("output_paths", {})
            output_path = (
                output_paths.get("composite")
                or output_paths.get("styled_only")
                or result.get("output_composite")
                or result.get("output_styled_only")
            )
        elif tool == "style-transfer-ref":
            output_paths = result.get("output_paths", {})
            output_path = (
                output_paths.get("composite")
                or output_paths.get("styled_only")
                or result.get("output_composite")
                or result.get("output_styled_only")
            )
        elif tool == "color-grading":
            output_paths = result.get("output_paths", {})
            outputs = result.get("outputs", {})
            output_path = (
                output_paths.get("output_image")
                or output_paths.get("composite")
                or outputs.get("ai_graded")
                or outputs.get("manual_graded")
            )
        elif tool == "segmentation":
            output_path = (
                result.get("output_path")
                or result.get("mask_path")
                or result.get("output")
                or result.get("mask")
            )
        elif tool in ("comfy-edit", "comfy-remix", "comfy-inpaint"):
            # ComfyUI endpoints return output_images list
            output_images = result.get("output_images", [])
            if output_images:
                output_path = output_images[0]  # Take first output
            debug_print(f"ComfyUI output images: {output_images}", "INFO")

        if output_path:
            result["output_image_path"] = output_path
            debug_print(f"Extracted output path: {output_path}", "INFO")
        else:
            debug_print("Warning: No output path found in result", "WARN")

        return result

    except requests.exceptions.Timeout:
        error_msg = "Tool execution timed out"
        debug_print(error_msg, "ERROR")
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Tool execution failed: {e}"
        debug_print(error_msg, "ERROR")
        traceback.print_exc()
        return {"error": error_msg}


def build_tool_payload(
    tool, image_path, prompt, params, output_dir: Optional[str] = None
):
    """Build payload for tool execution."""
    if output_dir is None:
        output_dir = str(WORKSPACE_OUTPUT_DIR)

    debug_print(f"Building payload for {tool}", "DEBUG")

    if tool == "style-transfer-text":
        payload = {
            "content": image_path,
            "style_text": params.get("style_text", prompt),
            "prompt": prompt,
            "output_dir": output_dir,
            "steps": params.get("steps", 50),
            "style_steps": params.get("style_steps", 25),
            "max_side": params.get("max_side", 1024),
            "negative_prompt": params.get("negative_prompt", ""),
        }
    elif tool == "style-transfer-ref":
        payload = {
            "content": image_path,
            "style": params.get("style_image", ""),
            "prompt": prompt,
            "output_dir": output_dir,
            "steps": params.get("steps", 50),
            "max_side": params.get("max_side", 1024),
            "negative_prompt": params.get("negative_prompt", ""),
        }
    elif tool == "color-grading":
        payload = {
            "image": image_path,
            "prompt": prompt,
            "mode": params.get("mode", "both"),
            "output_dir_images": output_dir,
            "output_dir_data": "/workspace/AIP/workspace/outputs/data",
        }
    elif tool == "segmentation":
        payload = {
            "image": image_path,
            "x": params.get("x", 150),
            "y": params.get("y", 200),
            "output_dir": "/workspace/AIP/workspace/outputs/segmentation",
        }
    elif tool == "comfy-edit":
        payload = {
            "image": image_path,
            "prompt": prompt,
        }
    elif tool == "comfy-remix":
        payload = {
            "image1": params.get("image1", image_path),
            "image2": params.get("image2", ""),
            "prompt": prompt,
        }
    elif tool == "comfy-inpaint":
        payload = {
            "image": image_path,
            "mask": params.get("mask", ""),
        }
    else:
        payload = {"image": image_path, "prompt": prompt}

    debug_print(f"Built payload: {json.dumps(payload, indent=2)}", "DEBUG")
    return payload


# =============================================================================
# MAIN TEST SUITE
# =============================================================================


def main():
    """Main test runner with options for individual tests or pipeline flow."""
    print("\n" + "=" * 60)
    print("AI IMAGE EDITING - TEST SUITE")
    print("=" * 60)
    debug_print(f"Server URL: {WORKSPACE_SERVER}", "INFO")
    debug_print(f"Output directory: {WORKSPACE_OUTPUT_DIR}", "INFO")

    # Check server health
    print("\n[HEALTH & STATUS CHECKS]")
    timed("health", check_health)
    timed("status", check_status)

    # Run individual endpoint tests
    print("\n" + "=" * 60)
    print("RUNNING INDIVIDUAL ENDPOINT TESTS")
    print("=" * 60)

    timed("style_transfer_text", test_style_transfer_text)
    timed("style_transfer_ref", test_style_transfer_ref)
    timed("color_grading", test_color_grading)
    timed("ai_suggestions", test_ai_suggestions)
    timed("classify_quick", test_classify_quick)
    timed("classify_full", test_classify_full)
    timed("sam_segment", test_sam_segment)

    # Optional: Test complete AI editing flow with different prompts
    run_pipeline_tests = False  # Set to True to test full pipeline
    if run_pipeline_tests:
        print("\n" + "=" * 60)
        print("TESTING COMPLETE AI EDITING FLOWS")
        print("=" * 60)

        test_prompts = [
            "make this image look like a vintage film photograph",
            "apply a soft pastel watercolor effect",
            "enhance the colors with cinematic teal and orange tones",
        ]

        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n\n{'#' * 60}")
            print(f"FLOW TEST {i}/{len(test_prompts)}")
            print("#" * 60)

            TEST_IMAGE = DEFAULT_CONTENT_IMAGE
            result = timed(f"flow_{i}", ai_editing_flow, TEST_IMAGE, prompt)

    # Print timing summary
    print("\n" + "=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)
    for name, duration in sorted(timings.items()):
        print(f"{name:30s}: {duration:7.3f} sec")
    print("=" * 60)

    total_time = sum(timings.values())
    print(f"{'TOTAL TIME':30s}: {total_time:7.3f} sec")
    print("=" * 60)


if __name__ == "__main__":
    main()
