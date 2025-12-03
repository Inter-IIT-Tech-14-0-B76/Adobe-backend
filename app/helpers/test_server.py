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


def run_ai_editing_pipeline(
    image_path: str,
    user_prompt: str,
    image_paths: Optional[list] = None,
    image_analyses: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Run the complete AI editing pipeline: Analyze -> Classify -> Execute Tool.

    Args:
        image_path: Path to the input image file (primary/first image)
        user_prompt: User's editing prompt/request
        image_paths: Optional list of all image paths (for multi-image scenarios like style_transfer_ref)
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
        elif tool == "default_mode" or tool == "style_transfer":
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
