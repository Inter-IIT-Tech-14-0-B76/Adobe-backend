from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import requests

from config import COMFY_URL, COMFY_WORKFLOW_DIR, DEFAULT_WORKSPACE


class ComfyError(Exception):
    """Generic error raised for ComfyUI-related failures."""

    pass


def upload_image_to_comfy(image_path: str) -> Dict[str, Any]:
    """
    Upload an image file to ComfyUI via /upload/image.

    Returns a dict like:
    {
        "name": "xyz.png",
        "type": "input",
        "subfolder": ""
    }
    """
    url = f"{COMFY_URL}/upload/image"
    try:
        with open(image_path, "rb") as f:
            files = {"image": (os.path.basename(image_path), f)}
            resp = requests.post(url, files=files, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        raise ComfyError(f"Failed to upload image to ComfyUI: {e}") from e


def load_workflow_template(workspace_name: str) -> Dict[str, Any]:
    """
    Load a workflow JSON based on a 'workspace' name.

    For workspace_name="default", expects a file:
        COMFY_WORKFLOW_DIR / "default.json"

    Returns the full workflow dict containing:
        - "workflow": dict of nodes
        - "input_fields": list of paths for image injection
        - "text_input_fields": list of paths for text/prompt injection
        - "description": optional description
    """
    workspace_name = workspace_name or DEFAULT_WORKSPACE
    wf_path = COMFY_WORKFLOW_DIR / f"{workspace_name}.json"
    if not wf_path.exists():
        raise ComfyError(f"Workflow template not found: {wf_path}")

    try:
        with wf_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise ComfyError(f"Failed to load workflow template {wf_path}: {e}") from e


def _traverse_and_set(
    root: Dict[str, Any],
    field_path: str,
    value: Any,
) -> None:
    """
    Traverse a nested dict/list structure and set a value at the given path.

    Args:
        root: The root dict (e.g., workflow["workflow"])
        field_path: Dot-separated path like "10.inputs.image"
        value: The value to set
    """
    if not isinstance(field_path, str) or not field_path:
        raise ComfyError(f"Invalid field path: {field_path!r}")

    parts = field_path.split(".")
    cur = root

    # Traverse to the parent of the target
    for i, seg in enumerate(parts[:-1]):
        if isinstance(cur, dict):
            if seg not in cur:
                raise ComfyError(f"Missing key '{seg}' while traversing '{field_path}'")
            cur = cur[seg]
        elif isinstance(cur, list) and seg.isdigit():
            idx = int(seg)
            if idx < 0 or idx >= len(cur):
                raise ComfyError(
                    f"Index {idx} out of bounds while traversing '{field_path}'"
                )
            cur = cur[idx]
        else:
            raise ComfyError(
                f"Unexpected structure at '{'.'.join(parts[: i + 1])}' "
                f"while traversing '{field_path}': {type(cur).__name__}"
            )

    # Set the final value
    final_key = parts[-1]
    if isinstance(cur, dict):
        cur[final_key] = value
    elif isinstance(cur, list) and final_key.isdigit():
        idx = int(final_key)
        if idx < 0 or idx >= len(cur):
            raise ComfyError(
                f"Index {idx} out of bounds for final key in '{field_path}'"
            )
        cur[idx] = value
    else:
        raise ComfyError(
            f"Cannot set '{field_path}': expected dict or list at final container, "
            f"got {type(cur).__name__}"
        )


def inject_image_filename(
    workflow: Dict[str, Any],
    image_filenames: List[str],
) -> None:
    """
    Inject image filenames into workflow according to workflow['input_fields'].

    Args:
        workflow: Full workflow dict with "workflow" and "input_fields" keys
        image_filenames: List of filenames to inject; must match length of input_fields

    This mutates `workflow` in-place.
    """
    input_fields = workflow.get("input_fields", [])

    if not input_fields:
        raise ComfyError("Workflow has no 'input_fields' defined")

    if len(input_fields) != len(image_filenames):
        raise ComfyError(
            f"Mismatch: {len(input_fields)} input_fields but {len(image_filenames)} filenames"
        )

    nodes_root = workflow.get("workflow")
    if not isinstance(nodes_root, dict):
        raise ComfyError("workflow['workflow'] must be a dict of nodes")

    for field_path, filename in zip(input_fields, image_filenames):
        _traverse_and_set(nodes_root, field_path, filename)


def inject_prompt_text(
    workflow: Dict[str, Any],
    prompt_text: str,
) -> None:
    """
    Inject prompt text into the workflow using 'text_input_fields' if available,
    otherwise fall back to finding any node with 'inputs.text'.

    Args:
        workflow: Full workflow dict
        prompt_text: The text/prompt to inject
    """
    if not prompt_text:
        return

    nodes_root = workflow.get("workflow", {})
    text_fields = workflow.get("text_input_fields", [])

    # If text_input_fields is defined, use it
    if text_fields:
        for field_path in text_fields:
            _traverse_and_set(nodes_root, field_path, prompt_text)
        return

    # Fallback: find first node with "text" in inputs
    for node in nodes_root.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if isinstance(inputs, dict) and "text" in inputs:
            inputs["text"] = prompt_text
            return

    # No text node found - silently ignore (some workflows are purely img2img)


def queue_prompt(
    workflow: Dict[str, Any],
    client_id: Optional[str] = None,
) -> str:
    """
    Send the workflow to ComfyUI /prompt and return the prompt_id.

    Args:
        workflow: Full workflow dict containing "workflow" key with the node graph
        client_id: Optional client identifier for tracking

    Returns:
        The prompt_id from ComfyUI
    """
    url = f"{COMFY_URL}/prompt"
    client_id = client_id or uuid.uuid4().hex

    # Extract the actual node graph from our wrapper structure
    prompt_data = workflow.get("workflow", workflow)

    payload = {
        "prompt": prompt_data,
        "client_id": client_id,
    }

    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError as e:
        # Try to get error details from response
        error_detail = ""
        try:
            error_json = e.response.json()
            error_detail = f": {error_json}"
        except Exception:
            error_detail = f": {e.response.text[:500]}" if e.response.text else ""
        raise ComfyError(f"ComfyUI rejected prompt{error_detail}") from e
    except Exception as e:
        raise ComfyError(f"Failed to queue prompt in ComfyUI: {e}") from e

    prompt_id = data.get("prompt_id")
    if not prompt_id:
        raise ComfyError(f"ComfyUI /prompt response missing 'prompt_id': {data}")

    return prompt_id


def fetch_prompt_history(prompt_id: str) -> Dict[str, Any]:
    """
    Fetch /history/{prompt_id} from ComfyUI.
    """
    url = f"{COMFY_URL}/history/{prompt_id}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        raise ComfyError(f"Failed to fetch history for prompt {prompt_id}: {e}") from e


def download_image_from_comfy(
    filename: str,
    subfolder: str = "",
    image_type: str = "output",
) -> bytes:
    """
    Download a single image from ComfyUI using /view endpoint.

    Returns raw image bytes.
    """
    params = {
        "filename": filename,
        "subfolder": subfolder,
        "type": image_type,
    }
    url = f"{COMFY_URL}/view"
    try:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        raise ComfyError(
            f"Failed to download image from ComfyUI (filename={filename}): {e}"
        ) from e


def wait_for_images(
    prompt_id: str,
    timeout_seconds: float = 60.0,
    poll_interval: float = 1.0,
) -> List[bytes]:
    """
    Poll ComfyUI /history/{prompt_id} until images are ready or timeout.

    Returns a list of image bytes (one element per output image found).
    """
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        history = fetch_prompt_history(prompt_id)
        if not history or prompt_id not in history:
            time.sleep(poll_interval)
            continue

        prompt_data = history[prompt_id]

        # Check for execution errors
        status_data = prompt_data.get("status", {})
        if status_data.get("status_str") == "error":
            messages = status_data.get("messages", [])
            raise ComfyError(f"ComfyUI execution failed: {messages}")

        outputs = prompt_data.get("outputs", {}) or {}

        images_bytes: List[bytes] = []

        for node_output in outputs.values():
            images_info = node_output.get("images", []) or []
            for img_info in images_info:
                filename = img_info.get("filename")
                subfolder = img_info.get("subfolder", "")
                img_type = img_info.get("type", "output")
                if filename:
                    img_bytes = download_image_from_comfy(
                        filename=filename, subfolder=subfolder, image_type=img_type
                    )
                    images_bytes.append(img_bytes)

        if images_bytes:
            return images_bytes

        time.sleep(poll_interval)

    raise ComfyError(f"Timed out waiting for ComfyUI images (prompt_id={prompt_id})")


def process_image_with_comfy(
    image_path: str,
    workspace_name: Optional[str] = None,
    prompt_text: Optional[str] = None,
    timeout_seconds: float = 60.0,
) -> bytes:
    """
    End-to-end helper:

    1. Upload image to ComfyUI.
    2. Load workspace (workflow JSON) by name.
    3. Inject uploaded image filename using input_fields.
    4. Optionally inject prompt text using text_input_fields.
    5. Queue /prompt.
    6. Wait for images.
    7. Return the FIRST image bytes.

    Args:
        image_path: Path to the input image file
        workspace_name: Name of workflow template (default: DEFAULT_WORKSPACE)
        prompt_text: Optional text prompt for generation
        timeout_seconds: Max time to wait for ComfyUI to complete

    Returns:
        Raw bytes of the first output image
    """
    upload_info = upload_image_to_comfy(image_path)
    image_name = upload_info.get("name")
    if not image_name:
        raise ComfyError(f"ComfyUI upload response missing 'name': {upload_info}")

    workspace_name = workspace_name or DEFAULT_WORKSPACE
    workflow = load_workflow_template(workspace_name)

    # Inject the uploaded filename (expects a list)
    inject_image_filename(workflow, [image_name])

    # Inject prompt if provided
    if prompt_text:
        inject_prompt_text(workflow, prompt_text)

    prompt_id = queue_prompt(workflow)

    images = wait_for_images(prompt_id, timeout_seconds=timeout_seconds)

    if not images:
        raise ComfyError(f"No images returned from ComfyUI for prompt_id={prompt_id}")

    return images[0]
