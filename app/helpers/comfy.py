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


def inject_image_filename(
    workflow: Dict[str, Any],
    image_filename: str,
    load_node_id: Optional[str] = None,
) -> None:
    """
    Update the workflow so that its LoadImage node points to the given filename.

    If load_node_id is provided, uses that node id.
    Otherwise:
      - tries to find the first node with class_type containing "LoadImage"
        or inputs["image"] being a string.
    """
    if load_node_id and load_node_id in workflow:
        workflow[load_node_id].setdefault("inputs", {})["image"] = image_filename
        return

    # Auto-discover a suitable node
    for node_id, node in workflow.items():
        class_type = node.get("class_type", "") or ""
        inputs = node.get("inputs", {}) or {}

        if "image" in inputs and isinstance(inputs["image"], str):
            inputs["image"] = image_filename
            return

        # Some nodes may have class_type like "LoadImage" or similar
        if "loadimage" in class_type.lower() or "load image" in class_type.lower():
            inputs["image"] = image_filename
            return

    raise ComfyError(
        "Could not find a LoadImage-like node to inject the filename into."
    )


def inject_prompt_text(
    workflow: Dict[str, Any],
    prompt_text: str,
    text_node_id: Optional[str] = None,
) -> None:
    """
    Optionally inject prompt text into the workflow.

    If text_node_id is given, tries to set workflow[text_node_id]["inputs"]["text"].
    Otherwise, will attempt to find any node with an 'inputs.text' field.
    """
    if not prompt_text:
        return

    if text_node_id and text_node_id in workflow:
        workflow[text_node_id].setdefault("inputs", {})["text"] = prompt_text
        return

    # Fallback: first node with "text" in inputs
    for node in workflow.values():
        inputs = node.get("inputs", {}) or {}
        if "text" in inputs:
            inputs["text"] = prompt_text
            return
    # If no text node found, we silently ignore; some workflows are purely img2img.


def queue_prompt(
    workflow: Dict[str, Any],
    client_id: Optional[str] = None,
) -> str:
    """
    Send the workflow to ComfyUI /prompt and return the prompt_id.

    The 'workflow' argument here should be the full graph dict, as exported from ComfyUI.
    """
    url = f"{COMFY_URL}/prompt"
    client_id = client_id or uuid.uuid4().hex

    payload = {
        "prompt": workflow,
        "client_id": client_id,
    }

    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
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
    load_node_id: Optional[str] = None,
    text_node_id: Optional[str] = None,
    timeout_seconds: float = 60.0,
) -> bytes:
    """
    End-to-end helper:

    1. Upload image to ComfyUI.
    2. Load workspace (workflow JSON) by name.
    3. Inject uploaded image filename into LoadImage node.
    4. Optionally inject prompt text.
    5. Queue /prompt.
    6. Wait for images.
    7. Return the FIRST image bytes.

    For now, this assumes a single-output flow, but you can adapt it to
    return all images if you want.
    """
    upload_info = upload_image_to_comfy(image_path)
    image_name = upload_info.get("name")
    if not image_name:
        raise ComfyError(f"ComfyUI upload response missing 'name': {upload_info}")

    workspace_name = workspace_name or DEFAULT_WORKSPACE
    workflow = load_workflow_template(workspace_name)

    inject_image_filename(workflow, image_name, load_node_id=load_node_id)

    if prompt_text:
        inject_prompt_text(workflow, prompt_text, text_node_id=text_node_id)

    prompt_id = queue_prompt(workflow)

    images = wait_for_images(prompt_id, timeout_seconds=timeout_seconds)

    if not images:
        raise ComfyError(f"No images returned from ComfyUI for prompt_id={prompt_id}")

    return images[0]
