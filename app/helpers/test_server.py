"""
AI Editing Flow Helper Functions
Implements the complete flow: Analyze -> Classify -> Execute Tool
"""

import requests
import json
import time
import os
from pathlib import Path
from typing import Dict, Any, Optional

from config import WORKSPACE_SERVER, WORKSPACE_OUTPUT_DIR

timings = {}


def timed(name, func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    timings[name] = round(time.time() - start, 3)
    return result


def pretty_print(resp, title=""):
    if title:
        print(f"\n{'='*60}")
        print(f"{title}")
        print('='*60)
    
    try:
        data = resp.json()
        print(json.dumps(data, indent=2))
        return data
    except:
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.text}")
        return None


def check_health():
    print("\n[HEALTH CHECK]")
    r = requests.get(f"{WORKSPACE_SERVER}/health")
    return pretty_print(r)


def check_status():
    print("\n[SERVER STATUS]")
    r = requests.get(f"{WORKSPACE_SERVER}/status")
    return pretty_print(r)


def run_ai_editing_pipeline(image_path: str, user_prompt: str) -> Dict[str, Any]:
    """
    Run the complete AI editing pipeline: Analyze -> Classify -> Execute Tool.
    
    Args:
        image_path: Path to the input image file
        user_prompt: User's editing prompt/request
        
    Returns:
        Dict with 'error' key if failed, or result dict with output paths
    """
    try:
        analyze_resp = requests.post(
            f"{WORKSPACE_SERVER}/ai-suggestions",
            json={"image": image_path},
            timeout=60
        )
        
        if analyze_resp.status_code != 200:
            return {"error": "Image analysis failed", "response": analyze_resp.text}
        
        analysis_data = analyze_resp.json()
        ai_suggestions = analysis_data.get('suggestions', {})
        
    except Exception as e:
        return {"error": f"Analysis error: {str(e)}"}
    
    try:
        classify_resp = requests.post(
            f"{WORKSPACE_SERVER}/classify",
            json={
                "prompt": user_prompt,
                "quick": False
            },
            timeout=30
        )
        
        if classify_resp.status_code != 200:
            return {"error": "Classification failed", "response": classify_resp.text}
        
        classification_data = classify_resp.json()
        classification = classification_data.get('classification', {})
        tool = classification.get('tool', 'unknown')
        params = classification.get('parameters', {})
        
    except Exception as e:
        return {"error": f"Classification error: {str(e)}"}
    
    try:
        result = execute_tool(
            tool=tool,
            image_path=image_path,
            prompt=user_prompt,
            params=params,
            ai_suggestions=ai_suggestions
        )
        
        if result.get('error'):
            return result
        
        return result
        
    except Exception as e:
        return {"error": f"Tool execution error: {str(e)}"}


def ai_editing_flow(image_path, user_prompt):
    """Legacy test function - wraps run_ai_editing_pipeline with logging."""
    print("\n" + "="*60)
    print("STARTING AI EDITING FLOW")
    print("="*60)
    print(f"Image: {image_path}")
    print(f"Prompt: {user_prompt}")
    
    result = run_ai_editing_pipeline(image_path, user_prompt)
    
    if result.get('error'):
        print(f"Error: {result.get('error')}")
    else:
        print("Tool execution complete")
        print(f"\n{'='*60}")
        print("FLOW COMPLETE")
        print('='*60)
        print(f"Output: {result.get('output', 'N/A')}")
    
    return result


def execute_tool(tool, image_path, prompt, params, ai_suggestions):
    tool_endpoints = {
        'style-transfer-text': '/style-transfer/text',
        'style-transfer-ref': '/style-transfer/ref',
        'color-grading': '/color-grading',
        'segmentation': '/sam/segment',
    }
    
    endpoint = tool_endpoints.get(tool)
    
    if not endpoint:
        return {"error": f"Unknown tool: {tool}"}
    
    payload = build_tool_payload(tool, image_path, prompt, params)
    
    try:
        response = requests.post(
            f"{WORKSPACE_SERVER}{endpoint}",
            json=payload,
            timeout=120
        )
        
        if response.status_code != 200:
            return {
                "error": f"Tool API returned {response.status_code}",
                "response": response.text
            }
        
        result = response.json()
        
        output_path = None
        if tool == 'style-transfer-text':
            output_paths = result.get('output_paths', {})
            output_path = output_paths.get('composite') or output_paths.get('styled_only')
        elif tool == 'style-transfer-ref':
            output_paths = result.get('output_paths', {})
            output_path = output_paths.get('composite') or output_paths.get('styled_only')
        elif tool == 'color-grading':
            output_paths = result.get('output_paths', {})
            output_path = output_paths.get('output_image') or output_paths.get('composite')
        elif tool == 'segmentation':
            output_path = result.get('output_path') or result.get('mask_path')
        
        if output_path:
            result['output_image_path'] = output_path
        
        return result
        
    except Exception as e:
        return {"error": f"Tool execution failed: {e}"}


def build_tool_payload(tool, image_path, prompt, params, output_dir: Optional[str] = None):
    """Build payload for tool execution."""
    if output_dir is None:
        output_dir = str(WORKSPACE_OUTPUT_DIR)
    
    if tool == 'style-transfer-text':
        return {
            "content": image_path,
            "style_text": params.get('style_text', prompt),
            "prompt": prompt,
            "output_dir": output_dir,
            "steps": params.get('steps', 50),
            "style_steps": params.get('style_steps', 25),
            "max_side": params.get('max_side', 1024),
            "negative_prompt": params.get('negative_prompt', '')
        }
    
    elif tool == 'style-transfer-ref':
        return {
            "content": image_path,
            "style": params.get('style_image', ''),
            "prompt": prompt,
            "output_dir": output_dir,
            "steps": params.get('steps', 50),
            "max_side": params.get('max_side', 1024),
            "negative_prompt": params.get('negative_prompt', '')
        }
    
    elif tool == 'color-grading':
        return {
            "image": image_path,
            "prompt": prompt,
            "mode": params.get('mode', 'both'),
            "output_dir_images": output_dir,
            "output_dir_data": "/workspace/AIP/workspace/outputs/data"
        }
    
    elif tool == 'segmentation':
        return {
            "image": image_path,
            "x": params.get('x', 150),
            "y": params.get('y', 200),
            "output_dir": "/workspace/AIP/workspace/outputs/segmentation"
        }
    
    else:
        return {"image": image_path, "prompt": prompt}
# =============================================================================
# MAIN TEST SUITE
# =============================================================================
def main():
    """Main test runner"""
    print("\n" + "="*60)
    print("AI IMAGE EDITING - TEST SUITE")
    print("="*60)
    
    # Check server health
    timed("health_check", check_health)
    timed("status_check", check_status)
    
    # Test complete AI editing flow with different prompts
    test_prompts = [
        "make this image look like a vintage film photograph",
        "apply a soft pastel watercolor effect",
        "enhance the colors with cinematic teal and orange tones",
    ]
    
    print("\n" + "="*60)
    print("TESTING COMPLETE AI EDITING FLOWS")
    print("="*60)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n\n{'#'*60}")
        print(f"FLOW TEST {i}/{len(test_prompts)}")
        print('#'*60)
        
        TEST_IMAGE = "/workspace/AIP/workspace/outputs/images/final_metric.png"
        result = timed(
            f"flow_{i}",
            ai_editing_flow,
            TEST_IMAGE,
            prompt
        )
    
    # Optional: Run individual tool tests for debugging
    run_individual_tests = False
    if run_individual_tests:
        print("\n" + "="*60)
        print("RUNNING INDIVIDUAL TOOL TESTS")
        print("="*60)
        
        timed("test_ai_suggestions", test_ai_suggestions)
        timed("test_classify", test_classify)
        timed("test_color_grading", test_color_grading)
        timed("test_style_transfer_text", test_style_transfer_text)
        # timed("test_style_transfer_ref", test_style_transfer_ref)
        # timed("test_sam_segment", test_sam_segment)
    
    # Print timing summary
    print("\n" + "="*60)
    print("TIMING SUMMARY")
    print("="*60)
    for name, duration in timings.items():
        print(f"{name:30s}: {duration:7.3f} sec")
    print("="*60)
    
    total_time = sum(timings.values())
    print(f"{'TOTAL TIME':30s}: {total_time:7.3f} sec")
    print("="*60)


if __name__ == "__main__":
    main()