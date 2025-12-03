"""
AI Editing Flow Test Script
Implements the complete flow: Analyze -> Classify -> Execute Tool
"""

import requests
import json
import time
import os

WORKSPACE_SERVER = "http://localhost:8000"
TEST_IMAGE = "/workspace/AIP/workspace/outputs/images/final_metric.png"
OUTPUT_DIR = "/workspace/AIP/workspace/outputs/images"

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


def ai_editing_flow(image_path, user_prompt):
    print("\n" + "="*60)
    print("STARTING AI EDITING FLOW")
    print("="*60)
    print(f"Image: {image_path}")
    print(f"Prompt: {user_prompt}")
    
    print("\n[STEP 1/3] Analyzing image...")
    try:
        analyze_resp = requests.post(
            f"{WORKSPACE_SERVER}/ai-suggestions",
            json={"image": image_path},
            timeout=60
        )
        
        if analyze_resp.status_code != 200:
            print(f"Analysis failed: {analyze_resp.status_code}")
            return {"error": "Image analysis failed", "response": analyze_resp.text}
        
        analysis_data = analyze_resp.json()
        print("Image analysis complete")
        print(f"Suggestions: {json.dumps(analysis_data.get('suggestions', {}), indent=2)}")
        
        ai_suggestions = analysis_data.get('suggestions', {})
        
    except Exception as e:
        print(f"Analysis error: {e}")
        return {"error": str(e)}
    
    print("\n[STEP 2/3] Classifying task and extracting parameters...")
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
            print(f"Classification failed: {classify_resp.status_code}")
            return {"error": "Classification failed", "response": classify_resp.text}
        
        classification_data = classify_resp.json()
        print("Classification complete")
        print(f"Tool: {classification_data.get('classification', {})}")
        
        classification = classification_data.get('classification', {})
        tool = classification.get('tool', 'unknown')
        params = classification.get('parameters', {})
        
        print(f"\nSelected Tool: {tool}")
        print(f"Parameters: {json.dumps(params, indent=2)}")
        
    except Exception as e:
        print(f"Classification error: {e}")
        return {"error": str(e)}
    
    print(f"\n[STEP 3/3] Executing tool: {tool}...")
    
    try:
        result = execute_tool(
            tool=tool,
            image_path=image_path,
            prompt=user_prompt,
            params=params,
            ai_suggestions=ai_suggestions
        )
        
        if result.get('error'):
            print(f"Tool execution failed: {result['error']}")
            return result
        
        print("Tool execution complete")
        print(f"\n{'='*60}")
        print("FLOW COMPLETE")
        print('='*60)
        print(f"Output: {result.get('output', 'N/A')}")
        
        return result
        
    except Exception as e:
        print(f"Tool execution error: {e}")
        return {"error": str(e)}


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
    
    print(f"Calling endpoint: {endpoint}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
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
        
        return response.json()
        
    except Exception as e:
        return {"error": f"Tool execution failed: {e}"}


def build_tool_payload(tool, image_path, prompt, params):
    if tool == 'style-transfer-text':
        return {
            "content": image_path,
            "style_text": params.get('style_text', prompt),
            "prompt": prompt,
            "output_dir": OUTPUT_DIR,
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
            "output_dir": OUTPUT_DIR,
            "steps": params.get('steps', 50),
            "max_side": params.get('max_side', 1024),
            "negative_prompt": params.get('negative_prompt', '')
        }
    
    elif tool == 'color-grading':
        return {
            "image": image_path,
            "prompt": prompt,
            "mode": params.get('mode', 'both'),
            "output_dir_images": OUTPUT_DIR,
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


def test_style_transfer_text():
    print("\n[TEST] Style Transfer (Text)")
    r = requests.post(
        f"{WORKSPACE_SERVER}/style-transfer/text",
        json={
            "content": TEST_IMAGE,
            "style_text": "soft pastel art style with gentle tones",
            "prompt": "apply a smooth dreamy pastel look",
            "output_dir": OUTPUT_DIR,
            "steps": 40,
            "style_steps": 20
        },
        timeout=120
    )
    return pretty_print(r, "Style Transfer Text Result")


def test_style_transfer_ref():
    print("\n[TEST] Style Transfer (Reference)")
    style_img = "/workspace/AIP/workspace/outputs/images/style_transfer_text_generated_style_1764690427.png"
    
    r = requests.post(
        f"{WORKSPACE_SERVER}/style-transfer/ref",
        json={
            "content": TEST_IMAGE,
            "style": style_img,
            "prompt": "match lighting and texture from the reference",
            "output_dir": OUTPUT_DIR,
            "steps": 50
        },
        timeout=120
    )
    return pretty_print(r, "Style Transfer Reference Result")


def test_color_grading():
    print("\n[TEST] Color Grading")
    r = requests.post(
        f"{WORKSPACE_SERVER}/color-grading",
        json={
            "image": TEST_IMAGE,
            "prompt": "cinematic teal shadows and warm highlights",
            "mode": "both",
            "output_dir_images": OUTPUT_DIR
        },
        timeout=120
    )
    return pretty_print(r, "Color Grading Result")


def test_ai_suggestions():
    print("\n[TEST] AI Suggestions")
    r = requests.post(
        f"{WORKSPACE_SERVER}/ai-suggestions",
        json={"image": TEST_IMAGE},
        timeout=60
    )
    return pretty_print(r, "AI Suggestions Result")


def test_classify():
    print("\n[TEST] Classify")
    r = requests.post(
        f"{WORKSPACE_SERVER}/classify",
        json={
            "prompt": "make this image look like a vintage film photograph",
            "quick": False
        },
        timeout=30
    )
    return pretty_print(r, "Classification Result")


def test_sam_segment():
    print("\n[TEST] SAM Segmentation")
    r = requests.post(
        f"{WORKSPACE_SERVER}/sam/segment",
        json={
            "image": TEST_IMAGE,
            "x": 150,
            "y": 200,
            "output_dir": "/workspace/AIP/workspace/outputs/segmentation"
        },
        timeout=120
    )
    return pretty_print(r, "SAM Segmentation Result")


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