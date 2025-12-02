"""
AI Suggestions Task
Based on: AI suggestions/moondream_suggestions.py
Input: image path
Output: JSON with image summary and list of editing prompts
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import re
import json
import os


def run_ai_suggestions(image_path, output_dir):
    """
    Run AI suggestions using moondream2 model.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save output JSON
        
    Returns:
        dict: Contains image_summary and edit_prompts
    """
    model_id = "vikhyatk/moondream2"
    print(f"Loading model {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True
    ).to(device)
    
    print(f"Loading image from {image_path}...")
    image = Image.open(image_path)
    
    # Hardcoded prompt from reference script
    query = """
You are an expert photo editor and creative prompt engineer for AI image-editing tools.

You are given a single image. Analyze it carefully and propose EDITING PROMPTS that can be directly used with an AI image-editing or generative editing model.

Requirements:

1. IMAGE-SPECIFIC ANALYSIS
   - First, briefly describe the image in 1–2 sentences: subject, lighting, mood, colors, and any obvious issues (too dark, flat contrast, blown highlights, color cast, cluttered background, etc.).

2. EDITING SUGGESTIONS AS PROMPTS
   - Then propose 6–10 short, ready-to-use editing prompts.
   - Each prompt must be a concrete visual edit, written as an instruction starting with a strong verb like: "Add", "Remove", "Replace", "Change", "Convert", "Increase", "Decrease", "Enhance", "Make".
   - Make them specific to THIS image (refer to the subject, background, colors, time of day, etc. when useful).
   - Cover a mix of:
     a) Basic photo enhancements (exposure, contrast, color balance, local adjustments).
     b) Generative edits that ADD or REMOVE elements (e.g., add props, change clothing, remove distractions, replace background, change weather or time of day).
     c) Style or mood transformations (e.g., different artistic or photographic styles, but only if they suit the image).

3. VISUAL ONLY
   - Do NOT mention or suggest sound, music, video, or animation.
   - Only describe visual edits to the still image.

4. OUTPUT FORMAT (IMPORTANT)
   - Use this exact structure:

   IMAGE_SUMMARY:
   - <1–2 short bullet points describing the image and its current problems, if any>

   EDIT_PROMPTS:
   1. <editing prompt 1>
   2. <editing prompt 2>
   3. <editing prompt 3>
   4. <editing prompt 4>
   5. <editing prompt 5>
   6. <editing prompt 6>
   7. <editing prompt 7>
   8. <editing prompt 8>
   9. <editing prompt 9>
   10. <editing prompt 10>

Only output the fields IMAGE_SUMMARY and EDIT_PROMPTS in the format above.
"""
    
    print("Running model inference...")
    raw_output = model.answer_question(image, query, tokenizer, device=device)
    
    # Parse output using same logic as reference script
    summary_match = re.search(r"IMAGE_SUMMARY\s*:?(.+?)EDIT_PROMPTS", raw_output, re.S | re.I)
    if summary_match:
        image_summary = summary_match.group(1).strip()
        image_summary = re.sub(r"^[\-•\s]+", "", image_summary).strip()
    else:
        image_summary = ""
    
    prompts_section = raw_output.split("EDIT_PROMPTS:")[-1].strip()
    edit_prompts = re.findall(r"\d+\.\s*(.+)", prompts_section)
    edit_prompts = [p.strip() for p in edit_prompts]
    
    # Create output
    result = {
        "task": "ai_suggestions",
        "image_summary": image_summary,
        "edit_prompts": edit_prompts,
        "raw_output": raw_output
    }
    
    # Save to JSON file
    import time
    timestamp = int(time.time())
    output_filename = f"ai_suggestions_{timestamp}.json"
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")
    print(f"\nIMAGE SUMMARY: {image_summary}")
    print(f"\nEDIT PROMPTS ({len(edit_prompts)} found):")
    for i, prompt in enumerate(edit_prompts, 1):
        print(f"{i}. {prompt}")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Suggestions - Generate editing prompts for an image")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output_dir", default="../outputs/data", help="Output directory for results")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    run_ai_suggestions(args.image, args.output_dir)
