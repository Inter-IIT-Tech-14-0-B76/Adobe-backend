"""
Prompt Classifier Task
Based on: Prompt Classifier/flan_t5_base.ipynb
Input: text prompt
Output: JSON with classification (style/color/edit/manual/remix)
"""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import json
import os


CLASSIFICATION_PROMPT = """
You are a classifier for an AI photo editing app.

Classify the user's request into EXACTLY ONE of these categories.
Output only the category word, with no explanation:

1. style  → style transfer, artistic theme (sketch, anime, ghibli, cyberpunk, oil painting, 3D render, etc.), changing overall look by generating a stylized version.
2. color  → color grading, filter-like changes (brightness, exposure, contrast, temperature, saturation, shadows, highlights). No adding/removing content.
3. edit   → modify image content (remove object, add object, replace object, restore background, face swap, inpaint, outpaint).
4. manual → simple image operations (crop, rotate, flip, resize, blur, sharpen, add text).
5. remix  → combining two images together (mix, blend, merge, fuse, hybrid image creation, use image A + image B, merge faces, overlay one image onto another).

RULES:
- Always select exactly one category.
- If the user requests multiple things, choose the dominant/primary intention.
- Respond with ONLY one word from: style, color, edit, manual, remix.

User request: "{query}"
"""

ALLOWED_CATEGORIES = {"style", "color", "edit", "manual", "remix"}


def classify_request(user_prompt, tokenizer, model, device):
    """
    Classify a user prompt into one of the allowed categories.
    
    Args:
        user_prompt: Text prompt to classify
        tokenizer: FLAN-T5 tokenizer
        model: FLAN-T5 model
        device: Device (cuda/cpu)
        
    Returns:
        str: One of: style, color, edit, manual, remix
    """
    prompt = CLASSIFICATION_PROMPT.format(query=user_prompt.strip())
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    # Generate short output
    outputs = model.generate(
        **inputs,
        max_new_tokens=8,
        do_sample=False,      # deterministic
        temperature=0.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    resp = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
    
    # Sanitize: take last line/word that matches allowed categories
    tokens = [t.strip(" .,\n") for t in resp.split()]
    for t in tokens[::-1]:
        if t in ALLOWED_CATEGORIES:
            return t
    
    # Fallback: find substring
    for category in ALLOWED_CATEGORIES:
        if category in resp:
            return category
    
    # Final fallback
    return "edit"


def run_prompt_classifier(prompt, output_dir):
    """
    Run prompt classification using FLAN-T5 model.
    
    Args:
        prompt: Text prompt to classify
        output_dir: Directory to save output JSON
        
    Returns:
        dict: Contains classification result
    """
    MODEL_NAME = "google/flan-t5-base"
    
    print(f"Loading model {MODEL_NAME}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    
    print(f"Classifying prompt: '{prompt}'")
    classification = classify_request(prompt, tokenizer, model, device)
    
    # Create output
    result = {
        "task": "prompt_classifier",
        "input_prompt": prompt,
        "classification": classification,
        "categories": {
            "style": "style transfer, artistic themes",
            "color": "color grading, filter-like changes",
            "edit": "modify image content (add/remove/replace objects)",
            "manual": "simple operations (crop, rotate, resize)",
            "remix": "combining two images together"
        }
    }
    
    # Save to JSON file
    import time
    timestamp = int(time.time())
    output_filename = f"prompt_classifier_{timestamp}.json"
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")
    print(f"Classification: {classification}")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prompt Classifier - Classify user prompts into task categories")
    parser.add_argument("--prompt", required=True, help="Text prompt to classify")
    parser.add_argument("--output_dir", default="../outputs/data", help="Output directory for results")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    run_prompt_classifier(args.prompt, args.output_dir)
