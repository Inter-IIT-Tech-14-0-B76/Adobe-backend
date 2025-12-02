"""
Color Grading Task
Based on: ColorGradingPY/script.py
Input: image path, optional prompt (for mode selection)
Output: Graded image + JSON with extracted parameters
"""
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import torch
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
import json
import os


def load_universal_model():
    """Loads the Mask2Former model and processor."""
    print("🔵 Loading Universal Segmenter...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "facebook/mask2former-swin-base-coco-panoptic"
    processor = Mask2FormerImageProcessor.from_pretrained(model_name)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model, device


def get_all_objects(image_path, processor, model, device):
    """Performs panoptic segmentation to identify all objects in the image."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)

    # Target size is required for correct mask scaling
    result = processor.post_process_panoptic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0]

    panoptic_seg = result["segmentation"].cpu().numpy()
    segments_info = result["segments_info"]

    found_objects = {}

    for segment in segments_info:
        label_id = segment["label_id"]
        mask_id = segment["id"]
        label_name = model.config.id2label[label_id]

        binary_mask = (panoptic_seg == mask_id)

        if label_name not in found_objects:
            found_objects[label_name] = binary_mask
        else:
            found_objects[label_name] = np.logical_or(found_objects[label_name], binary_mask)

    return image, found_objects


def get_grading_recipe(label_name):
    """Maps any COCO label to a specific Grading Style recipe."""
    label = label_name.lower()

    if any(x in label for x in ['tree', 'grass', 'bush', 'plant', 'flower', 'broccoli', 'vegetable', 'fruit']):
        return "BIO_FLORA"

    if any(x in label for x in ['sky', 'cloud', 'water', 'sea', 'river', 'lake', 'ocean', 'pool']):
        return "HYDRO_AERO"

    if any(x in label for x in ['person', 'face', 'dog', 'cat', 'bird', 'sheep', 'cow', 'horse', 'food', 'sandwich']):
        return "BIO_FAUNA"

    return "STRUCTURAL"


def apply_universal_grade(image_pil, object_masks):
    """Applies color grading adjustments based on identified object masks."""
    img_np = np.array(image_pil).astype(float)
    h, w, _ = img_np.shape

    print(f"\n🎨 Grading Report:")

    for label, binary_mask in object_masks.items():
        recipe = get_grading_recipe(label)
        print(f"   -> Found '{label}': Applying recipe [{recipe}]")

        mask_u8 = binary_mask.astype(np.uint8) * 255
        # Adaptive blur based on mask size to blend edges
        blur_amount = 45 if np.sum(binary_mask) > (h*w*0.1) else 15
        mask_blurred = cv2.GaussianBlur(mask_u8, (blur_amount, blur_amount), 0)
        mask = mask_blurred[:, :, None] / 255.0

        layer = img_np.copy()

        if recipe == "BIO_FLORA":
            layer[:,:,1] *= 1.20  # Boost Green
            layer[:,:,0] *= 0.90  # Reduce Red
            # Increase contrast
            layer = (layer - 128) * 1.1 + 128

        elif recipe == "HYDRO_AERO":
            layer[:,:,0] *= 0.85  # Reduce Red
            layer[:,:,2] *= 1.10  # Boost Blue
            # Increase contrast slightly more
            layer = (layer - 128) * 1.15 + 128

        elif recipe == "BIO_FAUNA":
            layer[:,:,0] *= 1.08  # Slight Warmth
            layer[:,:,2] *= 0.95  # Reduce Blue
            # Subtle gamma adjustment
            layer = 255 * (layer / 255) ** 0.9

        elif recipe == "STRUCTURAL":
            # High contrast and desaturation for structures
            mean_val = np.mean(layer)
            layer = (layer - mean_val) * 1.25 + mean_val
            hsv = cv2.cvtColor(np.clip(layer,0,255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(float)
            hsv[:,:,1] *= 0.9 # Desaturate
            layer = cv2.cvtColor(np.clip(hsv,0,255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(float)

        # Composite the layer back onto the image using the mask
        img_np = img_np * (1 - mask) + layer * mask

    return np.clip(img_np, 0, 255).astype(np.uint8)


def extract_parameters_from_ai_grading(image_path, processor, model, device):
    """
    Runs segmentation and calculates global slider parameters 
    based on the weighted area of identified objects.
    """
    original, found_objects = get_all_objects(image_path, processor, model, device)

    # Analyze composition
    total_pixels = original.size[0] * original.size[1]
    recipe_weights = {}

    for label, mask in found_objects.items():
        recipe = get_grading_recipe(label)
        pixel_count = np.sum(mask)
        weight = pixel_count / total_pixels

        if recipe not in recipe_weights:
            recipe_weights[recipe] = 0
        recipe_weights[recipe] += weight

    print(f"\n📊 Image Composition:")
    for recipe, weight in sorted(recipe_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"   {recipe}: {weight*100:.1f}%")

    # Base coefficients for each recipe type
    recipe_coeffs = {
        'BIO_FLORA': {'r': 0.90, 'g': 1.20, 'b': 1.0, 'contrast': 1.1, 'sat': 1.0, 'gamma': 1.0},
        'HYDRO_AERO': {'r': 0.85, 'g': 1.0, 'b': 1.10, 'contrast': 1.15, 'sat': 1.0, 'gamma': 1.0},
        'BIO_FAUNA': {'r': 1.08, 'g': 1.0, 'b': 0.95, 'contrast': 1.0, 'sat': 1.0, 'gamma': 0.9},
        'STRUCTURAL': {'r': 1.0, 'g': 1.0, 'b': 1.0, 'contrast': 1.25, 'sat': 0.9, 'gamma': 1.0}
    }

    # Calculate weighted averages for global parameters
    weighted = {'r': 0, 'g': 0, 'b': 0, 'contrast': 0, 'sat': 0, 'gamma': 0}

    for recipe, weight in recipe_weights.items():
        coeffs = recipe_coeffs[recipe]
        for k in weighted:
            weighted[k] += coeffs[k] * weight

    # Fallback to neutral if no objects found
    if sum(recipe_weights.values()) == 0:
         weighted = {'r': 1, 'g': 1, 'b': 1, 'contrast': 1, 'sat': 1, 'gamma': 1}

    # Convert coefficients to standard "slider" values
    params = {
        'exposure': 0.0,
        'contrast': round((weighted['contrast'] - 1.0) * 100, 2),
        'saturation': round((weighted['sat'] - 1.0) * 100, 2),
        'temperature': round((weighted['r'] - weighted['b']) * 100, 2),
        'tint': round((weighted['g'] - 1.0) * 100, 2),
        'gamma': round(-np.log2(weighted['gamma']) * 100 if weighted['gamma'] > 0 else 0, 2)
    }

    print(f"\n🎚️ EXTRACTED PARAMETERS:")
    for k, v in params.items():
        print(f"   {k.capitalize():12s}: {v:+.2f}")

    return original, found_objects, params


def apply_manual_grade(image_path, params):
    """
    Applies manual 'Lightroom-style' slider adjustments to an image.
    
    Args:
        image_path (str): Path to the input image.
        params (dict): Dictionary containing slider values (typically -100 to 100).
                       Expected keys: 'exposure', 'contrast', 'saturation', 
                       'temperature', 'tint', 'gamma'.
    """
    print(f"\n🎨 Applying Manual Grading with params: {params}")
    
    # Load image and convert to float for precise math
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img).astype(float) / 255.0

    # 1. EXPOSURE (Simple multiplication)
    exp_val = params.get('exposure', 0) / 50.0 
    img_np = img_np * (2 ** exp_val)

    # 2. TEMPERATURE (Warmth vs Cool)
    temp_val = params.get('temperature', 0) / 100.0
    img_np[:, :, 0] *= (1 + temp_val)     # Red
    img_np[:, :, 2] *= (1 - temp_val)     # Blue

    # 3. TINT (Green vs Magenta)
    tint_val = params.get('tint', 0) / 100.0
    img_np[:, :, 1] *= (1 + tint_val)

    # Clip values before converting back to PIL for Contrast/Sat
    img_np = np.clip(img_np, 0, 1)
    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))

    # 4. CONTRAST
    contrast_val = 1.0 + (params.get('contrast', 0) / 100.0)
    enhancer = ImageEnhance.Contrast(img_pil)
    img_pil = enhancer.enhance(contrast_val)

    # 5. SATURATION
    sat_val = 1.0 + (params.get('saturation', 0) / 100.0)
    enhancer = ImageEnhance.Color(img_pil)
    img_pil = enhancer.enhance(sat_val)

    # 6. GAMMA (Midtone brightness)
    gamma_param = params.get('gamma', 0)
    gamma_val = 1.0 - (gamma_param / 200.0) 
    
    img_np = np.array(img_pil).astype(float) / 255.0
    img_np = np.power(img_np, gamma_val)
    
    # Final cleanup
    img_np = np.clip(img_np, 0, 1) * 255.0
    return img_np.astype(np.uint8)


def run_color_grading(image_path, prompt, output_dir_images, output_dir_data, mode='ai'):
    """
    Run color grading on an image.
    
    Args:
        image_path: Path to input image
        prompt: Text prompt (can influence mode selection)
        output_dir_images: Directory to save output images
        output_dir_data: Directory to save JSON metadata
        mode: 'ai' (auto grading), 'manual' (with extracted params), or 'both'
        
    Returns:
        dict: Contains output paths and extracted parameters
    """
    import time
    timestamp = int(time.time())
    
    # Determine mode from prompt if needed
    if prompt:
        prompt_lower = prompt.lower()
        if any(word in prompt_lower for word in ['manual', 'slider', 'adjust', 'increase', 'decrease']):
            mode = 'manual'
    
    result = {
        "task": "color_grading",
        "input_image": image_path,
        "prompt": prompt,
        "mode": mode,
        "outputs": {}
    }
    
    # Load AI model
    processor, model, device = load_universal_model()
    
    # Extract parameters
    print("\n" + "="*70)
    print("EXTRACTING COLOR GRADING PARAMETERS")
    print("="*70)
    
    original_img, detected_objects, ai_params = extract_parameters_from_ai_grading(
        image_path, processor, model, device
    )
    
    result["extracted_parameters"] = ai_params
    result["detected_objects"] = list(detected_objects.keys())
    
    # Apply AI grading (pixel-level masking)
    if mode in ['ai', 'both']:
        print("\n" + "="*70)
        print("APPLYING AI GRADING")
        print("="*70)
        
        ai_graded_img = apply_universal_grade(original_img, detected_objects)
        ai_output_path = os.path.join(output_dir_images, f"color_grading_ai_{timestamp}.png")
        Image.fromarray(ai_graded_img).save(ai_output_path)
        result["outputs"]["ai_graded"] = ai_output_path
        print(f"\n✓ AI graded image saved: {ai_output_path}")
    
    # Apply manual grading with extracted parameters
    if mode in ['manual', 'both']:
        print("\n" + "="*70)
        print("APPLYING MANUAL GRADING")
        print("="*70)
        
        manual_result = apply_manual_grade(image_path, ai_params)
        manual_output_path = os.path.join(output_dir_images, f"color_grading_manual_{timestamp}.png")
        Image.fromarray(manual_result).save(manual_output_path)
        result["outputs"]["manual_graded"] = manual_output_path
        print(f"\n✓ Manual graded image saved: {manual_output_path}")
    
    # Save metadata JSON
    json_output_path = os.path.join(output_dir_data, f"color_grading_{timestamp}.json")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    result["metadata_file"] = json_output_path
    
    print("\n" + "="*70)
    print("COLOR GRADING COMPLETE")
    print("="*70)
    print(f"\nExtracted Parameters:")
    for k, v in ai_params.items():
        print(f"  {k.capitalize():12s}: {v:+.2f}")
    print(f"\nMetadata saved: {json_output_path}")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Color Grading - AI-powered color grading with parameter extraction")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--prompt", default="", help="Optional text prompt")
    parser.add_argument("--output_dir_images", default="../outputs/images", help="Output directory for images")
    parser.add_argument("--output_dir_data", default="../outputs/data", help="Output directory for JSON")
    parser.add_argument("--mode", choices=['ai', 'manual', 'both'], default='both',
                       help="Grading mode (default: both)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir_images, exist_ok=True)
    os.makedirs(args.output_dir_data, exist_ok=True)
    run_color_grading(args.image, args.prompt, args.output_dir_images, args.output_dir_data, args.mode)
