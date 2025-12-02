"""
Main Entrypoint for AI Image Editing Pipeline
This script is called from the backend with task-specific arguments.

IMPORTANT: Run setup.py first to install all dependencies:
    python setup.py

Usage:
    python init.py --task ai_suggestions --image "path/to/image.jpg"
"""
import argparse
import os
import sys
import subprocess


def main():
    parser = argparse.ArgumentParser(description="AI Image Editing Pipeline")
    parser.add_argument("--task", required=True, 
                       choices=["ai_suggestions", "classify_prompt", "style_with_ref", 
                               "style_with_prompt", "color_grading"],
                       help="Task to execute")
    parser.add_argument("--image", default="", help="Input image path (or comma-separated paths)")
    parser.add_argument("--prompt", default="", help="Text prompt for the task")
    parser.add_argument("--output_dir", default="", help="Custom output directory (optional)")
    
    args = parser.parse_args()
    
    # Get workspace directory
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if virtual environment exists
    venv_dir = os.path.join(workspace_dir, ".venv")
    if os.name == "nt":  # Windows
        python_exe = os.path.join(venv_dir, "Scripts", "python.exe")
    else:  # Linux/Mac
        python_exe = os.path.join(venv_dir, "bin", "python")
    
    if not os.path.exists(python_exe):
        print("="*70)
        print("ERROR: Virtual environment not found!")
        print("="*70)
        print("\nPlease run setup first:")
        print("  python setup.py")
        print("\nThis will install all dependencies in one go.")
        print("="*70)
        sys.exit(1)
    
    # Set default output directories
    if not args.output_dir:
        args.output_dir = os.path.join(workspace_dir, "outputs", "images")
    
    output_dir_images = os.path.join(workspace_dir, "outputs", "images")
    output_dir_data = os.path.join(workspace_dir, "outputs", "data")
    os.makedirs(output_dir_images, exist_ok=True)
    os.makedirs(output_dir_data, exist_ok=True)
    
    # Map task to script
    task_scripts = {
        "ai_suggestions": os.path.join(workspace_dir, "tasks", "ai_suggestions.py"),
        "classify_prompt": os.path.join(workspace_dir, "tasks", "prompt_classifier.py"),
        "color_grading": os.path.join(workspace_dir, "tasks", "color_grading.py"),
        "style_with_ref": os.path.join(workspace_dir, "tasks", "style_transfer_ref.py"),
        "style_with_prompt": os.path.join(workspace_dir, "tasks", "style_transfer_text.py"),
    }
    
    if args.task not in task_scripts:
        print(f"Error: Task '{args.task}' not yet implemented")
        sys.exit(1)
    
    script_path = task_scripts[args.task]
    
    # Build command based on task
    if args.task == "ai_suggestions":
        cmd = [python_exe, script_path, "--image", args.image, "--output_dir", output_dir_data]
    elif args.task == "classify_prompt":
        cmd = [python_exe, script_path, "--prompt", args.prompt, "--output_dir", output_dir_data]
    elif args.task == "color_grading":
        cmd = [python_exe, script_path, "--image", args.image, "--prompt", args.prompt, 
               "--output_dir_images", output_dir_images, "--output_dir_data", output_dir_data]
    elif args.task == "style_with_ref":
        # Expects comma-separated images: content,style
        images = args.image.split(",")
        if len(images) < 2:
            print("Error: style_with_ref requires two images (content,style)")
            sys.exit(1)
        cmd = [python_exe, script_path, "--content", images[0], "--style", images[1], 
               "--prompt", args.prompt, "--output_dir", output_dir_images]
    elif args.task == "style_with_prompt":
        # Single content image, style from text
        cmd = [python_exe, script_path, "--content", args.image, "--style_text", args.prompt,
               "--prompt", args.prompt, "--output_dir", output_dir_images]
    else:
        print(f"Error: Task '{args.task}' not yet implemented")
        sys.exit(1)
    
    # Run the task script in the venv
    print(f"\n{'='*60}")
    print(f"Running Task: {args.task}")
    print(f"{'='*60}\n")
    
    try:
        subprocess.check_call(cmd)
        print(f"\n{'='*60}")
        print(f"✓ Task completed successfully!")
        print(f"{'='*60}\n")
    except subprocess.CalledProcessError as e:
        print(f"\n{'='*60}")
        print(f"✗ Task failed with error code: {e.returncode}")
        print(f"{'='*60}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
