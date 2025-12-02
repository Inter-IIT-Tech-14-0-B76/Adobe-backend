"""
One-time Setup Script
Install all dependencies for all tasks in a single virtual environment.
Run this once before using any tasks.

Usage:
    python setup.py
"""
import os
import sys
import subprocess
import venv


def main():
    print("="*70)
    print("AI Image Editing Pipeline - One-Time Setup")
    print("="*70)
    print("\nThis will create a single virtual environment with all dependencies.")
    print("This may take 10-15 minutes on first run.\n")
    
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.join(workspace_dir, ".venv")
    requirements_file = os.path.join(workspace_dir, "requirements.txt")
    
    # Create virtual environment
    if not os.path.exists(venv_dir):
        print(f"Creating virtual environment at: {venv_dir}")
        builder = venv.EnvBuilder(with_pip=True)
        builder.create(venv_dir)
        print("✓ Virtual environment created\n")
    else:
        print(f"✓ Virtual environment already exists at: {venv_dir}\n")
    
    # Determine pip and python paths
    if os.name == "nt":  # Windows
        pip_exe = os.path.join(venv_dir, "Scripts", "pip.exe")
        python_exe = os.path.join(venv_dir, "Scripts", "python.exe")
    else:  # Linux/Mac
        pip_exe = os.path.join(venv_dir, "bin", "pip")
        python_exe = os.path.join(venv_dir, "bin", "python")
    
    if not os.path.exists(pip_exe):
        print(f"✗ Error: pip not found in virtual environment")
        sys.exit(1)
    
    # Upgrade pip, wheel, setuptools
    print("Upgrading pip, wheel, setuptools...")
    subprocess.check_call([python_exe, "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"])
    print("✓ Core packages upgraded\n")
    
    # Install all requirements
    if os.path.exists(requirements_file):
        print(f"Installing all dependencies from: {requirements_file}")
        print("This may take several minutes...\n")
        subprocess.check_call([pip_exe, "install", "-r", requirements_file])
        print("\n✓ All dependencies installed successfully!\n")
    else:
        print(f"✗ Error: requirements.txt not found at {requirements_file}")
        sys.exit(1)
    
    print("="*70)
    print("Setup Complete!")
    print("="*70)
    print("\nYou can now run any task using:")
    print("  python init.py --task <task_name> --image <path> --prompt <prompt>\n")
    print("Available tasks:")
    print("  - ai_suggestions")
    print("  - classify_prompt")
    print("  - color_grading")
    print("  - style_with_ref")
    print("  - style_with_prompt")
    print("="*70)


if __name__ == "__main__":
    main()
