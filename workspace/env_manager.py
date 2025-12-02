"""
Environment Manager
Creates and manages virtual environments for different AI tasks.
Each task has its own venv to avoid dependency conflicts.
"""
import os
import subprocess
import sys
import venv


def ensure_venv_and_install(venv_dir, requirements_file):
    """
    Create a virtual environment and install requirements.
    
    Args:
        venv_dir: Path to virtual environment directory
        requirements_file: Path to requirements.txt file
        
    Returns:
        str: Path to python executable in venv
    """
    print(f"\n{'='*60}")
    print(f"Environment Setup")
    print(f"{'='*60}")
    
    # Create venv if it doesn't exist
    if not os.path.exists(venv_dir):
        print(f"Creating virtual environment at: {venv_dir}")
        builder = venv.EnvBuilder(with_pip=True)
        builder.create(venv_dir)
        print("✓ Virtual environment created")
    else:
        print(f"✓ Virtual environment already exists at: {venv_dir}")
    
    # Determine pip and python paths
    if os.name == "nt":  # Windows
        pip_exe = os.path.join(venv_dir, "Scripts", "pip.exe")
        python_exe = os.path.join(venv_dir, "Scripts", "python.exe")
    else:  # Linux/Mac
        pip_exe = os.path.join(venv_dir, "bin", "pip")
        python_exe = os.path.join(venv_dir, "bin", "python")
    
    if not os.path.exists(pip_exe):
        raise RuntimeError(f"pip not found in venv: {venv_dir}")
    
    # Upgrade pip, wheel, setuptools
    print("\nUpgrading pip, wheel, setuptools...")
    subprocess.check_call([python_exe, "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"])
    print("✓ Core packages upgraded")
    
    # Install requirements if present
    if os.path.exists(requirements_file):
        print(f"\nInstalling requirements from: {requirements_file}")
        subprocess.check_call([pip_exe, "install", "-r", requirements_file])
        print("✓ Requirements installed")
    else:
        print(f"⚠ Warning: requirements file not found: {requirements_file}")
    
    print(f"{'='*60}\n")
    return python_exe
