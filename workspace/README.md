# AI Image Editing Pipeline - Workspace

> **Production-ready AI image editing pipeline for mobile app backend**

This workspace provides a unified entrypoint for 5 different AI-powered image editing tasks, each with isolated virtual environments to prevent dependency conflicts.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Windows PowerShell (or bash on Linux/Mac)
- 8GB+ RAM (16GB+ recommended)
- GPU with CUDA support (optional but highly recommended for performance)

### Installation

1. **Clone the repository** (if not already done)
   ```bash
   cd "c:\Users\Shahu Patil\Documents\Adobe-AI-Pipelines\workspace"
   ```

2. **Run any task** - The system will automatically:
   - Create a task-specific virtual environment
   - Install required dependencies
   - Download necessary models (first run only)
   - Execute the task

### Basic Usage

```bash
# From the workspace directory, run:
python init.py --task <task_name> --image <path> --prompt "<prompt>"
```

## 📋 Available Tasks

### 1. AI Suggestions
Generate intelligent editing suggestions for an image.

```bash
python init.py --task ai_suggestions --image "photo.jpg"
```

**Input:** Single image  
**Output:** JSON file in `outputs/data/` with:
- Image summary (description of lighting, mood, issues)
- 6-10 ready-to-use editing prompts

**Example Output:**
```json
{
  "task": "ai_suggestions",
  "image_summary": "Portrait with soft natural lighting...",
  "edit_prompts": [
    "Increase brightness and exposure by 20%",
    "Remove background distractions",
    "Add warm color temperature",
    ...
  ]
}
```

---

### 2. Prompt Classifier
Classify user prompts into task categories.

```bash
python init.py --task classify_prompt --prompt "make it look like a Ghibli film"
```

**Input:** Text prompt  
**Output:** JSON file in `outputs/data/` with classification

**Categories:**
- `style` - Style transfer, artistic themes
- `color` - Color grading, brightness, contrast
- `edit` - Add/remove/replace objects
- `manual` - Crop, rotate, resize
- `remix` - Combine two images

**Example Output:**
```json
{
  "task": "prompt_classifier",
  "input_prompt": "make it look like a Ghibli film",
  "classification": "style"
}
```

---

### 3. Color Grading
AI-powered color grading with parameter extraction.

```bash
python init.py --task color_grading --image "photo.jpg" --prompt "enhance colors"
```

**Input:** Image + optional prompt  
**Output:** 
- Graded images in `outputs/images/`
- JSON with extracted parameters in `outputs/data/`

**Features:**
- Object-aware color grading (different recipes for sky, trees, people, etc.)
- Extracts numerical parameters (exposure, contrast, saturation, temperature, tint, gamma)
- Generates both AI-graded and parameter-based versions

**Example Output:**
```json
{
  "extracted_parameters": {
    "exposure": 0.0,
    "contrast": 15.5,
    "saturation": -5.2,
    "temperature": -3.8,
    "tint": 8.4,
    "gamma": 12.3
  },
  "detected_objects": ["person", "tree", "sky"],
  "outputs": {
    "ai_graded": "color_grading_ai_1234567890.png",
    "manual_graded": "color_grading_manual_1234567890.png"
  }
}
```

---

### 4. Style Transfer (with Reference Image)
Apply style from a reference image to content image.

```bash
python init.py --task style_with_ref --image "content.jpg,style.jpg" --prompt "apply cinematic style"
```

**Input:** Two images (content,style) + prompt  
**Output:** Styled images in `outputs/images/`

**Features:**
- Uses SDXL + ControlNet + IP-Adapter
- Preserves original background using rembg
- Only styles the foreground object

**Note:** Images are comma-separated with NO spaces

---

### 5. Style Transfer (with Text Prompt)
Generate style from text description and apply to image.

```bash
python init.py --task style_with_prompt --image "content.jpg" --prompt "oil painting style"
```

**Input:** Single image + text style description  
**Output:** Styled images in `outputs/images/`

**Features:**
- First generates a style reference image from text
- Then applies it using ControlNet + IP-Adapter
- Preserves original background

---

## 📁 Architecture

```
workspace/
├── init.py                          # Main entrypoint (called from backend)
├── env_manager.py                   # Virtual environment management
├── tasks/                           # Individual task implementations
│   ├── ai_suggestions.py           # ✓ Implemented
│   ├── prompt_classifier.py        # ✓ Implemented
│   ├── style_transfer_ref.py       # ✓ Implemented
│   ├── style_transfer_text.py      # ✓ Implemented
│   └── color_grading.py            # ✓ Implemented
├── outputs/                         # Output directory
│   ├── images/                      # Generated images
│   └── data/                        # JSON outputs and text data
├── requirements_ai_suggestions.txt  # ✓ Created
├── requirements_prompt_classifier.txt # ✓ Created
├── requirements_style_transfer.txt   # ✓ Created
└── requirements_color_grading.txt    # ✓ Created
```

## Tasks

### 1. AI Suggestions ✓
- **Input**: Single image
- **Output**: JSON with image summary and editing prompts
- **Reference**: `AI suggestions/moondream_suggestions.py`
- **Model**: vikhyatk/moondream2

### 2. Prompt Classifier ✓
- **Input**: Text prompt
- **Output**: Task classification (style/color/edit/manual/remix)
- **Reference**: `Prompt Classifier/flan_t5_base.ipynb`
- **Model**: google/flan-t5-base

### 3. Style Transfer with Reference Image ✓
- **Input**: Content image + Style image + Prompt
- **Output**: Stylized image with original background preserved
- **Reference**: `StyleTransfer/transfer_with_image.py`
- **Models**: SDXL + ControlNet + IP-Adapter + rembg

### 4. Style Transfer with Text Prompt ✓
- **Input**: Content image + Text prompt
- **Output**: Stylized image with original background preserved
- **Reference**: `StyleTransfer/transfer_with_text.py`
- **Models**: SDXL (text-to-image) + ControlNet + IP-Adapter + rembg

### 5. Color Grading ✓
- **Input**: Image + Prompt
- **Output**: Color-graded image + extracted parameters
- **Reference**: `ColorGradingPY/script.py`
- **Model**: facebook/mask2former-swin-base-coco-panoptic

## 🔧 Advanced Usage

### Custom Output Directory
```bash
python init.py --task ai_suggestions --image "photo.jpg" --output_dir "custom/path"
```

### Multiple Image Processing
For style transfer with reference, provide comma-separated paths:
```bash
python init.py --task style_with_ref --image "img1.jpg,img2.jpg" --prompt "stylize"
```

### Environment Variables
Each task creates its own isolated virtual environment:
- `.venv_ai_suggestions/` - For moondream model
- `.venv_classify_prompt/` - For FLAN-T5 model  
- `.venv_color_grading/` - For Mask2Former model
- `.venv_style_with_ref/` - For SDXL + ControlNet
- `.venv_style_with_prompt/` - For SDXL + ControlNet

## 🐍 Python Backend Integration

### Simple Integration
From your app backend, make a single subprocess call:

```python
import subprocess
import json
import os

def run_ai_task(task_name, image_path, prompt=""):
    """
    Execute an AI task and return results.
    
    Args:
        task_name: One of ['ai_suggestions', 'classify_prompt', 'color_grading', 
                          'style_with_ref', 'style_with_prompt']
        image_path: Path to input image (or comma-separated for style_with_ref)
        prompt: Text prompt for the task
        
    Returns:
        dict: Task results from JSON output
    """
    workspace_path = "c:/Users/Shahu Patil/Documents/Adobe-AI-Pipelines/workspace"
    
    cmd = [
        "python",
        os.path.join(workspace_path, "init.py"),
        "--task", task_name,
        "--image", image_path,
        "--prompt", prompt
    ]
    
    # Run the task
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Task failed: {result.stderr}")
    
    # For tasks that output JSON, read the latest file
    if task_name in ['ai_suggestions', 'classify_prompt', 'color_grading']:
        data_dir = os.path.join(workspace_path, "outputs", "data")
        json_files = sorted([f for f in os.listdir(data_dir) if f.startswith(task_name)])
        if json_files:
            with open(os.path.join(data_dir, json_files[-1])) as f:
                return json.load(f)
    
    return {"status": "success", "stdout": result.stdout}


# Example usage:
# 1. Get AI suggestions
suggestions = run_ai_task("ai_suggestions", "uploads/photo.jpg")
print(suggestions["edit_prompts"])

# 2. Classify user prompt
classification = run_ai_task("classify_prompt", "", prompt="make it cinematic")
print(classification["classification"])  # "style"

# 3. Color grade image
grading = run_ai_task("color_grading", "uploads/photo.jpg", "enhance")
print(grading["extracted_parameters"])

# 4. Style transfer
styled = run_ai_task("style_with_ref", "content.jpg,style.jpg", "apply style")
```

### FastAPI Integration Example

```python
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import subprocess
import os
import uuid

app = FastAPI()

WORKSPACE = "c:/Users/Shahu Patil/Documents/Adobe-AI-Pipelines/workspace"
UPLOAD_DIR = "uploads"

@app.post("/api/ai-suggestions")
async def ai_suggestions(file: UploadFile = File(...)):
    # Save uploaded file
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.jpg")
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Run task
    subprocess.run([
        "python", os.path.join(WORKSPACE, "init.py"),
        "--task", "ai_suggestions",
        "--image", file_path
    ], check=True)
    
    # Return JSON result
    data_dir = os.path.join(WORKSPACE, "outputs", "data")
    json_file = sorted([f for f in os.listdir(data_dir) if f.startswith("ai_suggestions")])[-1]
    
    return FileResponse(os.path.join(data_dir, json_file))


@app.post("/api/color-grading")
async def color_grading(file: UploadFile = File(...), prompt: str = ""):
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.jpg")
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    subprocess.run([
        "python", os.path.join(WORKSPACE, "init.py"),
        "--task", "color_grading",
        "--image", file_path,
        "--prompt", prompt
    ], check=True)
    
    # Return graded image
    img_dir = os.path.join(WORKSPACE, "outputs", "images")
    img_file = sorted([f for f in os.listdir(img_dir) if f.startswith("color_grading_ai")])[-1]
    
    return FileResponse(os.path.join(img_dir, img_file))
```

## ⚙️ How It Works

### Automatic Environment Management

The `init.py` entrypoint automatically handles:

1. **Virtual Environment Creation** - First-time setup creates isolated venv
2. **Dependency Installation** - Installs task-specific requirements
3. **Model Downloads** - Downloads AI models on first run (cached for future use)
4. **Task Execution** - Runs the requested task in isolated environment

### First Run vs Subsequent Runs

**First Run (slower):**
- Creates virtual environment (~30 seconds)
- Installs Python packages (~2-5 minutes)
- Downloads AI models (~5-15 minutes depending on task and internet speed)
- Executes task

**Subsequent Runs (faster):**
- Uses existing virtual environment
- Uses cached models
- Executes task immediately

### Disk Space Requirements

- **AI Suggestions:** ~2GB (moondream2 model)
- **Prompt Classifier:** ~1GB (FLAN-T5 base)
- **Color Grading:** ~600MB (Mask2Former)
- **Style Transfer:** ~7GB (SDXL + ControlNet + IP-Adapter)

**Total:** ~15-20GB for all tasks

### GPU Acceleration

Tasks automatically detect and use CUDA GPU if available:
- **With GPU:** 5-10x faster inference
- **Without GPU:** Still works, but slower (CPU mode)

To check GPU support:
```python
import torch
print(torch.cuda.is_available())  # Should return True if GPU is available
```

## 🔍 Troubleshooting

### Common Issues

**Issue: "CUDA out of memory"**
- Reduce image resolution (use smaller --max_side value)
- Close other GPU applications
- Use CPU mode by setting device to 'cpu' in task scripts

**Issue: "Model download is slow"**
- First run downloads large models (5-15 minutes is normal)
- Subsequent runs use cached models
- Ensure stable internet connection

**Issue: "pip install fails"**
- Upgrade pip: `python -m pip install --upgrade pip`
- Install Visual C++ Build Tools (Windows)
- Check Python version (3.8+ required)

**Issue: "Task-specific virtual environment not found"**
- Delete the `.venv_*` folder and re-run
- The script will recreate it automatically

### Logs and Debugging

Enable verbose output:
```bash
python init.py --task ai_suggestions --image "photo.jpg" 2>&1 | Tee-Object -FilePath "task.log"
```

Check virtual environment:
```bash
# Windows
.\.venv_ai_suggestions\Scripts\python.exe --version

# Linux/Mac
./.venv_ai_suggestions/bin/python --version
```

### Manual Installation (if auto-install fails)

```bash
# Create virtual environment manually
python -m venv .venv_ai_suggestions

# Activate it
.\.venv_ai_suggestions\Scripts\Activate.ps1  # Windows PowerShell
# OR
source .venv_ai_suggestions/bin/activate     # Linux/Mac

# Install requirements
pip install -r requirements_ai_suggestions.txt

# Run task directly
python tasks/ai_suggestions.py --image "photo.jpg" --output_dir "outputs/data"
```

## 📊 Performance Benchmarks

Approximate execution times (RTX 3060, 12GB VRAM):

| Task | First Run | Subsequent Runs |
|------|-----------|-----------------|
| AI Suggestions | 15 min | 30-45 sec |
| Prompt Classifier | 8 min | 5-10 sec |
| Color Grading | 12 min | 40-60 sec |
| Style Transfer (ref) | 20 min | 60-90 sec |
| Style Transfer (text) | 25 min | 90-120 sec |

*First run includes model download + environment setup*

## 🛡️ Production Deployment

### RunPod / Cloud GPU Setup

1. **Create RunPod instance** with CUDA support
2. **Clone repository**
   ```bash
   git clone https://github.com/Inter-IIT-Tech-14-0-B76/Adobe-AI-Pipelines.git
   cd Adobe-AI-Pipelines/workspace
   ```

3. **Pre-warm all environments** (recommended)
   ```bash
   # Run each task once to download models
   python init.py --task ai_suggestions --image "test.jpg"
   python init.py --task classify_prompt --prompt "test"
   python init.py --task color_grading --image "test.jpg"
   python init.py --task style_with_ref --image "test1.jpg,test2.jpg" --prompt "test"
   python init.py --task style_with_prompt --image "test.jpg" --prompt "test"
   ```

4. **Create snapshot** - Save the instance with all models cached

5. **Scale** - Deploy multiple instances from snapshot

### Docker Deployment (Optional)

Create `Dockerfile`:
```dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.10 python3-pip git
WORKDIR /app
COPY workspace/ /app/
RUN pip install -r requirements.txt

# Pre-download models (optional but recommended)
RUN python init.py --task ai_suggestions --image "dummy.jpg" || true

CMD ["python", "init.py"]
```

## 📝 Notes

- **Model Caching:** Models are cached in `~/.cache/huggingface/`
- **Concurrent Requests:** Each task holds models in memory - limit concurrent executions to avoid OOM
- **Image Formats:** Supports JPG, PNG, WebP (automatically converted to RGB)
- **Output Naming:** Uses Unix timestamps to avoid filename conflicts

## 📄 License

Refer to the main repository LICENSE file.

## 🤝 Contributing

This workspace is part of the Adobe-AI-Pipelines project. For contributions, please refer to the main repository.

## Development Status

- [x] Project structure created
- [x] Environment manager implemented  
- [x] AI Suggestions task implemented
- [x] Prompt Classifier task implemented
- [x] Style Transfer with reference image implemented
- [x] Style Transfer with text prompt implemented
- [x] Color Grading task implemented
- [x] Comprehensive documentation
- [x] Combined requirements file
- [ ] Docker containerization
- [ ] Unit tests
- [ ] CI/CD pipeline
