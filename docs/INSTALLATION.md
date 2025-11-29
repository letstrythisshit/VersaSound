# VersaSound Installation Guide

Complete guide to installing VersaSound for ComfyUI.

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
- [Troubleshooting](#troubleshooting)
- [Verification](#verification)
- [Updating](#updating)

## System Requirements

### Minimum Requirements

- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.8 or higher
- **RAM**: 8GB
- **Storage**: 10GB free space
- **GPU**: 4GB VRAM (or CPU, but slower)

### Recommended Requirements

- **OS**: Linux or Windows 11
- **Python**: 3.10 or higher
- **RAM**: 16GB+
- **Storage**: 20GB+ free space (for models and cache)
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **CUDA**: 11.8 or higher (for NVIDIA GPUs)

### Supported Hardware

**GPUs:**
- ✅ NVIDIA GPUs (CUDA) - Recommended
- ✅ Apple Silicon (MPS) - Supported
- ✅ AMD GPUs (ROCm) - Experimental
- ⚠️ CPU - Supported but very slow

## Installation Methods

### Method 1: Automatic Installation (Recommended)

1. **Navigate to ComfyUI custom_nodes directory:**
```bash
cd /path/to/ComfyUI/custom_nodes
```

2. **Clone VersaSound:**
```bash
git clone https://github.com/yourusername/VersaSound.git
cd VersaSound
```

3. **Run the installer:**
```bash
cd comfyui_extension
python install.py
```

The installer will:
- ✅ Check Python version
- ✅ Install required packages
- ✅ Download pretrained models (~2GB)
- ✅ Verify installation

4. **Restart ComfyUI**

### Method 2: Manual Installation

If the automatic installer doesn't work:

1. **Clone repository:**
```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/yourusername/VersaSound.git
cd VersaSound
```

2. **Install dependencies:**
```bash
cd comfyui_extension
pip install -r requirements.txt
```

3. **Create checkpoint directory:**
```bash
mkdir -p checkpoints
```

4. **Download models manually** (optional - they'll auto-download on first use):
```bash
# Visual Encoder (~800MB)
wget https://huggingface.co/versasound/visual-encoder-v1/resolve/main/model.safetensors \
  -O checkpoints/visual_encoder.safetensors

# Audio Generator (~1.2GB)
wget https://huggingface.co/versasound/audio-generator-v1/resolve/main/model.safetensors \
  -O checkpoints/audio_generator.safetensors

# Temporal Aligner (~200MB)
wget https://huggingface.co/versasound/temporal-aligner-v1/resolve/main/model.safetensors \
  -O checkpoints/temporal_aligner.safetensors
```

5. **Restart ComfyUI**

### Method 3: Development Installation

For developers who want to modify VersaSound:

1. **Clone with development extras:**
```bash
git clone https://github.com/yourusername/VersaSound.git
cd VersaSound
```

2. **Install in development mode:**
```bash
pip install -e ".[dev]"
```

3. **Install pre-commit hooks:**
```bash
pre-commit install
```

## Platform-Specific Instructions

### Windows

**Using Conda (Recommended):**
```bash
conda create -n versasound python=3.10
conda activate versasound
cd VersaSound/comfyui_extension
pip install -r requirements.txt
```

**Using PowerShell:**
```powershell
cd C:\path\to\ComfyUI\custom_nodes
git clone https://github.com/yourusername/VersaSound.git
cd VersaSound\comfyui_extension
python install.py
```

**Common Windows Issues:**
- If you get a "torch not found" error, install PyTorch first:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

### Linux

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/yourusername/VersaSound.git
cd VersaSound/comfyui_extension
python3 install.py
```

**CUDA Setup:**
Ensure CUDA is installed and available:
```bash
nvidia-smi
nvcc --version
```

### macOS (Apple Silicon)

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/yourusername/VersaSound.git
cd VersaSound/comfyui_extension
python3 install.py
```

**Note**: MPS (Metal Performance Shaders) will be automatically detected and used.

## Troubleshooting

### "Module not found" errors

**Solution**: Reinstall dependencies
```bash
cd comfyui_extension
pip install -r requirements.txt --force-reinstall
```

### CUDA out of memory

**Solutions**:
1. Reduce batch size in config
2. Enable gradient checkpointing
3. Use lower precision (fp16)

Edit `comfyui_extension/configs/default_config.yaml`:
```yaml
memory:
  max_batch_size: 4  # Reduce this
  clear_cache_after_generation: true

device:
  precision: "fp16"  # Use fp16 instead of fp32
```

### Models not downloading

**Solution**: Download manually using wget or browser
```bash
cd comfyui_extension/checkpoints

# Download from alternative mirror (if available)
wget <alternative_url> -O visual_encoder.safetensors
```

### ComfyUI doesn't show VersaSound nodes

**Checklist**:
1. ✅ Extension is in `ComfyUI/custom_nodes/VersaSound`
2. ✅ Dependencies are installed
3. ✅ ComfyUI was restarted after installation
4. ✅ Check ComfyUI console for error messages

**Check logs**:
Look for VersaSound messages in ComfyUI startup logs:
```
VersaSound v1.0.0 loaded successfully
Registered 6 nodes
```

### Import errors with transformers

**Solution**: Update transformers
```bash
pip install --upgrade transformers accelerate
```

### Slow performance on CPU

**Expected behavior**: CPU inference is ~10x slower than GPU

**Solutions**:
1. Use a GPU if available
2. Reduce video resolution
3. Reduce `num_inference_steps` in Audio Generator node
4. Process shorter clips

## Verification

### Verify Installation

Run the verification script:
```bash
cd comfyui_extension
python install.py --verify-only
```

Expected output:
```
✅ Python version: 3.10.x
✅ torch: Installed
✅ torchaudio: Installed
✅ transformers: Installed
✅ CUDA available: NVIDIA GeForce RTX 3090
✅ Model checkpoints: 3 found
✅ Installation completed successfully!
```

### Test in ComfyUI

1. Launch ComfyUI
2. Right-click on canvas → Add Node → VersaSound
3. You should see:
   - Visual Feature Extractor
   - Audio Generator
   - Temporal Synchronizer
   - Audio Refiner
   - Latent to Visual Features
   - Audio Blender

## Updating

### Update VersaSound

```bash
cd /path/to/ComfyUI/custom_nodes/VersaSound
git pull
cd comfyui_extension
pip install -r requirements.txt --upgrade
python install.py  # Re-download models if needed
```

### Update Dependencies Only

```bash
cd comfyui_extension
pip install -r requirements.txt --upgrade
```

## Uninstallation

To remove VersaSound:

```bash
cd /path/to/ComfyUI/custom_nodes
rm -rf VersaSound
```

Then restart ComfyUI.

## Getting Help

If you encounter issues:

1. Check this guide's troubleshooting section
2. Search [GitHub Issues](https://github.com/yourusername/VersaSound/issues)
3. Create a new issue with:
   - Your OS and Python version
   - Full error message
   - Steps to reproduce

## Next Steps

After installation:
- 📖 Read the [User Guide](USER_GUIDE.md)
- 🎬 Try the [Example Workflows](../examples/)
- 🎨 Explore [Fine-tuning](FINETUNING_GUIDE.md) for custom sounds
