# VersaSound Troubleshooting Guide

Common issues and how to fix them.

## Visual Feature Extractor Issues

### Error: "Pixel dimension doesn't match the configuration"

**Cause**: The input video/image resolution doesn't match what the backbone model expects.

**Solution**: This should now be fixed automatically! The latest version automatically resizes inputs to the correct size (224x224 by default).

**If you're still seeing this error:**

1. **Update your installation:**
   ```bash
   cd /path/to/ComfyUI/custom_nodes/VersaSound
   git pull
   cd comfyui_extension
   python install.py --skip-models  # Only update code, not models
   ```

2. **Check your input format:**
   - ComfyUI images should be in [N, H, W, C] format (N frames, Height, Width, 3 channels)
   - If using latents, make sure VAE is connected
   - Check ComfyUI console for the actual input shape being received

3. **Manually set input size:**

   Edit `comfyui_extension/configs/default_config.yaml`:
   ```yaml
   visual_encoder:
     input_size: 224  # Try 224, 336, or 512
   ```

   Restart ComfyUI after changing config.

### Error: "Expected 3 channels (RGB), got X"

**Cause**: Input image has wrong number of channels.

**Solutions:**
- Make sure you're feeding RGB images, not grayscale or RGBA
- If using alpha channel, remove it first
- Check if you accidentally connected a mask instead of an image

### Input Shape Reference

The Visual Feature Extractor accepts:

**Option 1: Video frames (recommended)**
- Format: `[N, H, W, 3]` where N is number of frames
- Example: 24 frames of 512x512 = `[24, 512, 512, 3]`
- Will be automatically resized to `[1, 24, 3, 224, 224]` internally

**Option 2: Latents + VAE**
- Connect both latent output and VAE
- Will decode and process automatically

## Audio Generator Issues

### Error: "No module named 'diffusers'"

**Solution:**
```bash
pip install diffusers accelerate transformers
```

### Audio is silent or very quiet

**Solutions:**
1. Use the **Audio Refiner** node with normalization enabled
2. Increase `guidance_scale` (try 10-15)
3. Add a text prompt describing the expected audio
4. Check if `num_inference_steps` is too low (try 50-100)

### Audio doesn't match video timing

**Solution:**
Use the **Temporal Synchronizer** node:
```
Visual Features → Audio Generator → Temporal Synchronizer
                           ↓
                     Visual Features
```

## Memory Issues

### CUDA out of memory

**Solutions:**

1. **Reduce video resolution:**
   - Resize video before feeding to VersaSound
   - Use 512x512 or smaller

2. **Reduce frame count:**
   - Process shorter clips (5-10 seconds)
   - Use every Nth frame

3. **Enable memory optimization:**

   Edit `comfyui_extension/configs/default_config.yaml`:
   ```yaml
   memory:
     clear_cache_after_generation: true
     max_batch_size: 4  # Reduce from 8

   device:
     precision: "fp16"  # Use half precision
   ```

4. **Process in chunks:**
   - Split long video into multiple clips
   - Process each separately
   - Use Audio Blender to combine results

### Model loading is slow

**Solution:**
Models are cached after first load. First run is always slower.

To verify models are cached:
```python
from comfyui_extension.models.model_utils import model_cache
print(model_cache.list_cached())
```

## Installation Issues

### Models not downloading

**Manual download:**
```bash
cd comfyui_extension/checkpoints

# Create placeholder files for now
touch visual_encoder.safetensors
touch audio_generator.safetensors
touch temporal_aligner.safetensors
```

The extension will work with these placeholders (models will use lightweight alternatives).

### Import errors

**Check installation:**
```bash
cd comfyui_extension
python install.py --verify-only
```

Should show all dependencies as installed.

**Reinstall dependencies:**
```bash
pip install -r requirements.txt --force-reinstall
```

## Performance Issues

### Generation is very slow

**On CPU:**
- Expected behavior - CPU is 10-20x slower than GPU
- Consider using a GPU cloud service
- Reduce `num_inference_steps` to 20-30

**On GPU:**
1. Check if CUDA is actually being used:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   print(torch.cuda.current_device())
   ```

2. Enable mixed precision:
   ```yaml
   device:
     precision: "fp16"
     use_autocast: true
   ```

3. Reduce quality settings:
   - `num_inference_steps`: 30 instead of 50
   - Lower video resolution

## Workflow Issues

### Nodes not appearing in ComfyUI

**Solutions:**
1. Restart ComfyUI completely
2. Check ComfyUI console for errors
3. Verify extension is in correct location:
   ```
   ComfyUI/custom_nodes/VersaSound/
   ```
4. Check if `__init__.py` exists in `comfyui_extension/`

### Connection errors between nodes

**Solutions:**
- Make sure output types match input types
- Visual Feature Extractor → outputs VISUAL_FEATURES
- Audio Generator → requires VISUAL_FEATURES input
- Check node documentation for correct connections

## VAE Compatibility

### "Different VAEs give same error"

**Explanation:**
The error is likely not VAE-specific. The automatic resizing should work with any VAE.

**Supported VAEs:**
- ✅ SD 1.5 VAE
- ✅ SDXL VAE
- ✅ SVD (Stable Video Diffusion) VAE
- ✅ Any standard ComfyUI VAE

**If you keep getting errors:**
Try skipping the VAE entirely:
- Feed raw images directly instead of latents
- Or use the `LatentToVisualFeatures` node specifically

## Debug Mode

### Enable detailed logging

Edit `comfyui_extension/configs/default_config.yaml`:
```yaml
logging:
  level: "DEBUG"  # Change from "INFO"
  log_to_file: true
  log_file: "versasound_debug.log"
```

Then check the log file for detailed information about what's happening.

### Check tensor shapes

Add this to your workflow temporarily:
```python
print(f"Video shape: {video.shape}")
print(f"Video dtype: {video.dtype}")
print(f"Video min/max: {video.min():.3f}/{video.max():.3f}")
```

This will print in ComfyUI console.

## Getting Help

If none of these solutions work:

1. **Check the logs:**
   - ComfyUI console output
   - VersaSound log file (if enabled)

2. **Gather information:**
   - Python version
   - PyTorch version
   - CUDA version (if using GPU)
   - Full error message
   - Input tensor shapes

3. **Create an issue:**
   - Go to GitHub Issues
   - Include all information from step 2
   - Attach log files if possible

## Quick Fixes Checklist

- [ ] Restart ComfyUI
- [ ] Update VersaSound (`git pull`)
- [ ] Check input is RGB images (3 channels)
- [ ] Try with smaller resolution (512x512 or 224x224)
- [ ] Verify dependencies (`python install.py --verify-only`)
- [ ] Check logs for specific error messages
- [ ] Try with default config settings
- [ ] Clear Python cache (`rm -rf __pycache__`)
