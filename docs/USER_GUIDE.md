# VersaSound User Guide

Complete guide to using VersaSound for video-to-audio generation in ComfyUI.

## Table of Contents

- [Introduction](#introduction)
- [Quick Start](#quick-start)
- [Node Reference](#node-reference)
- [Workflows](#workflows)
- [Parameters Guide](#parameters-guide)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Introduction

VersaSound is a universal video-to-audio generation extension for ComfyUI. It analyzes visual content and generates synchronized, high-quality audio matching the video.

### Key Features

- **Universal**: Works with ANY video content (no hardcoded scenarios)
- **Multiple Models**: Supports AudioLDM2, Stable Audio, and more
- **Temporal Sync**: Perfect audio-visual alignment
- **Flexible Workflows**: Raw video or latent-based pipelines
- **Production Ready**: Memory optimization, refinement, blending

### System Requirements

- **GPU**: NVIDIA with 8GB+ VRAM (recommended)
- **RAM**: 16GB+
- **Python**: 3.8+
- **ComfyUI**: Latest version

## Quick Start

### Basic Workflow

The simplest workflow to generate audio from video:

```
Load Video → Visual Feature Extractor → Audio Generator → Save Audio
```

**Steps**:
1. Load the `basic_workflow.json` from `examples/`
2. Connect your video source
3. Run the workflow
4. Audio is saved to output directory

### First-Time Setup

1. **Install VersaSound**:
   ```bash
   cd /path/to/ComfyUI/custom_nodes
   git clone https://github.com/yourusername/VersaSound.git
   cd VersaSound/comfyui_extension
   python install.py
   ```

2. **Restart ComfyUI**

3. **Verify Installation**:
   - Right-click canvas → Add Node → VersaSound
   - You should see all 6 nodes

## Node Reference

### 1. Visual Feature Extractor

**Purpose**: Extract visual features from video frames

**Inputs**:
- `video` (IMAGE) - ComfyUI video/image batch [N, H, W, C]
- `latents` (LATENT) - *Optional* Latent representation
- `vae` (VAE) - *Optional* VAE for decoding latents

**Outputs**:
- `visual_features` (VISUAL_FEATURES) - Extracted features

**Parameters**:
- `backbone` (str): Vision model to use
  - `"videomae"` (default) - Pretrained video understanding
  - `"clip"` - CLIP visual encoder
  - `"dinov2"` - Self-supervised vision features
- `input_size` (int): Input resolution (224, 336, 512)
  - Auto-resizes to this size

**Usage Notes**:
- Choose ONE input: Either `video` OR (`latents` + `vae`)
- Don't connect both video and latents!
- Auto-resizes and normalizes inputs

**Example**:
```
Video [65, 256, 256, 3] → Visual Features [1, 65, 768]
```

### 2. Audio Generator

**Purpose**: Generate audio from visual features

**Inputs**:
- `visual_features` (VISUAL_FEATURES) - From Visual Feature Extractor

**Outputs**:
- `audio` (AUDIO) - Generated audio waveform

**Parameters**:
- `model_type` (str): Audio generation model
  - `"audioldm2"` (default) - High quality, versatile
  - `"stable_audio"` - Faster, good quality
  - `"audiocraft"` - Creative, varied outputs
- `prompt` (str): Text description of desired audio
  - Example: "Footsteps on gravel, birds chirping"
- `negative_prompt` (str): What to avoid
  - Example: "music, speech, silence"
- `num_inference_steps` (int): Quality vs speed
  - 20-30: Fast, lower quality
  - 50: Balanced (default)
  - 75-100: High quality, slower
- `guidance_scale` (float): Adherence to input
  - 7-10: More creative
  - 10-12: Balanced (default: 10)
  - 12-15: Strict adherence
- `sample_rate` (int): Audio sample rate
  - 16000: Default (16kHz)
  - 22050: Better quality
  - 44100: CD quality
- `duration` (float): Audio duration in seconds
  - Should match video duration
- `seed` (int): Random seed
  - -1: Random each time
  - Fixed number: Reproducible results

**Usage Notes**:
- Text prompts significantly improve results
- Higher `guidance_scale` = more literal interpretation
- Duration should match your video length

**Example**:
```python
prompt: "Ocean waves crashing on beach, seagulls"
negative_prompt: "music, voices"
num_inference_steps: 50
guidance_scale: 10.0
```

### 3. Temporal Synchronizer

**Purpose**: Align audio timing with visual events

**Inputs**:
- `audio` (AUDIO) - Generated audio
- `visual_features` (VISUAL_FEATURES) - Original features
- `video` (IMAGE) - *Optional* Original video for timing

**Outputs**:
- `audio` (AUDIO) - Temporally synchronized audio

**Parameters**:
- `sync_method` (str): Synchronization algorithm
  - `"learned"` (default) - Neural alignment (best)
  - `"dtw"` - Dynamic Time Warping (precise)
  - `"cross_correlation"` - Cross-correlation (fast)
- `sync_strength` (float): How strongly to sync (0.0-1.0)
  - 0.0: No synchronization
  - 0.5: Balanced (default)
  - 1.0: Maximum synchronization
- `adaptive_sync` (bool): Auto-adjust to content
  - true (default): Adapts to video timing
  - false: Fixed synchronization

**Usage Notes**:
- Essential for videos with distinct timing (footsteps, impacts)
- `learned` method is best for most cases
- Increase `sync_strength` for more precise alignment

**When to Use**:
- ✅ Videos with rhythmic motion (walking, dancing)
- ✅ Impact events (door slam, object drop)
- ✅ Repeating actions (hammering, typing)
- ⚠️ Less critical for continuous ambient sounds

### 4. Audio Refiner

**Purpose**: Enhance and normalize audio output

**Inputs**:
- `audio` (AUDIO) - Raw or generated audio

**Outputs**:
- `audio` (AUDIO) - Refined audio

**Parameters**:
- `normalize` (bool): Normalize volume
  - true: Consistent volume (recommended)
  - false: Preserve original levels
- `denoise_strength` (float): Noise reduction (0.0-1.0)
  - 0.0: No denoising
  - 0.3-0.5: Subtle cleanup
  - 0.5-0.7: Moderate denoising
  - 0.7-1.0: Aggressive (may affect quality)
- `enhance_bass` (bool): Boost low frequencies
- `bass_boost` (float): Bass boost amount (1.0 = no change)
  - 1.0-1.3: Subtle to moderate boost
- `enhance_treble` (bool): Boost high frequencies
- `treble_boost` (float): Treble boost amount
  - 1.0-1.2: Subtle to moderate boost
- `compression_ratio` (float): Dynamic range compression
  - 0.0: No compression
  - 0.5-0.7: Moderate (more consistent volume)
  - 0.8-1.0: Heavy compression
- `limiter_threshold` (float): Prevent clipping
  - 1.0: Standard (default)
  - <1.0: More conservative

**Usage Notes**:
- Always enable `normalize` for consistent output
- Start with low `denoise_strength` (0.3-0.5)
- Bass/treble enhance based on content type
- Compression makes quiet/loud parts more even

**Recommended Presets**:

**Clean & Natural**:
```
normalize: true
denoise_strength: 0.3
enhance_bass: false
enhance_treble: false
compression_ratio: 0.5
```

**Enhanced Impact**:
```
normalize: true
denoise_strength: 0.5
enhance_bass: true
bass_boost: 1.3
compression_ratio: 0.7
```

**Clear Details**:
```
normalize: true
denoise_strength: 0.4
enhance_treble: true
treble_boost: 1.2
compression_ratio: 0.6
```

### 5. Latent to Visual Features

**Purpose**: Convert latents to visual features (alternative to Visual Feature Extractor)

**Inputs**:
- `latents` (LATENT) - Latent representation from VAE
- `vae` (VAE) - VAE for decoding

**Outputs**:
- `visual_features` (VISUAL_FEATURES) - Extracted features

**Parameters**:
- `backbone` (str): Same as Visual Feature Extractor
- `input_size` (int): Target resolution

**Usage Notes**:
- Use when working with latent diffusion pipelines
- Automatically decodes latents → extracts features
- More memory efficient than storing raw frames

**When to Use**:
- Stable Video Diffusion workflows
- AnimateDiff pipelines
- Any latent-based video generation
- Memory-constrained scenarios

### 6. Audio Blender

**Purpose**: Combine audio from multiple sources

**Inputs**:
- `audio1` (AUDIO) - First audio source
- `audio2` (AUDIO) - Second audio source
- `audio3`-`audio8` (AUDIO) - *Optional* Additional sources

**Outputs**:
- `audio` (AUDIO) - Blended audio

**Parameters**:
- `blend_mode` (str): How to combine audio
  - `"mix"`: Weighted average (default)
  - `"layer"`: Layer audio2 on top of audio1
  - `"crossfade"`: Smooth transition
- `weight` (float): Balance between sources (0.0-1.0)
  - 0.0: 100% audio1
  - 0.5: Equal mix
  - 1.0: 100% audio2
- `crossfade_duration` (float): Crossfade time (seconds)
  - Only used in crossfade mode
- `normalize_output` (bool): Prevent clipping
  - true (recommended): Normalize combined audio
  - false: Raw mix (may clip)

**Usage Notes**:
- Can blend 2-8 audio sources
- `mix` mode is good for layering complementary sounds
- `layer` mode preserves both sources more distinctly
- Always enable `normalize_output`

**Use Cases**:
- Background ambience + foreground actions
- Multiple visual sources → combined audio
- Audio transitions between scenes

## Workflows

### Workflow 1: Basic Video-to-Audio

**Goal**: Simple, fast audio generation

**Nodes**:
```
Load Video → Visual Feature Extractor → Audio Generator → Save Audio
```

**Settings**:
- Visual Extractor: `backbone="videomae"`
- Audio Generator: `num_inference_steps=50`, `guidance_scale=10`

**Best For**: Quick previews, testing

### Workflow 2: High-Quality Production

**Goal**: Professional-quality synchronized audio

**Nodes**:
```
Load Video → Visual Feature Extractor → Audio Generator →
Temporal Synchronizer → Audio Refiner → Save Audio
```

**Settings**:
- Visual Extractor: `backbone="videomae"`
- Audio Generator: `num_inference_steps=75`, `guidance_scale=12`
- Temporal Sync: `sync_method="learned"`, `sync_strength=0.5`
- Audio Refiner: `normalize=true`, `denoise_strength=0.5`

**Best For**: Final production, content creation

### Workflow 3: Latent Pipeline

**Goal**: Work with latent diffusion outputs

**Nodes**:
```
Video → VAE Encode → Latent to Visual Features → Audio Generator → Save Audio
```

**Settings**:
- Use same VAE that created latents
- Otherwise same as basic workflow

**Best For**: SVD, AnimateDiff, other latent workflows

### Workflow 4: Layered Soundscape

**Goal**: Complex, multi-source audio

**Nodes**:
```
Video 1 → Visual Extractor → Audio Generator → \
                                                 Audio Blender → Save Audio
Video 2 → Visual Extractor → Audio Generator → /
```

**Settings**:
- Different prompts for each generator
- Audio Blender: `blend_mode="mix"`, `weight=0.5`

**Best For**: Complex scenes, layered audio

## Parameters Guide

### Quality vs Speed

**Fast (Real-time-ish)**:
```
num_inference_steps: 20-30
guidance_scale: 7
Skip Audio Refiner
Skip Temporal Sync
```

**Balanced (Recommended)**:
```
num_inference_steps: 50
guidance_scale: 10
Basic Audio Refiner
Optional Temporal Sync
```

**High Quality (Production)**:
```
num_inference_steps: 75-100
guidance_scale: 12-15
Full Audio Refiner with all enhancements
Temporal Synchronizer
```

### Prompt Engineering

**Good Prompts**:
- Be specific: "Footsteps on wooden floor" not "walking"
- Include environment: "in a large empty room with echo"
- Mention texture: "crunching leaves", "splashing water"
- Add atmosphere: "distant thunder", "quiet forest"

**Negative Prompts**:
- Exclude unwanted: "no music, no speech"
- Remove artifacts: "no distortion, no clipping"
- Prevent silence: "not silent"

**Examples**:

| Video Content | Prompt | Negative Prompt |
|---------------|--------|-----------------|
| Person walking | "Footsteps on gravel path, birds in background" | "music, speech" |
| Ocean scene | "Ocean waves crashing, seagulls calling, wind" | "music, silence" |
| City street | "Traffic sounds, car horns, footsteps on pavement" | "music, speech" |
| Forest | "Wind through trees, rustling leaves, distant birds" | "music, machinery" |
| Waterfall | "Rushing water, splashing on rocks, ambient forest" | "music, speech" |

### Memory Optimization

**If getting OOM (Out of Memory) errors**:

1. **Reduce video resolution**:
   - 512x512 or smaller
   - Resize before Visual Extractor

2. **Shorten clips**:
   - Process 5-10 seconds at a time
   - Blend results together

3. **Lower inference steps**:
   - Reduce to 20-30

4. **Edit config** (`configs/default_config.yaml`):
   ```yaml
   memory:
     max_batch_size: 4
     clear_cache_after_generation: true
   device:
     precision: "fp16"
   ```

### Device Selection

**Automatic device selection**:
- NVIDIA GPU (CUDA): Automatically detected
- Apple Silicon (MPS): Automatically detected
- CPU: Fallback (very slow)

**Override in config**:
```yaml
device:
  device_type: "cuda"  # or "mps" or "cpu"
  precision: "fp16"     # or "fp32" or "bf16"
```

## Best Practices

### 1. Video Preparation

**Resolution**:
- Target: 512x512 or 256x256
- Higher resolution = more VRAM usage
- Quality plateaus above 512x512

**Duration**:
- Ideal: 5-10 seconds per clip
- Max recommended: 30 seconds
- Longer videos: split into chunks

**Frame Rate**:
- 24-30 FPS is ideal
- Auto-adapts to input framerate

**Format**:
- RGB images (3 channels)
- ComfyUI IMAGE format [N, H, W, C]
- Values in range [0, 1]

### 2. Audio Generation

**Prompts**:
- Always use text prompts
- Be specific and descriptive
- Include environmental context

**Guidance Scale**:
- Start with 10
- Increase for more literal results
- Decrease for more creativity

**Steps**:
- 50 is a good default
- Increase only if quality insufficient
- Diminishing returns above 100

### 3. Synchronization

**When to use Temporal Sync**:
- Videos with distinct events (impacts, footsteps)
- Rhythmic motion (walking, dancing)
- When audio timing is critical

**When to skip**:
- Continuous ambient sounds
- Static scenes
- When speed is priority

### 4. Refinement

**Always normalize**:
- Prevents volume inconsistencies
- Essential for production use

**Denoise conservatively**:
- Start with 0.3-0.5
- Too much removes detail
- Monitor for artifacts

**EQ based on content**:
- Bass boost: Impacts, footsteps, explosions
- Treble boost: Details, rustling, high-frequency events

### 5. Blending

**Separate concerns**:
- One generator for ambient
- One for specific actions
- Blend with appropriate weights

**Normalize output**:
- Always enable when blending
- Prevents clipping

**Experiment with modes**:
- Mix: General purpose
- Layer: Preserve both sources
- Crossfade: Transitions

## Troubleshooting

### No audio output

**Causes**:
- Video too short (<1 second)
- Guidance scale too low
- No text prompt

**Solutions**:
- Ensure video is at least 1 second
- Increase `guidance_scale` to 10-15
- Add descriptive text prompt

### Audio doesn't match video

**Solutions**:
- Use **Temporal Synchronizer** node
- Increase `sync_strength`
- Check video framerate is consistent

### Poor quality audio

**Solutions**:
- Increase `num_inference_steps` (75-100)
- Add descriptive text prompts
- Use **Audio Refiner** with normalization
- Increase `guidance_scale`

### Memory errors

**Solutions**:
- Reduce video resolution
- Process shorter clips
- Lower `num_inference_steps`
- Enable memory optimization in config

### Slow generation

**On CPU**:
- Expected - CPU is 10-20x slower
- Consider GPU cloud service
- Reduce steps to 20-30

**On GPU**:
- Check CUDA is being used (ComfyUI console)
- Enable mixed precision (fp16)
- Reduce quality settings

### Nodes not appearing

**Solutions**:
- Restart ComfyUI completely
- Check console for errors
- Verify installation: `python install.py --verify-only`
- Check extension path: `ComfyUI/custom_nodes/VersaSound/`

## Advanced Usage

### Custom Configurations

Edit `comfyui_extension/configs/default_config.yaml`:

```yaml
# Visual encoder settings
visual_encoder:
  backbone: "videomae"
  input_size: 224
  feature_dim: 768

# Audio generator settings
audio_generator:
  default_model: "audioldm2"
  num_inference_steps: 50
  guidance_scale: 10.0

# Memory optimization
memory:
  max_batch_size: 8
  clear_cache_after_generation: true
  use_gradient_checkpointing: false

# Device settings
device:
  device_type: "auto"  # auto, cuda, mps, cpu
  precision: "fp16"    # fp16, fp32, bf16
```

Restart ComfyUI after editing.

### Chaining Refiners

For maximum quality:

```
Audio Generator → Audio Refiner 1 → Audio Refiner 2 → Save
                  (denoise)         (EQ + normalize)
```

### Multi-source Blending

Chain multiple blenders for 3+ sources:

```
Audio 1 → \
           Blender 1 → \
Audio 2 → /             \
                         Blender 2 → Save
Audio 3 → /
```

### Integration with Other Nodes

VersaSound works with:
- Video generation nodes (SVD, AnimateDiff)
- Image processing (upscalers, filters)
- Audio processing (ComfyUI audio nodes)
- Custom VAEs

## Next Steps

- Try the [Example Workflows](../examples/)
- Read the [Troubleshooting Guide](TROUBLESHOOTING.md)
- Explore [Fine-tuning](FINETUNING_GUIDE.md) for custom sounds
- Check [Training Guide](TRAINING_GUIDE.md) for advanced users

## Support

- GitHub Issues: [Report bugs](https://github.com/yourusername/VersaSound/issues)
- Documentation: This guide + TROUBLESHOOTING.md
- Examples: See `examples/` directory
