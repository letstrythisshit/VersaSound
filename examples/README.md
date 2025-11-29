# VersaSound Example Workflows

This directory contains example ComfyUI workflows demonstrating various use cases of VersaSound.

## Available Workflows

### 1. Basic Workflow (`basic_workflow.json`)

**Purpose**: Simple video-to-audio generation

**Nodes**:
- Load Video
- Visual Feature Extractor
- Audio Generator
- Save Audio

**Use Case**: Quick audio generation from video input

**Steps to Use**:
1. Load `basic_workflow.json` in ComfyUI
2. Connect your video source to "Load Video" node
3. Adjust `guidance_scale` (7-15) for audio quality
4. Run the workflow
5. Audio will be saved to output directory

### 2. Latent Workflow (`latent_workflow.json`)

**Purpose**: Generate audio from latent representations

**Nodes**:
- Load Video
- VAE Encode
- Latent to Visual Features
- Audio Generator
- Save Audio

**Use Case**: When working with latent diffusion pipelines (Stable Video Diffusion, AnimateDiff, etc.)

**Steps to Use**:
1. Load `latent_workflow.json` in ComfyUI
2. Connect your video pipeline to VAE encoder
3. Connect VAE to Latent to Visual Features node
4. Adjust audio generation parameters
5. Run the workflow

### 3. Advanced Workflow (`advanced_workflow.json`)

**Purpose**: High-quality audio with temporal synchronization

**Nodes**:
- Load Video
- Visual Feature Extractor
- Audio Generator
- Temporal Synchronizer
- Audio Refiner
- Save Audio

**Use Case**: Professional-quality audio that's perfectly synced to video timing

**Steps to Use**:
1. Load `advanced_workflow.json` in ComfyUI
2. Connect your video source
3. The Temporal Synchronizer ensures audio matches video timing
4. The Audio Refiner normalizes and enhances the output
5. Adjust refinement parameters:
   - `normalize`: Enable for consistent volume
   - `denoise_strength`: 0.3-0.7 for noise reduction
   - `enhance_bass`/`enhance_treble`: For frequency shaping
6. Run the workflow

### 4. Blending Workflow (`blending_workflow.json`)

**Purpose**: Combine audio from multiple visual sources

**Nodes**:
- Load Video 1
- Load Video 2
- Visual Feature Extractor (x2)
- Audio Generator (x2)
- Audio Blender
- Save Audio

**Use Case**: Create complex soundscapes by blending audio from different visual sources

**Steps to Use**:
1. Load `blending_workflow.json` in ComfyUI
2. Connect two different video sources
3. Generate audio from each
4. Use Audio Blender to combine:
   - `blend_mode`: "mix", "layer", or "crossfade"
   - `weight`: 0.0-1.0 (0.5 = equal mix)
   - `crossfade_duration`: Time for smooth transitions
5. Run the workflow

## Workflow Modifications

### Adjusting Quality

**For faster generation** (lower quality):
```
Audio Generator:
  num_inference_steps: 20-30
  guidance_scale: 7
```

**For better quality** (slower):
```
Audio Generator:
  num_inference_steps: 50-100
  guidance_scale: 10-15
```

### Memory Optimization

If you get CUDA out of memory errors:

1. **Reduce video resolution** before Visual Feature Extractor
2. **Process shorter clips** (5-10 seconds)
3. **Lower inference steps** to 20-30
4. **Enable memory clearing** in config

### Adding Text Guidance

Add text prompts to Audio Generator for more control:

```
prompt: "Footsteps on gravel, birds chirping"
negative_prompt: "music, speech, silence"
```

## Custom Workflows

You can combine these examples to create custom workflows:

### Example: Multi-stage refinement
```
Video → Visual Features → Audio Generator → Audio Refiner 1 → Audio Refiner 2 → Save
```

### Example: Blended with temporal sync
```
Video 1 → Visual Features → Audio Gen → \
                                        → Audio Blender → Temporal Sync → Save
Video 2 → Visual Features → Audio Gen → /
```

### Example: Latent + Refinement
```
Video → VAE → Latent to Visual Features → Audio Gen → Audio Refiner → Save
```

## Troubleshooting

### No audio output
- Check that video has at least 1 second duration
- Increase `guidance_scale` to 10-15
- Add a text prompt describing expected audio

### Audio doesn't match video
- Use the **Temporal Synchronizer** node
- Ensure video framerate is consistent
- Check that video duration matches intended audio duration

### Memory errors
- Reduce video resolution (512x512 or smaller)
- Process shorter clips
- Lower `num_inference_steps`

### Poor audio quality
- Increase `num_inference_steps` to 50-100
- Use **Audio Refiner** with `normalize=True`
- Add descriptive text prompts
- Increase `guidance_scale`

## Tips for Best Results

1. **Video Quality**: Higher quality video → better visual features → better audio
2. **Text Prompts**: Descriptive prompts help guide generation
3. **Temporal Sync**: Always use for videos with distinct timing (footsteps, impacts, etc.)
4. **Audio Refinement**: Enable normalization for consistent volume
5. **Experimentation**: Try different `guidance_scale` values for your content

## Next Steps

After trying these workflows:
- Read the [User Guide](../docs/USER_GUIDE.md) for detailed parameter explanations
- Check [Troubleshooting](../docs/TROUBLESHOOTING.md) if you encounter issues
- Explore [Fine-tuning](../docs/FINETUNING_GUIDE.md) for custom sounds
