# VersaSound: Universal Video-to-Audio Generation for ComfyUI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

**VersaSound** is a complete, production-ready ComfyUI extension for universal video-to-audio generation with comprehensive training and fine-tuning capabilities. Generate synchronized, high-quality audio for ANY video content using state-of-the-art AI models.

## ✨ Key Features

- **🎬 Universal Video Support**: Works with any video content - no hardcoded scenarios
- **🔄 ComfyUI Integration**: Seamless workflow integration with custom nodes
- **🎯 Temporal Synchronization**: Automatic audio-visual alignment (lip-sync, impacts, rhythm)
- **🎛️ Fine-tuning Ready**: Easy customization for specialized sound effects
- **💪 Production-Ready**: Robust error handling, memory optimization, comprehensive logging
- **🚀 Efficient**: Optimized for consumer hardware (8GB+ VRAM)
- **📚 Complete Training System**: Full training pipeline included

## 🎥 What Can It Do?

VersaSound can generate synchronized audio for:
- 🗣️ **Speech and Dialogue**: Natural speech generation with lip-sync
- 🎬 **Action Scenes**: Impacts, explosions, footsteps, motion sounds
- 🌿 **Nature**: Wind, water, animals, ambient sounds
- 🎵 **Music**: Background music, rhythm-synced audio
- 🏭 **Foley Effects**: Custom sound effects for any object or action
- 🎮 **And More**: Any video content you can imagine!

## 📋 Table of Contents

- [Installation](#installation)
- [Troubleshooting](#troubleshooting)
- [Quick Start](#quick-start)
- [ComfyUI Nodes](#comfyui-nodes)
- [Training](#training)
- [Fine-tuning](#fine-tuning)
- [Documentation](#documentation)
- [Examples](#examples)
- [Requirements](#requirements)
- [License](#license)

## 🚀 Installation

### For ComfyUI Users

1. **Clone the repository** into your ComfyUI `custom_nodes` directory:
```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/yourusername/VersaSound.git
```

2. **Run the installation script**:
```bash
cd VersaSound/comfyui_extension
python install.py
```

This will:
- Install all required dependencies
- Download pretrained models (~2GB)
- Verify the installation

3. **Restart ComfyUI**

4. **Look for VersaSound nodes** in the "VersaSound" category

### Manual Installation

If you prefer to install manually:

```bash
cd VersaSound/comfyui_extension
pip install -r requirements.txt

# Download models manually or they'll download on first use
```

### System Requirements

**Minimum:**
- Python 3.8+
- PyTorch 2.0+
- 8GB RAM
- 4GB VRAM (GPU) or CPU (slower)

**Recommended:**
- Python 3.10+
- PyTorch 2.0+ with CUDA
- 16GB RAM
- 8GB+ VRAM (NVIDIA GPU)

## 🔧 Troubleshooting

### AudioLDM2 Compatibility Error

**Error**: `'GPT2Model' object has no attribute '_update_model_kwargs_for_generation'`

**Cause**: Incompatible versions of `transformers` and `diffusers` libraries.

**Solution**: Install compatible versions:
```bash
pip install 'transformers>=4.30.0,<5.0.0' 'diffusers>=0.25.0,<0.32.0'
```

**Note**: Transformers 5.x has breaking changes in the generation API that are incompatible with AudioLDM2. Always use transformers 4.x.

### No Audio Output / Silent Audio

If you're getting silent audio or the generation completes too quickly:

1. Check the console logs for "Using dummy audio generator" warning
2. This means AudioLDM2 failed to load - see error above
3. Install correct library versions
4. Restart ComfyUI after fixing

### Memory Issues

If you encounter CUDA out of memory errors:

1. Reduce video resolution before processing
2. Process shorter video clips
3. Enable `clear_cache_after_generation` in config
4. Use CPU mode (slower but uses system RAM)

## 🎯 Quick Start

### Basic Video-to-Audio Generation

1. **Load your video** in ComfyUI (or use video latents from generation)

2. **Add VersaSound nodes** to your workflow:
   - `Visual Feature Extractor` - Extract features from video
   - `Audio Generator` - Generate audio from features
   - `Audio Refiner` - (Optional) Enhance audio quality

3. **Connect the nodes**:
```
Video → Visual Feature Extractor → Audio Generator → Audio Output
```

4. **Configure parameters**:
   - Choose audio model (AudioLDM2, Stable Audio, etc.)
   - Add text prompts for guidance (optional)
   - Adjust quality settings

5. **Generate!** 🎵

### Advanced Workflow with Synchronization

```
Video → Visual Feature Extractor → Audio Generator
                ↓                          ↓
                → Temporal Synchronizer ←
                         ↓
                   Audio Refiner
                         ↓
                    Audio Output
```

## 🎛️ ComfyUI Nodes

VersaSound provides the following nodes:

### Visual Processing
- **Visual Feature Extractor**: Extract comprehensive visual features from video
- **Latent to Visual Features**: Work directly with ComfyUI latents for efficiency

### Audio Generation
- **Audio Generator**: Core audio generation from visual features
  - Supports multiple backend models (AudioLDM2, Stable Audio, AudioCraft)
  - Text-guided generation
  - Adjustable quality and duration

### Synchronization
- **Temporal Synchronizer**: Ensure audio-visual alignment
  - Automatic mode for general content
  - Specialized modes: lip-sync, impact-sync, rhythm-sync
  - DTW-based alignment with minimal distortion

### Post-Processing
- **Audio Refiner**: Enhance generated audio
  - Noise reduction
  - Normalization (peak, RMS, LUFS)
  - EQ presets (voice, music, SFX, cinematic)
  - Reverb and effects

### Utilities
- **Audio Blender**: Mix multiple audio tracks
- **Scene Analyzer**: Analyze video content (coming soon)
- **Batch Processor**: Process multiple videos (coming soon)

## 🏋️ Training

VersaSound includes a complete training system for training from scratch or fine-tuning on custom data.

### Training from Scratch

The training process has 3 stages:

**Stage 1: Visual Encoder**
```bash
cd training_system
python train.py \
    --stage visual \
    --config configs/training_config.yaml \
    --data-config configs/dataset_config.yaml \
    --batch-size 16 \
    --epochs 20
```

**Stage 2: Audio Generator**
```bash
python train.py \
    --stage audio \
    --config configs/training_config.yaml \
    --data-config configs/dataset_config.yaml \
    --batch-size 8 \
    --epochs 30
```

**Stage 3: End-to-End Fine-tuning**
```bash
python train.py \
    --stage end2end \
    --config configs/training_config.yaml \
    --data-config configs/dataset_config.yaml \
    --batch-size 4 \
    --epochs 10
```

### Supported Datasets

- **VGGSound**: 200k+ video clips with audio
- **AudioSet**: Large-scale audio event dataset
- **Greatest Hits**: Material sounds dataset
- **Custom**: Your own video-audio pairs

See `training_system/configs/dataset_config.yaml` for configuration.

## 🎨 Fine-tuning

### Quick Fine-tuning with LoRA

Fine-tune VersaSound for specialized sound effects:

```bash
cd fine_tuning

# Prepare your custom dataset
python prepare_custom_data.py \
    --data-dir /path/to/your/videos \
    --output-dir ./prepared_data

# Fine-tune with LoRA
python finetune.py \
    --model-path ../comfyui_extension/checkpoints/audio_generator.safetensors \
    --data-dir ./prepared_data \
    --finetune-mode lora \
    --lora-rank 16 \
    --epochs 10 \
    --output-dir ./finetuned_models
```

### Custom Dataset Format

For fine-tuning, organize your data as:

```
your_dataset/
├── video1.mp4
├── video1.wav
├── video2.mp4
├── video2.wav
└── ...
```

Or use an index file (`index.json`):
```json
[
  {
    "video_path": "path/to/video1.mp4",
    "audio_path": "path/to/audio1.wav",
    "duration": 5.0,
    "labels": ["custom_category"]
  },
  ...
]
```

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Installation Guide](docs/INSTALLATION.md)**: Detailed installation instructions
- **[User Guide](docs/USER_GUIDE.md)**: Complete guide to using VersaSound
- **[Training Guide](docs/TRAINING_GUIDE.md)**: Training from scratch
- **[Fine-tuning Guide](docs/FINETUNING_GUIDE.md)**: Fine-tuning for custom sounds
- **[API Reference](docs/API_REFERENCE.md)**: Python API documentation

## 🎬 Examples

Example workflows are provided in the `examples/` directory:

- `basic_video_to_audio.json`: Simple video-to-audio generation
- `batch_processing.json`: Process multiple videos at once
- `custom_conditioning.json`: Advanced control with text prompts
- `latent_workflow.json`: Working with video latents

Load these in ComfyUI to see VersaSound in action!

## 🏗️ Project Structure

```
VersaSound/
├── comfyui_extension/       # ComfyUI plugin (installable)
│   ├── nodes.py             # All ComfyUI nodes
│   ├── models/              # Model implementations
│   ├── utils/               # Utilities
│   ├── configs/             # Configurations
│   └── checkpoints/         # Pretrained weights
│
├── training_system/         # Complete training pipeline
│   ├── train.py             # Main training script
│   ├── datasets/            # Dataset handlers
│   ├── models/              # Training models
│   └── configs/             # Training configs
│
├── fine_tuning/             # Fine-tuning tools
│   ├── finetune.py          # Fine-tuning script
│   └── prepare_custom_data.py
│
├── docs/                    # Documentation
├── examples/                # Example workflows
└── tests/                   # Unit tests
```

## 🔧 Advanced Configuration

### Model Configuration

Edit `comfyui_extension/configs/default_config.yaml` to customize:

- Visual encoder backbone (VideoMAE, CLIP, DINOv2)
- Audio generation model (AudioLDM2, Stable Audio, AudioCraft)
- Memory optimization settings
- Processing parameters

### Performance Tuning

For systems with limited VRAM:

```yaml
memory:
  gradient_checkpointing: true
  max_batch_size: 4
  clear_cache_after_generation: true
```

For high-performance systems:

```yaml
memory:
  max_batch_size: 16
device:
  precision: "fp16"  # or "bf16"
```

## 🧪 Testing

Run the test suite:

```bash
cd tests
pytest
```

Individual test modules:
```bash
pytest test_nodes.py      # Test ComfyUI nodes
pytest test_models.py     # Test model architectures
pytest test_training.py   # Test training pipeline
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

VersaSound builds upon excellent open-source projects:

- **AudioLDM2** - Audio generation model
- **VideoMAE** - Visual feature extraction
- **ComfyUI** - Workflow interface
- **Hugging Face** - Model hosting and tools

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/VersaSound/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/VersaSound/discussions)
- **Documentation**: [docs/](docs/)

## 🗺️ Roadmap

- [x] Core video-to-audio generation
- [x] ComfyUI integration
- [x] Temporal synchronization
- [x] Training system
- [x] Fine-tuning support
- [ ] Real-time generation mode
- [ ] Additional audio model backends
- [ ] Scene understanding improvements
- [ ] Multi-modal conditioning (image + text + audio)
- [ ] Web UI (standalone)

## 📊 Performance

Benchmarks on NVIDIA RTX 3090:

| Task | Time | VRAM |
|------|------|------|
| Visual Feature Extraction (1s video) | ~0.5s | ~2GB |
| Audio Generation (1s, 50 steps) | ~8s | ~4GB |
| Full Pipeline (5s video) | ~45s | ~6GB |

*Performance varies based on hardware, settings, and content*

## ⭐ Star History

If you find VersaSound useful, please star the repository!

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/VersaSound&type=Date)](https://star-history.com/#yourusername/VersaSound&Date)

---

**Made with ❤️ for the ComfyUI community**
