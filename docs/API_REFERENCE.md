# VersaSound API Reference

Technical reference for VersaSound models, utilities, and APIs.

## Table of Contents

- [Data Types](#data-types)
- [Models](#models)
- [Utilities](#utilities)
- [ComfyUI Nodes](#comfyui-nodes)
- [Configuration](#configuration)

## Data Types

### VisualFeatures

**Location**: `comfyui_extension/custom_types.py`

```python
from comfyui_extension.custom_types import VisualFeatures

class VisualFeatures:
    """Container for visual features extracted from video"""

    def __init__(
        self,
        features: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            features: Visual features tensor [B, T, D]
                - B: Batch size (usually 1)
                - T: Temporal dimension (number of frames)
                - D: Feature dimension (768, 1024, etc.)
            metadata: Optional metadata dict
        """

    # Properties
    @property
    def shape(self) -> torch.Size:
        """Returns features shape"""

    @property
    def device(self) -> torch.device:
        """Returns device (cpu, cuda, mps)"""

    @property
    def dtype(self) -> torch.dtype:
        """Returns data type (float32, float16, etc.)"""

    # Methods
    def to(self, device: Union[str, torch.device]) -> 'VisualFeatures':
        """Move features to device"""

    def get_temporal_length(self) -> int:
        """Returns number of temporal frames"""

    def get_feature_dim(self) -> int:
        """Returns feature dimension"""
```

**Usage Example**:
```python
# Create VisualFeatures
import torch
from comfyui_extension.custom_types import VisualFeatures

features_tensor = torch.randn(1, 24, 768)  # 24 frames, 768 dims
visual_features = VisualFeatures(
    features=features_tensor,
    metadata={'backbone': 'videomae', 'fps': 24}
)

# Access properties
print(visual_features.shape)  # torch.Size([1, 24, 768])
print(visual_features.get_temporal_length())  # 24
print(visual_features.get_feature_dim())  # 768

# Move to GPU
visual_features = visual_features.to('cuda')
```

### AudioData

**Location**: `comfyui_extension/custom_types.py`

```python
from comfyui_extension.custom_types import AudioData

class AudioData:
    """Container for audio data"""

    def __init__(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            waveform: Audio waveform [B, C, T]
                - B: Batch size
                - C: Channels (1=mono, 2=stereo)
                - T: Time samples
            sample_rate: Sample rate in Hz
            metadata: Optional metadata dict
        """

    # Properties
    @property
    def shape(self) -> torch.Size:
        """Returns waveform shape"""

    @property
    def duration(self) -> float:
        """Returns duration in seconds"""

    @property
    def num_channels(self) -> int:
        """Returns number of channels"""

    # Methods
    def to(self, device: Union[str, torch.device]) -> 'AudioData':
        """Move audio to device"""

    def to_mono(self) -> 'AudioData':
        """Convert to mono (average channels)"""

    def to_stereo(self) -> 'AudioData':
        """Convert to stereo (duplicate if mono)"""

    def resample(self, new_sample_rate: int) -> 'AudioData':
        """Resample to new sample rate"""
```

**Usage Example**:
```python
import torch
from comfyui_extension.custom_types import AudioData

# Create AudioData
waveform = torch.randn(1, 1, 80000)  # 1 batch, 1 channel, 80000 samples
audio = AudioData(
    waveform=waveform,
    sample_rate=16000,
    metadata={'source': 'generated'}
)

# Access properties
print(audio.duration)  # 5.0 seconds (80000 / 16000)
print(audio.num_channels)  # 1 (mono)

# Convert to stereo
audio_stereo = audio.to_stereo()
print(audio_stereo.shape)  # torch.Size([1, 2, 80000])

# Resample to 44.1kHz
audio_hq = audio.resample(44100)
```

## Models

### UniversalVisualEncoder

**Location**: `comfyui_extension/models/visual_encoder.py`

```python
from comfyui_extension.models.visual_encoder import UniversalVisualEncoder

class UniversalVisualEncoder(nn.Module):
    """
    Universal visual feature extractor
    Supports multiple pretrained backbones
    """

    def __init__(
        self,
        backbone: str = "videomae",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        feature_dim: int = 768,
        temporal_pooling: str = "attention",
        spatial_pooling: str = "avg",
        device: str = "auto"
    ):
        """
        Args:
            backbone: Vision model ("videomae", "clip", "dinov2")
            pretrained: Load pretrained weights
            freeze_backbone: Freeze backbone weights
            feature_dim: Output feature dimension
            temporal_pooling: "attention", "avg", "max", "none"
            spatial_pooling: "avg", "max", "cls_token"
            device: Device to use
        """

    def forward(
        self,
        video: torch.Tensor,
        return_all_tokens: bool = False
    ) -> Union[torch.Tensor, VisualFeatures]:
        """
        Args:
            video: Video tensor [B, T, C, H, W]
            return_all_tokens: Return all spatial tokens

        Returns:
            Visual features [B, T, D] or VisualFeatures object
        """
```

**Usage Example**:
```python
import torch
from comfyui_extension.models.visual_encoder import UniversalVisualEncoder

# Create model
encoder = UniversalVisualEncoder(
    backbone="videomae",
    pretrained=True,
    freeze_backbone=True,
    feature_dim=768
)

# Process video
video = torch.randn(1, 24, 3, 224, 224)  # 24 frames, RGB, 224x224
features = encoder(video)

print(features.shape)  # torch.Size([1, 24, 768])
```

**Supported Backbones**:

| Backbone | Feature Dim | Best For |
|----------|-------------|----------|
| videomae | 768 | General video (default) |
| clip | 512-1024 | Text-aligned features |
| dinov2 | 768-1024 | Self-supervised features |

### UniversalAudioGenerator

**Location**: `comfyui_extension/models/audio_generator.py`

```python
from comfyui_extension.models.audio_generator import UniversalAudioGenerator

class UniversalAudioGenerator(nn.Module):
    """
    Universal audio generator from visual features
    Supports multiple audio generation models
    """

    def __init__(
        self,
        model_type: str = "audioldm2",
        pretrained: bool = True,
        visual_feature_dim: int = 768,
        audio_feature_dim: int = 1024,
        device: str = "auto"
    ):
        """
        Args:
            model_type: "audioldm2", "stable_audio", "audiocraft"
            pretrained: Load pretrained weights
            visual_feature_dim: Input feature dimension
            audio_feature_dim: Internal audio feature dimension
            device: Device to use
        """

    def forward(
        self,
        visual_features: Union[torch.Tensor, VisualFeatures],
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 10.0,
        sample_rate: int = 16000,
        duration: float = 5.0,
        seed: Optional[int] = None
    ) -> AudioData:
        """
        Args:
            visual_features: Visual features [B, T, D]
            prompt: Text guidance
            negative_prompt: Negative text guidance
            num_inference_steps: Diffusion steps
            guidance_scale: CFG scale
            sample_rate: Output sample rate
            duration: Output duration (seconds)
            seed: Random seed

        Returns:
            AudioData object with generated audio
        """
```

**Usage Example**:
```python
from comfyui_extension.models.audio_generator import UniversalAudioGenerator

# Create generator
generator = UniversalAudioGenerator(
    model_type="audioldm2",
    pretrained=True
)

# Generate audio
audio = generator(
    visual_features=features,
    prompt="Footsteps on wooden floor",
    negative_prompt="music, speech",
    num_inference_steps=50,
    guidance_scale=10.0,
    duration=5.0,
    seed=42
)

print(audio.duration)  # 5.0
print(audio.sample_rate)  # 16000
```

**Supported Models**:

| Model | Speed | Quality | Best For |
|-------|-------|---------|----------|
| audioldm2 | Medium | High | General purpose (default) |
| stable_audio | Fast | Good | Real-time applications |
| audiocraft | Slow | Very High | Creative/artistic |

### TemporalAlignmentModule

**Location**: `comfyui_extension/models/temporal_aligner.py`

```python
from comfyui_extension.models.temporal_aligner import TemporalAlignmentModule

class TemporalAlignmentModule(nn.Module):
    """
    Aligns audio timing with visual events
    """

    def __init__(
        self,
        visual_dim: int = 768,
        audio_dim: int = 1024,
        hidden_dim: int = 512,
        num_layers: int = 4,
        device: str = "auto"
    ):
        """
        Args:
            visual_dim: Visual feature dimension
            audio_dim: Audio feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of transformer layers
            device: Device to use
        """

    def forward(
        self,
        audio: Union[torch.Tensor, AudioData],
        visual_features: Union[torch.Tensor, VisualFeatures],
        sync_method: str = "learned",
        sync_strength: float = 0.5,
        adaptive: bool = True
    ) -> AudioData:
        """
        Args:
            audio: Audio to align
            visual_features: Visual features for alignment
            sync_method: "learned", "dtw", "cross_correlation"
            sync_strength: Alignment strength (0.0-1.0)
            adaptive: Adaptive synchronization

        Returns:
            Temporally aligned AudioData
        """
```

**Usage Example**:
```python
from comfyui_extension.models.temporal_aligner import TemporalAlignmentModule

# Create aligner
aligner = TemporalAlignmentModule(
    visual_dim=768,
    audio_dim=1024
)

# Align audio
aligned_audio = aligner(
    audio=generated_audio,
    visual_features=visual_features,
    sync_method="learned",
    sync_strength=0.5,
    adaptive=True
)
```

**Sync Methods**:

| Method | Accuracy | Speed | Best For |
|--------|----------|-------|----------|
| learned | High | Medium | General purpose (default) |
| dtw | Very High | Slow | Precise alignment needed |
| cross_correlation | Medium | Fast | Real-time applications |

## Utilities

### DeviceManager

**Location**: `comfyui_extension/utils/device_management.py`

```python
from comfyui_extension.utils.device_management import DeviceManager

# Singleton instance
device_manager = DeviceManager()

# Get device
device = device_manager.get_device()  # Returns torch.device

# Check capabilities
has_cuda = device_manager.has_cuda()  # bool
has_mps = device_manager.has_mps()    # bool

# Get device info
info = device_manager.get_device_info()
# {'device': 'cuda', 'name': 'NVIDIA GeForce RTX 3090', 'memory': 24576}

# Memory management
device_manager.clear_cache()  # Clear GPU cache
device_manager.optimize_memory()  # Enable memory optimizations
```

### VideoProcessor

**Location**: `comfyui_extension/utils/video_processing.py`

```python
from comfyui_extension.utils.video_processing import VideoProcessor

processor = VideoProcessor()

# Normalize video (ComfyUI format → model input)
normalized = processor.normalize_video(
    video,  # [N, H, W, C] in [0, 1]
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)  # Returns [B, T, C, H, W] normalized

# Denormalize (model output → ComfyUI format)
denormalized = processor.denormalize_video(normalized)

# Resize video
resized = processor.resize_video(
    video,
    target_size=(224, 224),
    interpolation="bilinear"
)

# Extract frames
frames = processor.extract_frames(
    video_path="path/to/video.mp4",
    fps=24,
    max_frames=100
)
```

### AudioProcessor

**Location**: `comfyui_extension/utils/audio_processing.py`

```python
from comfyui_extension.utils.audio_processing import AudioProcessor

processor = AudioProcessor()

# Load audio
audio = processor.load_audio(
    "path/to/audio.wav",
    sample_rate=16000,
    mono=True
)

# Save audio
processor.save_audio(
    audio_data,
    "output.wav",
    sample_rate=16000
)

# Resample
resampled = processor.resample(
    audio,
    original_rate=16000,
    target_rate=44100
)

# Normalize
normalized = processor.normalize(
    audio,
    target_level=-20  # dB
)

# Apply effects
denoised = processor.denoise(audio, strength=0.5)
enhanced = processor.enhance_frequencies(audio, bass_boost=1.2)
compressed = processor.compress(audio, ratio=0.7)
```

### LatentProcessor

**Location**: `comfyui_extension/utils/latent_utils.py`

```python
from comfyui_extension.utils.latent_utils import LatentProcessor

processor = LatentProcessor()

# Decode latents to images
images = processor.decode_latents(
    latents,  # ComfyUI LATENT dict
    vae       # ComfyUI VAE
)  # Returns [N, H, W, C] images

# Encode images to latents
latents = processor.encode_to_latents(
    images,   # [N, H, W, C]
    vae       # ComfyUI VAE
)  # Returns ComfyUI LATENT dict

# Get latent info
info = processor.get_latent_info(latents)
# {'shape': [1, 4, 32, 32], 'device': 'cuda', ...}
```

## ComfyUI Nodes

### Node Class Structure

All VersaSound nodes follow this structure:

```python
class VersaSoundNode:
    @classmethod
    def INPUT_TYPES(cls):
        """Define input parameters"""
        return {
            "required": {
                "param_name": ("TYPE", {"default": value}),
            },
            "optional": {
                "optional_param": ("TYPE",),
            }
        }

    RETURN_TYPES = ("TYPE1", "TYPE2")
    RETURN_NAMES = ("output1", "output2")
    FUNCTION = "process"
    CATEGORY = "VersaSound"

    def process(self, **kwargs):
        """Main processing function"""
        # Implementation
        return (output1, output2)
```

### Custom Types

VersaSound registers these custom types with ComfyUI:

```python
# In __init__.py
NODE_CLASS_MAPPINGS = {
    "VersaSound_VisualFeatureExtractor": VisualFeatureExtractor,
    "VersaSound_AudioGenerator": AudioGenerator,
    # ... etc
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VersaSound_VisualFeatureExtractor": "Visual Feature Extractor",
    "VersaSound_AudioGenerator": "Audio Generator",
    # ... etc
}
```

### Accessing Nodes Programmatically

```python
# Import nodes
from comfyui_extension.nodes import (
    VisualFeatureExtractor,
    AudioGenerator,
    TemporalSynchronizer,
    AudioRefiner,
    LatentToVisualFeatures,
    AudioBlender
)

# Use node directly (outside ComfyUI)
extractor = VisualFeatureExtractor()
visual_features = extractor.extract(
    video=video_tensor,
    backbone="videomae",
    input_size=224
)
```

## Configuration

### Config File Structure

**Location**: `comfyui_extension/configs/default_config.yaml`

```yaml
# Visual Encoder Configuration
visual_encoder:
  backbone: "videomae"
  input_size: 224
  feature_dim: 768
  temporal_pooling: "attention"
  spatial_pooling: "avg"
  freeze_backbone: true

# Audio Generator Configuration
audio_generator:
  default_model: "audioldm2"
  num_inference_steps: 50
  guidance_scale: 10.0
  sample_rate: 16000
  default_duration: 5.0

# Temporal Aligner Configuration
temporal_aligner:
  hidden_dim: 512
  num_layers: 4
  sync_method: "learned"
  sync_strength: 0.5

# Audio Refiner Configuration
audio_refiner:
  normalize: true
  denoise_strength: 0.5
  bass_boost: 1.0
  treble_boost: 1.0
  compression_ratio: 0.7

# Memory Configuration
memory:
  max_batch_size: 8
  clear_cache_after_generation: true
  use_gradient_checkpointing: false

# Device Configuration
device:
  device_type: "auto"  # auto, cuda, mps, cpu
  precision: "fp16"     # fp16, fp32, bf16
  use_autocast: true

# Logging Configuration
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  log_to_file: false
  log_file: "versasound.log"
```

### Loading Config Programmatically

```python
from comfyui_extension.utils.config_utils import load_config

# Load default config
config = load_config()

# Load custom config
config = load_config("path/to/custom_config.yaml")

# Access config values
backbone = config['visual_encoder']['backbone']
num_steps = config['audio_generator']['num_inference_steps']

# Override config
config['device']['precision'] = "fp32"
```

### Config Classes

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class VisualEncoderConfig:
    backbone: str = "videomae"
    input_size: int = 224
    feature_dim: int = 768
    temporal_pooling: str = "attention"
    spatial_pooling: str = "avg"
    freeze_backbone: bool = True

@dataclass
class AudioGeneratorConfig:
    default_model: str = "audioldm2"
    num_inference_steps: int = 50
    guidance_scale: float = 10.0
    sample_rate: int = 16000
    default_duration: float = 5.0

@dataclass
class DeviceConfig:
    device_type: str = "auto"
    precision: str = "fp16"
    use_autocast: bool = True
```

## Error Handling

### Custom Exceptions

```python
from comfyui_extension.utils.exceptions import (
    VersaSoundError,
    ModelLoadError,
    InvalidInputError,
    DeviceError
)

try:
    # VersaSound operations
    pass
except ModelLoadError as e:
    print(f"Failed to load model: {e}")
except InvalidInputError as e:
    print(f"Invalid input: {e}")
except VersaSoundError as e:
    print(f"VersaSound error: {e}")
```

### Logging

```python
from comfyui_extension.utils.logging_utils import get_logger

logger = get_logger(__name__)

logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
```

## Examples

### Complete Pipeline

```python
import torch
from comfyui_extension.models.visual_encoder import UniversalVisualEncoder
from comfyui_extension.models.audio_generator import UniversalAudioGenerator
from comfyui_extension.models.temporal_aligner import TemporalAlignmentModule
from comfyui_extension.utils.audio_processing import AudioProcessor

# 1. Load video
video = torch.randn(1, 24, 3, 224, 224)  # 24 frames

# 2. Extract visual features
encoder = UniversalVisualEncoder(backbone="videomae")
visual_features = encoder(video)

# 3. Generate audio
generator = UniversalAudioGenerator(model_type="audioldm2")
audio = generator(
    visual_features=visual_features,
    prompt="Footsteps on gravel",
    num_inference_steps=50,
    guidance_scale=10.0
)

# 4. Temporal alignment
aligner = TemporalAlignmentModule()
aligned_audio = aligner(
    audio=audio,
    visual_features=visual_features,
    sync_method="learned"
)

# 5. Save audio
processor = AudioProcessor()
processor.save_audio(aligned_audio, "output.wav")
```

### Custom Training Loop

```python
from comfyui_extension.training.trainer import VersaSoundTrainer
from comfyui_extension.training.datasets import VideoAudioDataset

# Create dataset
dataset = VideoAudioDataset(
    video_dir="data/videos",
    audio_dir="data/audio",
    transform=None
)

# Create trainer
trainer = VersaSoundTrainer(
    visual_encoder_config=...,
    audio_generator_config=...,
    device="cuda"
)

# Train
trainer.train(
    dataset=dataset,
    num_epochs=100,
    batch_size=4,
    learning_rate=1e-4
)
```

## Migration Guide

### From Older Versions

**v1.0.0 → v1.1.0**:
- `VisualFeatureExtractor` now auto-resizes inputs
- Added ImageNet normalization (automatic)
- Config file structure changed (backwards compatible)

## Performance Tips

### Optimization Checklist

1. **Enable mixed precision**:
   ```python
   config['device']['precision'] = "fp16"
   ```

2. **Clear cache after generation**:
   ```python
   config['memory']['clear_cache_after_generation'] = True
   ```

3. **Use gradient checkpointing** (training):
   ```python
   config['memory']['use_gradient_checkpointing'] = True
   ```

4. **Batch processing**:
   ```python
   # Process multiple videos in batch
   videos = torch.stack([video1, video2, video3])
   features = encoder(videos)
   ```

5. **Freeze backbone** (training):
   ```python
   encoder = UniversalVisualEncoder(freeze_backbone=True)
   ```

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for development guidelines.

## License

MIT License - see [LICENSE](../LICENSE)
