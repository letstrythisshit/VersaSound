"""
Custom data types for VersaSound ComfyUI extension
Defines structured containers for passing data between nodes
"""

import torch
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


class VisualFeatures:
    """
    Container for visual features extracted from video

    Attributes:
        backbone: Core visual features from backbone network [B, T, D]
        motion: Motion features (optical flow, velocity) [B, T, M]
        semantic: Semantic/scene understanding features [B, T, S]
        temporal: Temporally-modeled features [B, T, D]
        audio_cues: Audio-relevant cues (intensity, contacts, etc.) Dict[str, Tensor]
        metadata: Additional metadata (fps, duration, resolution, etc.)
    """

    def __init__(
        self,
        backbone_features: torch.Tensor,
        motion_features: Optional[torch.Tensor] = None,
        semantic_features: Optional[torch.Tensor] = None,
        temporal_features: Optional[torch.Tensor] = None,
        audio_cues: Optional[Dict[str, torch.Tensor]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.backbone = backbone_features
        self.motion = motion_features
        self.semantic = semantic_features
        self.temporal = temporal_features
        self.audio_cues = audio_cues or {}
        self.metadata = metadata or {}

        # Validate metadata
        self._ensure_metadata()

    def _ensure_metadata(self):
        """Ensure required metadata fields exist"""
        defaults = {
            'fps': 24.0,
            'duration': self.backbone.shape[1] / 24.0 if self.backbone is not None else 0.0,
            'num_frames': self.backbone.shape[1] if self.backbone is not None else 0,
            'device': str(self.backbone.device) if self.backbone is not None else 'cpu',
            'resolution': (224, 224),
        }

        for key, default_value in defaults.items():
            if key not in self.metadata:
                self.metadata[key] = default_value

    def to(self, device: torch.device) -> 'VisualFeatures':
        """Move all tensors to specified device"""
        self.backbone = self.backbone.to(device) if self.backbone is not None else None
        self.motion = self.motion.to(device) if self.motion is not None else None
        self.semantic = self.semantic.to(device) if self.semantic is not None else None
        self.temporal = self.temporal.to(device) if self.temporal is not None else None

        self.audio_cues = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in self.audio_cues.items()
        }

        self.metadata['device'] = str(device)
        return self

    def cpu(self) -> 'VisualFeatures':
        """Move all tensors to CPU"""
        return self.to(torch.device('cpu'))

    def cuda(self) -> 'VisualFeatures':
        """Move all tensors to CUDA"""
        return self.to(torch.device('cuda'))

    def clone(self) -> 'VisualFeatures':
        """Create a deep copy of the features"""
        return VisualFeatures(
            backbone_features=self.backbone.clone() if self.backbone is not None else None,
            motion_features=self.motion.clone() if self.motion is not None else None,
            semantic_features=self.semantic.clone() if self.semantic is not None else None,
            temporal_features=self.temporal.clone() if self.temporal is not None else None,
            audio_cues={k: v.clone() if isinstance(v, torch.Tensor) else v
                       for k, v in self.audio_cues.items()},
            metadata=self.metadata.copy()
        )

    def __repr__(self) -> str:
        return (f"VisualFeatures(backbone={self.backbone.shape if self.backbone is not None else None}, "
                f"motion={self.motion.shape if self.motion is not None else None}, "
                f"temporal={self.temporal.shape if self.temporal is not None else None}, "
                f"fps={self.metadata.get('fps')}, duration={self.metadata.get('duration')}s)")


class AudioData:
    """
    Container for audio waveform + metadata

    Attributes:
        waveform: Audio waveform tensor [channels, samples] or [samples]
        sample_rate: Sample rate in Hz
        metadata: Additional metadata (duration, format, etc.)
    """

    def __init__(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 48000,
        metadata: Optional[Dict[str, Any]] = None
    ):
        # Ensure waveform is 2D [channels, samples]
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim > 2:
            raise ValueError(f"Waveform must be 1D or 2D, got shape {waveform.shape}")

        self.waveform = waveform
        self.sample_rate = sample_rate
        self.metadata = metadata or {}

        # Auto-populate metadata
        self.metadata['duration'] = self.duration
        self.metadata['num_channels'] = self.num_channels
        self.metadata['num_samples'] = self.waveform.shape[-1]

    @property
    def duration(self) -> float:
        """Get audio duration in seconds"""
        return self.waveform.shape[-1] / self.sample_rate

    @property
    def num_channels(self) -> int:
        """Get number of audio channels"""
        return self.waveform.shape[0]

    @property
    def num_samples(self) -> int:
        """Get number of audio samples"""
        return self.waveform.shape[-1]

    def to(self, device: torch.device) -> 'AudioData':
        """Move waveform to specified device"""
        self.waveform = self.waveform.to(device)
        return self

    def cpu(self) -> 'AudioData':
        """Move waveform to CPU"""
        return self.to(torch.device('cpu'))

    def cuda(self) -> 'AudioData':
        """Move waveform to CUDA"""
        return self.to(torch.device('cuda'))

    def to_mono(self) -> 'AudioData':
        """Convert to mono by averaging channels"""
        if self.num_channels == 1:
            return self

        mono_waveform = self.waveform.mean(dim=0, keepdim=True)
        return AudioData(
            waveform=mono_waveform,
            sample_rate=self.sample_rate,
            metadata=self.metadata.copy()
        )

    def resample(self, target_sr: int) -> 'AudioData':
        """Resample audio to target sample rate"""
        if target_sr == self.sample_rate:
            return self

        try:
            import torchaudio
            resampler = torchaudio.transforms.Resample(
                orig_freq=self.sample_rate,
                new_freq=target_sr
            ).to(self.waveform.device)

            resampled = resampler(self.waveform)

            return AudioData(
                waveform=resampled,
                sample_rate=target_sr,
                metadata=self.metadata.copy()
            )
        except ImportError:
            raise ImportError("torchaudio is required for resampling")

    def clone(self) -> 'AudioData':
        """Create a deep copy"""
        return AudioData(
            waveform=self.waveform.clone(),
            sample_rate=self.sample_rate,
            metadata=self.metadata.copy()
        )

    def __repr__(self) -> str:
        return (f"AudioData(channels={self.num_channels}, samples={self.num_samples}, "
                f"sr={self.sample_rate}Hz, duration={self.duration:.2f}s)")


@dataclass
class SceneAnalysis:
    """
    Results from scene analysis

    Attributes:
        scene_type: Primary scene type (e.g., "speech", "action", "nature")
        confidence: Confidence score [0, 1]
        attributes: Dict of scene attributes and their scores
        event_timeline: Optional list of detected events with timestamps
    """
    scene_type: str
    confidence: float
    attributes: Dict[str, float]
    event_timeline: Optional[list] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SyncReport:
    """
    Report from temporal synchronization

    Attributes:
        confidence: Overall synchronization confidence [0, 1]
        alignment_error: Average alignment error in seconds
        num_events: Number of events synchronized
        warp_applied: Whether time-stretching was applied
        metrics: Additional synchronization metrics
    """

    def __init__(
        self,
        confidence: float,
        alignment_error: float,
        num_events: int,
        warp_applied: bool = False,
        metrics: Optional[Dict[str, Any]] = None
    ):
        self.confidence = confidence
        self.alignment_error = alignment_error
        self.num_events = num_events
        self.warp_applied = warp_applied
        self.metrics = metrics or {}

    def __repr__(self) -> str:
        return (f"SyncReport(confidence={self.confidence:.3f}, "
                f"error={self.alignment_error*1000:.1f}ms, "
                f"events={self.num_events}, warped={self.warp_applied})")


# Type aliases for ComfyUI integration
VISUAL_FEATURES_TYPE = "VISUAL_FEATURES"
AUDIO_DATA_TYPE = "AUDIO"
SCENE_ANALYSIS_TYPE = "SCENE_ANALYSIS"
SYNC_REPORT_TYPE = "SYNC_REPORT"
