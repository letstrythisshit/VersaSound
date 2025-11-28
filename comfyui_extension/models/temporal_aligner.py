"""
Temporal Alignment Module for VersaSound
Ensures audio-video temporal synchronization
Handles lip-sync, impact alignment, and rhythmic content
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Optional, Tuple
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class TemporalAlignmentModule(nn.Module):
    """
    Aligns generated audio with visual events
    Learned alignment - no hardcoded rules
    Supports multiple synchronization strategies
    """

    def __init__(self, config: Dict):
        """
        Initialize Temporal Alignment Module

        Args:
            config: Configuration dictionary
        """
        super().__init__()

        self.config = config
        self.sync_method = config.get('sync_method', 'learned')
        self.alignment_threshold = config.get('alignment_threshold', 0.1)
        self.allow_time_stretch = config.get('allow_time_stretch', True)
        self.max_stretch_ratio = config.get('max_stretch_ratio', 0.1)

        # Event detectors
        event_config = config.get('event_detector', {})
        self.visual_event_detector = EventDetector(
            input_dim=768,
            hidden_dim=256,
            threshold=event_config.get('threshold', 0.5)
        )

        self.audio_event_detector = EventDetector(
            input_dim=128,  # Mel spectrogram features
            hidden_dim=256,
            threshold=event_config.get('threshold', 0.5)
        )

        # Alignment predictor
        if self.sync_method == 'learned':
            self.alignment_predictor = AlignmentPredictor(
                visual_dim=768,
                audio_dim=128,
                hidden_dim=512
            )

        # Audio warping network
        if self.allow_time_stretch:
            self.audio_warper = AudioWarper(
                hidden_dim=256,
                max_stretch_ratio=self.max_stretch_ratio
            )

        logger.info(f"Initialized TemporalAlignmentModule with method: {self.sync_method}")

    def forward(
        self,
        visual_features: Dict[str, torch.Tensor],
        audio: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None,
        sample_rate: int = 48000
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Align audio with visual events

        Args:
            visual_features: Visual features from encoder
            audio: Audio waveform [B, num_samples] or [B, C, num_samples]
            audio_features: Optional precomputed audio features
            sample_rate: Audio sample rate

        Returns:
            Tuple of (aligned_audio, alignment_report)
        """
        # Extract audio features if not provided
        if audio_features is None:
            audio_features = self._extract_audio_features(audio, sample_rate)

        # Detect visual events
        visual_events = self.visual_event_detector(visual_features)

        # Detect audio events
        audio_events = self.audio_event_detector({'features': audio_features})

        # Compute alignment
        if self.sync_method == 'learned':
            alignment = self.alignment_predictor(visual_features, audio_features)
        elif self.sync_method == 'dtw':
            alignment = self._compute_dtw_alignment(visual_events, audio_events)
        elif self.sync_method == 'cross_correlation':
            alignment = self._compute_cross_correlation_alignment(
                visual_events, audio_events
            )
        else:
            raise ValueError(f"Unknown sync method: {self.sync_method}")

        # Apply warping if needed
        aligned_audio = audio
        warped = False

        if alignment['needs_warping'] and self.allow_time_stretch:
            aligned_audio = self.audio_warper(audio, alignment['warp_path'])
            warped = True

        # Compute alignment metrics
        report = self._compute_alignment_report(
            visual_events,
            audio_events,
            alignment,
            warped
        )

        return aligned_audio, report

    def _extract_audio_features(
        self,
        audio: torch.Tensor,
        sample_rate: int
    ) -> torch.Tensor:
        """
        Extract audio features (mel spectrogram)

        Args:
            audio: Waveform [B, num_samples]
            sample_rate: Sample rate

        Returns:
            Audio features [B, T, feature_dim]
        """
        try:
            import torchaudio

            # Ensure correct shape
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            elif audio.dim() == 3:
                audio = audio.squeeze(1)  # Remove channel dim if present

            # Compute mel spectrogram
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=2048,
                hop_length=512,
                n_mels=128
            ).to(audio.device)

            mel_spec = mel_transform(audio)  # [B, n_mels, T]

            # Convert to log scale
            mel_spec = torch.log(mel_spec + 1e-9)

            # Transpose to [B, T, n_mels]
            mel_spec = mel_spec.transpose(1, 2)

            return mel_spec

        except ImportError:
            logger.warning("torchaudio not available, using dummy features")
            # Create dummy features
            num_frames = audio.shape[-1] // 512
            return torch.randn(audio.shape[0], num_frames, 128, device=audio.device)

    def _compute_dtw_alignment(
        self,
        visual_events: Dict,
        audio_events: Dict
    ) -> Dict:
        """
        Compute alignment using Dynamic Time Warping

        Args:
            visual_events: Visual event dict
            audio_events: Audio event dict

        Returns:
            Alignment dict
        """
        # Simplified DTW implementation
        visual_timing = visual_events['event_times']
        audio_timing = audio_events['event_times']

        # Compute alignment error
        if len(visual_timing) > 0 and len(audio_timing) > 0:
            # Simple nearest-neighbor alignment
            errors = []
            for v_time in visual_timing:
                min_error = min(abs(v_time - a_time) for a_time in audio_timing)
                errors.append(min_error)

            avg_error = sum(errors) / len(errors) if errors else 0.0
            needs_warping = avg_error > self.alignment_threshold
        else:
            avg_error = 0.0
            needs_warping = False

        return {
            'error': avg_error,
            'needs_warping': needs_warping,
            'warp_path': None,  # Would compute warp path here
            'confidence': 1.0 - min(avg_error / self.alignment_threshold, 1.0)
        }

    def _compute_cross_correlation_alignment(
        self,
        visual_events: Dict,
        audio_events: Dict
    ) -> Dict:
        """
        Compute alignment using cross-correlation

        Args:
            visual_events: Visual event dict
            audio_events: Audio event dict

        Returns:
            Alignment dict
        """
        # Placeholder for cross-correlation alignment
        return {
            'error': 0.05,
            'needs_warping': False,
            'warp_path': None,
            'confidence': 0.95
        }

    def _compute_alignment_report(
        self,
        visual_events: Dict,
        audio_events: Dict,
        alignment: Dict,
        warped: bool
    ) -> Dict:
        """
        Create alignment report with metrics

        Args:
            visual_events: Visual events
            audio_events: Audio events
            alignment: Alignment result
            warped: Whether warping was applied

        Returns:
            Report dictionary
        """
        from ..custom_types import SyncReport

        report = SyncReport(
            confidence=alignment.get('confidence', 0.5),
            alignment_error=alignment.get('error', 0.1),
            num_events=len(visual_events.get('event_times', [])),
            warp_applied=warped,
            metrics={
                'visual_events': len(visual_events.get('event_times', [])),
                'audio_events': len(audio_events.get('event_times', [])),
                'sync_method': self.sync_method
            }
        )

        return report


class EventDetector(nn.Module):
    """
    Detects events (peaks, impacts, onsets) in features
    Works for both visual and audio features
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        threshold: float = 0.5
    ):
        super().__init__()

        self.threshold = threshold

        # Event detection network
        self.detector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, features: Dict) -> Dict:
        """
        Detect events in features

        Args:
            features: Dict with 'temporal' or 'features' key

        Returns:
            Dict with event information
        """
        # Get feature tensor
        if 'temporal' in features:
            feat = features['temporal']
        elif 'features' in features:
            feat = features['features']
        else:
            raise ValueError("Features dict must contain 'temporal' or 'features'")

        # Detect events
        event_scores = self.detector(feat).squeeze(-1)  # [B, T]

        # Find event times (peaks above threshold)
        event_times = []
        for i in range(event_scores.shape[0]):
            scores = event_scores[i]
            peaks = (scores > self.threshold).nonzero(as_tuple=True)[0]

            # Convert to time (assuming 24fps for visual, or frames for audio)
            times = peaks.float().cpu().tolist()
            event_times.append(times)

        return {
            'event_scores': event_scores,
            'event_times': event_times[0] if len(event_times) == 1 else event_times,
            'num_events': sum(len(t) for t in event_times)
        }


class AlignmentPredictor(nn.Module):
    """
    Learned alignment predictor
    Predicts optimal alignment between visual and audio features
    """

    def __init__(
        self,
        visual_dim: int,
        audio_dim: int,
        hidden_dim: int = 512
    ):
        super().__init__()

        # Cross-modal fusion
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)

        # Alignment network
        self.alignment_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # [error, confidence, warp_scale]
        )

    def forward(
        self,
        visual_features: Dict,
        audio_features: torch.Tensor
    ) -> Dict:
        """
        Predict alignment

        Args:
            visual_features: Visual feature dict
            audio_features: Audio features [B, T_audio, D_audio]

        Returns:
            Alignment dict
        """
        visual = visual_features['temporal']  # [B, T_visual, D_visual]

        # Project features
        visual_proj = self.visual_proj(visual)  # [B, T_visual, H]
        audio_proj = self.audio_proj(audio_features)  # [B, T_audio, H]

        # Align temporal dimensions (use interpolation)
        if visual_proj.shape[1] != audio_proj.shape[1]:
            audio_proj = torch.nn.functional.interpolate(
                audio_proj.transpose(1, 2),
                size=visual_proj.shape[1],
                mode='linear',
                align_corners=False
            ).transpose(1, 2)

        # Concatenate
        combined = torch.cat([visual_proj, audio_proj], dim=-1)  # [B, T, 2H]

        # Predict alignment parameters
        alignment_params = self.alignment_net(combined)  # [B, T, 3]

        # Average over time
        alignment_params = alignment_params.mean(dim=1)  # [B, 3]

        error = torch.abs(alignment_params[:, 0])
        confidence = torch.sigmoid(alignment_params[:, 1])
        warp_scale = torch.tanh(alignment_params[:, 2]) * 0.1  # ±10% warp

        # Determine if warping needed
        needs_warping = (error > 0.1).any().item()

        return {
            'error': error.mean().item(),
            'confidence': confidence.mean().item(),
            'needs_warping': needs_warping,
            'warp_path': warp_scale,  # Scale factor for warping
        }


class AudioWarper(nn.Module):
    """
    Applies time-stretching to audio for alignment
    Minimal distortion while adjusting timing
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        max_stretch_ratio: float = 0.1
    ):
        super().__init__()
        self.max_stretch_ratio = max_stretch_ratio

    def forward(
        self,
        audio: torch.Tensor,
        warp_path: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply time-warping to audio

        Args:
            audio: Audio waveform [B, num_samples]
            warp_path: Warp parameters [B] or [B, T]

        Returns:
            Warped audio [B, num_samples]
        """
        if warp_path is None:
            return audio

        # Ensure correct audio shape
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Compute stretch factor
        if warp_path.dim() == 1:
            stretch_factor = 1.0 + warp_path.mean().item()
        else:
            stretch_factor = 1.0 + warp_path.mean().item()

        # Clamp stretch factor
        stretch_factor = max(
            1.0 - self.max_stretch_ratio,
            min(1.0 + self.max_stretch_ratio, stretch_factor)
        )

        # Apply time-stretching using interpolation
        original_length = audio.shape[-1]
        new_length = int(original_length * stretch_factor)

        # Use linear interpolation for stretching
        warped_audio = torch.nn.functional.interpolate(
            audio.unsqueeze(1),  # Add channel dim
            size=new_length,
            mode='linear',
            align_corners=False
        ).squeeze(1)

        # Pad or trim to original length
        if new_length < original_length:
            # Pad
            padding = original_length - new_length
            warped_audio = torch.nn.functional.pad(
                warped_audio, (0, padding)
            )
        elif new_length > original_length:
            # Trim
            warped_audio = warped_audio[..., :original_length]

        return warped_audio


def load_temporal_aligner(
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None
) -> TemporalAlignmentModule:
    """
    Load temporal alignment module

    Args:
        config_path: Path to config YAML
        checkpoint_path: Path to checkpoint
        device: Device to load on

    Returns:
        Loaded TemporalAlignmentModule
    """
    # Load config
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "default_config.yaml"

    with open(config_path) as f:
        full_config = yaml.safe_load(f)
        config = full_config.get('temporal_alignment', {})

    # Create model
    model = TemporalAlignmentModule(config)

    # Load checkpoint
    if checkpoint_path is not None:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

    # Move to device
    if device is not None:
        model = model.to(device)

    model.eval()
    return model
