"""
Audio processing utilities for VersaSound
Handles audio loading, preprocessing, and transformations
"""

import torch
import numpy as np
import logging
from typing import Optional, Tuple, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Audio processing utilities
    Handles loading, resampling, and audio transformations
    """

    def __init__(
        self,
        target_sr: int = 48000,
        mono: bool = False,
        max_duration: Optional[float] = None
    ):
        """
        Initialize AudioProcessor

        Args:
            target_sr: Target sample rate in Hz
            mono: Whether to convert to mono
            max_duration: Maximum duration in seconds (None = unlimited)
        """
        self.target_sr = target_sr
        self.mono = mono
        self.max_duration = max_duration

    def load_audio(
        self,
        audio_path: Union[str, Path]
    ) -> Tuple[torch.Tensor, dict]:
        """
        Load audio from file

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (audio_tensor [channels, samples], metadata dict)
        """
        try:
            import torchaudio
            audio_path = str(audio_path)

            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)

            # Get metadata
            metadata = {
                'sample_rate': sample_rate,
                'duration': waveform.shape[-1] / sample_rate,
                'num_channels': waveform.shape[0],
                'num_samples': waveform.shape[-1]
            }

            # Preprocess
            waveform = self.preprocess_audio(waveform, sample_rate)

            # Update metadata after preprocessing
            metadata['processed_sr'] = self.target_sr
            metadata['processed_duration'] = waveform.shape[-1] / self.target_sr
            metadata['processed_channels'] = waveform.shape[0]

            logger.info(f"Loaded audio: {waveform.shape}, {self.target_sr} Hz")

            return waveform, metadata

        except ImportError:
            raise ImportError("torchaudio is required for audio loading")
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            raise

    def preprocess_audio(
        self,
        waveform: torch.Tensor,
        sample_rate: int
    ) -> torch.Tensor:
        """
        Preprocess audio waveform

        Args:
            waveform: Audio tensor [channels, samples]
            sample_rate: Current sample rate

        Returns:
            Preprocessed audio tensor
        """
        # Resample if needed
        if sample_rate != self.target_sr:
            waveform = self.resample(waveform, sample_rate, self.target_sr)

        # Convert to mono if needed
        if self.mono and waveform.shape[0] > 1:
            waveform = self.to_mono(waveform)

        # Limit duration if specified
        if self.max_duration is not None:
            max_samples = int(self.max_duration * self.target_sr)
            if waveform.shape[-1] > max_samples:
                waveform = waveform[..., :max_samples]

        return waveform

    def resample(
        self,
        waveform: torch.Tensor,
        orig_freq: int,
        new_freq: int
    ) -> torch.Tensor:
        """
        Resample audio to different sample rate

        Args:
            waveform: Audio tensor [channels, samples]
            orig_freq: Original sample rate
            new_freq: Target sample rate

        Returns:
            Resampled audio tensor
        """
        if orig_freq == new_freq:
            return waveform

        try:
            import torchaudio
            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_freq,
                new_freq=new_freq
            ).to(waveform.device)

            resampled = resampler(waveform)
            logger.debug(f"Resampled audio from {orig_freq} Hz to {new_freq} Hz")

            return resampled

        except ImportError:
            # Fallback to simple linear interpolation
            logger.warning("torchaudio not available, using simple resampling")
            ratio = new_freq / orig_freq
            new_length = int(waveform.shape[-1] * ratio)

            resampled = torch.nn.functional.interpolate(
                waveform.unsqueeze(0),
                size=new_length,
                mode='linear',
                align_corners=False
            ).squeeze(0)

            return resampled

    def to_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert audio to mono by averaging channels

        Args:
            waveform: Audio tensor [channels, samples]

        Returns:
            Mono audio tensor [1, samples]
        """
        if waveform.shape[0] == 1:
            return waveform

        return waveform.mean(dim=0, keepdim=True)

    def normalize(
        self,
        waveform: torch.Tensor,
        method: str = 'peak',
        target_level: float = -3.0
    ) -> torch.Tensor:
        """
        Normalize audio waveform

        Args:
            waveform: Audio tensor [channels, samples]
            method: Normalization method ('peak', 'rms', 'lufs')
            target_level: Target level in dB

        Returns:
            Normalized audio tensor
        """
        if method == 'peak':
            # Peak normalization
            peak = waveform.abs().max()
            if peak > 0:
                target_peak = 10 ** (target_level / 20)
                waveform = waveform * (target_peak / peak)

        elif method == 'rms':
            # RMS normalization
            rms = torch.sqrt(torch.mean(waveform ** 2))
            if rms > 0:
                target_rms = 10 ** (target_level / 20)
                waveform = waveform * (target_rms / rms)

        elif method == 'lufs':
            logger.warning("LUFS normalization not implemented, using RMS")
            return self.normalize(waveform, method='rms', target_level=target_level)

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        logger.debug(f"Normalized audio using {method} method")
        return waveform

    def extract_from_video(
        self,
        video_path: Union[str, Path]
    ) -> Tuple[torch.Tensor, dict]:
        """
        Extract audio from video file

        Args:
            video_path: Path to video file

        Returns:
            Tuple of (audio_tensor, metadata)
        """
        try:
            import torchvision
            video_path = str(video_path)

            # Read video (also extracts audio)
            _, audio, info = torchvision.io.read_video(
                video_path,
                pts_unit='sec'
            )

            if audio is None or audio.numel() == 0:
                raise ValueError(f"No audio track found in {video_path}")

            # audio shape from torchvision: [samples, channels]
            # Convert to [channels, samples]
            audio = audio.T

            sample_rate = info.get('audio_fps', 48000)

            metadata = {
                'sample_rate': sample_rate,
                'duration': audio.shape[-1] / sample_rate,
                'num_channels': audio.shape[0],
                'extracted_from_video': True
            }

            # Preprocess
            audio = self.preprocess_audio(audio, sample_rate)

            logger.info(f"Extracted audio from video: {audio.shape}")

            return audio, metadata

        except Exception as e:
            logger.error(f"Error extracting audio from video {video_path}: {e}")
            raise

    def compute_spectrogram(
        self,
        waveform: torch.Tensor,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128
    ) -> torch.Tensor:
        """
        Compute mel spectrogram

        Args:
            waveform: Audio tensor [channels, samples]
            n_fft: FFT size
            hop_length: Hop length
            n_mels: Number of mel bins

        Returns:
            Mel spectrogram [channels, n_mels, time]
        """
        try:
            import torchaudio

            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.target_sr,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels
            ).to(waveform.device)

            mel_spec = mel_transform(waveform)
            return mel_spec

        except ImportError:
            raise ImportError("torchaudio is required for spectrogram computation")

    def apply_effects(
        self,
        waveform: torch.Tensor,
        effects: dict
    ) -> torch.Tensor:
        """
        Apply audio effects

        Args:
            waveform: Audio tensor [channels, samples]
            effects: Dict of effects and their parameters

        Returns:
            Processed audio tensor
        """
        for effect_name, params in effects.items():
            if effect_name == 'gain':
                waveform = waveform * params.get('amount', 1.0)

            elif effect_name == 'reverb':
                waveform = self._apply_reverb(waveform, params)

            elif effect_name == 'eq':
                waveform = self._apply_eq(waveform, params)

            elif effect_name == 'noise_reduction':
                waveform = self._apply_noise_reduction(waveform, params)

            else:
                logger.warning(f"Unknown effect: {effect_name}")

        return waveform

    def _apply_reverb(self, waveform: torch.Tensor, params: dict) -> torch.Tensor:
        """Simple reverb effect using convolution"""
        # Simplified reverb - in production, use proper impulse response
        amount = params.get('amount', 0.3)

        # Create simple reverb impulse response
        ir_length = int(0.1 * self.target_sr)  # 100ms reverb
        ir = torch.exp(-torch.linspace(0, 5, ir_length))
        ir = ir / ir.sum()
        ir = ir.to(waveform.device)

        # Apply convolution
        reverb = torch.nn.functional.conv1d(
            waveform.unsqueeze(0),
            ir.view(1, 1, -1),
            padding=ir_length // 2
        ).squeeze(0)

        # Mix with dry signal
        wet_dry = amount
        return (1 - wet_dry) * waveform + wet_dry * reverb[:, :waveform.shape[-1]]

    def _apply_eq(self, waveform: torch.Tensor, params: dict) -> torch.Tensor:
        """Simple EQ using filtering"""
        # Placeholder for EQ - would need proper filter implementation
        logger.warning("EQ effect not fully implemented")
        return waveform

    def _apply_noise_reduction(self, waveform: torch.Tensor, params: dict) -> torch.Tensor:
        """Simple noise reduction using spectral gating"""
        # Placeholder for noise reduction
        logger.warning("Noise reduction not fully implemented")
        return waveform

    def pad_or_trim(
        self,
        waveform: torch.Tensor,
        target_length: int
    ) -> torch.Tensor:
        """
        Pad or trim audio to target length

        Args:
            waveform: Audio tensor [channels, samples]
            target_length: Target length in samples

        Returns:
            Padded/trimmed audio tensor
        """
        current_length = waveform.shape[-1]

        if current_length < target_length:
            # Pad with zeros
            padding = target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        elif current_length > target_length:
            # Trim
            waveform = waveform[..., :target_length]

        return waveform

    def detect_silence(
        self,
        waveform: torch.Tensor,
        threshold_db: float = -40.0
    ) -> list:
        """
        Detect silent regions in audio

        Args:
            waveform: Audio tensor [channels, samples]
            threshold_db: Silence threshold in dB

        Returns:
            List of (start, end) tuples for non-silent regions
        """
        # Convert to mono for silence detection
        if waveform.shape[0] > 1:
            mono = waveform.mean(dim=0)
        else:
            mono = waveform[0]

        # Compute energy in frames
        frame_length = int(0.02 * self.target_sr)  # 20ms frames
        hop_length = frame_length // 2

        frames = mono.unfold(0, frame_length, hop_length)
        energy = torch.sqrt(torch.mean(frames ** 2, dim=1))

        # Convert to dB
        energy_db = 20 * torch.log10(energy + 1e-10)

        # Find non-silent frames
        non_silent = energy_db > threshold_db

        # Convert frame indices to sample indices
        regions = []
        in_region = False
        start = 0

        for i, is_sound in enumerate(non_silent):
            if is_sound and not in_region:
                start = i * hop_length
                in_region = True
            elif not is_sound and in_region:
                end = i * hop_length
                regions.append((start, end))
                in_region = False

        if in_region:
            regions.append((start, len(mono)))

        return regions


def blend_audio(
    audio_list: list,
    weights: Optional[list] = None,
    method: str = 'weighted_sum'
) -> torch.Tensor:
    """
    Blend multiple audio tracks

    Args:
        audio_list: List of audio tensors [channels, samples]
        weights: Weights for each audio (None = equal weights)
        method: Blending method ('weighted_sum', 'max', 'average')

    Returns:
        Blended audio tensor
    """
    if len(audio_list) == 0:
        raise ValueError("No audio to blend")

    if len(audio_list) == 1:
        return audio_list[0]

    # Ensure all audio has same shape
    max_length = max(a.shape[-1] for a in audio_list)
    max_channels = max(a.shape[0] for a in audio_list)

    # Pad all audio to same length and channels
    padded_audio = []
    for audio in audio_list:
        # Pad channels if needed
        if audio.shape[0] < max_channels:
            channel_padding = torch.zeros(
                max_channels - audio.shape[0], audio.shape[-1],
                dtype=audio.dtype, device=audio.device
            )
            audio = torch.cat([audio, channel_padding], dim=0)

        # Pad length if needed
        if audio.shape[-1] < max_length:
            length_padding = max_length - audio.shape[-1]
            audio = torch.nn.functional.pad(audio, (0, length_padding))

        padded_audio.append(audio)

    # Blend based on method
    if method == 'weighted_sum':
        if weights is None:
            weights = [1.0 / len(padded_audio)] * len(padded_audio)

        blended = sum(w * a for w, a in zip(weights, padded_audio))

    elif method == 'max':
        blended = torch.stack(padded_audio, dim=0).max(dim=0)[0]

    elif method == 'average':
        blended = torch.stack(padded_audio, dim=0).mean(dim=0)

    else:
        raise ValueError(f"Unknown blending method: {method}")

    return blended
