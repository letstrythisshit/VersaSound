"""
Utilities for working with ComfyUI latent representations
Handles conversion between latents and visual features
"""

import torch
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class LatentProcessor:
    """
    Processes ComfyUI latent representations
    Converts latents to visual features efficiently
    """

    def __init__(self):
        self.vae_cache = {}

    def latent_to_visual_features(
        self,
        latent_dict: Dict[str, torch.Tensor],
        vae: Optional[object] = None,
        decode_stride: int = 8,
        use_cached: bool = True
    ) -> torch.Tensor:
        """
        Convert ComfyUI latent to visual features

        Args:
            latent_dict: ComfyUI latent dictionary with 'samples' key
            vae: ComfyUI VAE object (optional, for decoding)
            decode_stride: Decode every Nth frame (1 = decode all)
            use_cached: Use cached VAE decoder if available

        Returns:
            Visual features tensor [B, T, C, H, W] or [B, T, latent_dim]
        """
        if 'samples' not in latent_dict:
            raise ValueError("Latent dict must contain 'samples' key")

        latent_samples = latent_dict['samples']
        # latent_samples shape: [B, C, H, W] or [B, T, C, H, W]

        # If 4D, assume single frame/image
        if latent_samples.ndim == 4:
            latent_samples = latent_samples.unsqueeze(1)  # Add temporal dim

        B, T, C, H, W = latent_samples.shape

        logger.debug(f"Processing latent: {latent_samples.shape}")

        if vae is not None and decode_stride > 0:
            # Decode latents to pixel space
            decoded_frames = self._decode_latent_frames(
                latent_samples, vae, decode_stride
            )
            return decoded_frames
        else:
            # Return latents directly (will be processed by visual encoder)
            # Reshape to [B, T, C*H*W] for feature extraction
            features = latent_samples.view(B, T, -1)
            return features

    def _decode_latent_frames(
        self,
        latent: torch.Tensor,
        vae: object,
        stride: int
    ) -> torch.Tensor:
        """
        Decode latent frames to pixel space

        Args:
            latent: Latent tensor [B, T, C, H, W]
            vae: ComfyUI VAE decoder
            stride: Decode every Nth frame

        Returns:
            Decoded frames [B, T, 3, H', W']
        """
        B, T, C, H, W = latent.shape
        decoded_frames = []

        # Decode keyframes
        for t in range(0, T, stride):
            # Get latent for this frame
            latent_frame = latent[:, t]  # [B, C, H, W]

            # Decode using VAE
            try:
                with torch.no_grad():
                    decoded = vae.decode(latent_frame)
                decoded_frames.append(decoded)

            except Exception as e:
                logger.error(f"Error decoding frame {t}: {e}")
                # Use placeholder
                decoded = torch.zeros(B, 3, H * 8, W * 8, device=latent.device)
                decoded_frames.append(decoded)

        # Interpolate between decoded frames if stride > 1
        if stride > 1:
            decoded_frames = self._interpolate_frames(decoded_frames, T, stride)
        else:
            decoded_frames = torch.stack(decoded_frames, dim=1)

        logger.debug(f"Decoded {T} frames to {decoded_frames.shape}")
        return decoded_frames

    def _interpolate_frames(
        self,
        decoded_keyframes: list,
        total_frames: int,
        stride: int
    ) -> torch.Tensor:
        """
        Interpolate between decoded keyframes

        Args:
            decoded_keyframes: List of decoded keyframes
            total_frames: Total number of frames needed
            stride: Stride between keyframes

        Returns:
            Interpolated frames [B, T, C, H, W]
        """
        # Stack keyframes
        keyframes = torch.stack(decoded_keyframes, dim=1)  # [B, K, C, H, W]
        B, K, C, H, W = keyframes.shape

        # Create indices for all frames
        keyframe_indices = torch.arange(0, total_frames, stride, dtype=torch.float32)
        target_indices = torch.arange(0, total_frames, dtype=torch.float32)

        # Linear interpolation
        interpolated_frames = []

        for t in target_indices:
            # Find surrounding keyframes
            k_idx = t // stride
            k_idx = min(int(k_idx), K - 1)

            if k_idx >= K - 1:
                # Last keyframe
                interpolated_frames.append(keyframes[:, -1])
            else:
                # Interpolate between k_idx and k_idx + 1
                k_next = k_idx + 1
                alpha = (t - keyframe_indices[k_idx]) / stride

                frame = (1 - alpha) * keyframes[:, k_idx] + alpha * keyframes[:, k_next]
                interpolated_frames.append(frame)

        result = torch.stack(interpolated_frames, dim=1)
        return result

    def extract_latent_metadata(self, latent_dict: Dict) -> Dict:
        """
        Extract metadata from latent dictionary

        Args:
            latent_dict: ComfyUI latent dictionary

        Returns:
            Metadata dictionary
        """
        samples = latent_dict.get('samples')

        if samples is None:
            return {}

        metadata = {
            'shape': samples.shape,
            'device': str(samples.device),
            'dtype': str(samples.dtype),
        }

        # Check for temporal dimension
        if samples.ndim == 5:
            metadata['has_temporal'] = True
            metadata['num_frames'] = samples.shape[1]
        elif samples.ndim == 4:
            metadata['has_temporal'] = False
            metadata['num_frames'] = 1

        # Extract other metadata from dict
        for key in ['batch_index', 'noise_mask', 'samples_cfg']:
            if key in latent_dict:
                metadata[key] = latent_dict[key]

        return metadata

    def prepare_latent_for_vae(
        self,
        latent: torch.Tensor,
        vae_scaling_factor: float = 0.18215
    ) -> torch.Tensor:
        """
        Prepare latent for VAE decoding

        Args:
            latent: Latent tensor
            vae_scaling_factor: VAE scaling factor (SD default: 0.18215)

        Returns:
            Scaled latent ready for decoding
        """
        return latent / vae_scaling_factor


def detect_latent_type(latent_dict: Dict) -> str:
    """
    Detect the type of latent (SD1.5, SDXL, SVD, etc.)

    Args:
        latent_dict: ComfyUI latent dictionary

    Returns:
        Latent type string
    """
    if 'samples' not in latent_dict:
        return 'unknown'

    samples = latent_dict['samples']
    shape = samples.shape

    # Detect based on channel dimension
    if samples.ndim == 4:
        _, C, H, W = shape

        if C == 4:
            # Standard SD latent
            if H >= 128 or W >= 128:
                return 'sdxl'
            else:
                return 'sd15'
        elif C == 8:
            return 'sdxl_refiner'

    elif samples.ndim == 5:
        _, T, C, H, W = shape

        if C == 4:
            return 'svd'  # Stable Video Diffusion
        elif C == 8:
            return 'svd_xt'

    return 'unknown'


def get_vae_downsampling_factor(latent_type: str) -> int:
    """
    Get VAE downsampling factor for latent type

    Args:
        latent_type: Type of latent

    Returns:
        Downsampling factor
    """
    factors = {
        'sd15': 8,
        'sdxl': 8,
        'sdxl_refiner': 8,
        'svd': 8,
        'svd_xt': 8,
        'unknown': 8
    }

    return factors.get(latent_type, 8)
