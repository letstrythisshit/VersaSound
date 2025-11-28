"""
Video processing utilities for VersaSound
Handles video loading, preprocessing, and feature extraction helpers
"""

import torch
import numpy as np
import logging
from typing import Optional, Tuple, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Video processing utilities
    Handles various input formats: video files, image sequences, latents
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        target_fps: float = 24.0,
        max_frames: Optional[int] = None
    ):
        """
        Initialize VideoProcessor

        Args:
            target_size: Target resolution (height, width)
            target_fps: Target frame rate
            max_frames: Maximum number of frames to process (None = unlimited)
        """
        self.target_size = target_size
        self.target_fps = target_fps
        self.max_frames = max_frames

    def load_video(self, video_path: Union[str, Path]) -> Tuple[torch.Tensor, dict]:
        """
        Load video from file

        Args:
            video_path: Path to video file

        Returns:
            Tuple of (video_tensor [T, C, H, W], metadata dict)
        """
        try:
            import torchvision
            video_path = str(video_path)

            # Load video
            video, audio, info = torchvision.io.read_video(
                video_path,
                pts_unit='sec'
            )

            # video shape: [T, H, W, C]
            # Convert to [T, C, H, W]
            video = video.permute(0, 3, 1, 2).float() / 255.0

            # Get metadata
            metadata = {
                'fps': info.get('video_fps', 30.0),
                'duration': video.shape[0] / info.get('video_fps', 30.0),
                'num_frames': video.shape[0],
                'original_size': (video.shape[2], video.shape[3]),
                'has_audio': audio is not None and audio.numel() > 0
            }

            # Preprocess
            video = self.preprocess_video(video, metadata['fps'])

            logger.info(f"Loaded video: {video.shape}, {metadata['fps']} fps")

            return video, metadata

        except ImportError:
            raise ImportError("torchvision is required for video loading")
        except Exception as e:
            logger.error(f"Error loading video {video_path}: {e}")
            raise

    def preprocess_video(
        self,
        video: torch.Tensor,
        current_fps: float
    ) -> torch.Tensor:
        """
        Preprocess video tensor

        Args:
            video: Video tensor [T, C, H, W]
            current_fps: Current frame rate

        Returns:
            Preprocessed video tensor
        """
        # Resample FPS if needed
        if current_fps != self.target_fps:
            video = self.resample_fps(video, current_fps, self.target_fps)

        # Resize frames
        video = self.resize_video(video, self.target_size)

        # Limit frames if specified
        if self.max_frames is not None and video.shape[0] > self.max_frames:
            video = video[:self.max_frames]

        return video

    def resample_fps(
        self,
        video: torch.Tensor,
        from_fps: float,
        to_fps: float
    ) -> torch.Tensor:
        """
        Resample video to different frame rate

        Args:
            video: Video tensor [T, C, H, W]
            from_fps: Source frame rate
            to_fps: Target frame rate

        Returns:
            Resampled video tensor
        """
        if from_fps == to_fps:
            return video

        # Calculate new frame indices
        num_frames = video.shape[0]
        duration = num_frames / from_fps
        target_frames = int(duration * to_fps)

        # Use linear interpolation
        indices = torch.linspace(0, num_frames - 1, target_frames)

        # Interpolate frames
        resampled_video = []
        for idx in indices:
            if idx.is_integer():
                resampled_video.append(video[int(idx)])
            else:
                # Linear interpolation between frames
                idx_low = int(torch.floor(idx))
                idx_high = min(int(torch.ceil(idx)), num_frames - 1)
                alpha = idx - idx_low

                frame = (1 - alpha) * video[idx_low] + alpha * video[idx_high]
                resampled_video.append(frame)

        resampled_video = torch.stack(resampled_video, dim=0)
        logger.debug(f"Resampled video from {from_fps} fps to {to_fps} fps: "
                    f"{video.shape[0]} -> {resampled_video.shape[0]} frames")

        return resampled_video

    def resize_video(
        self,
        video: torch.Tensor,
        target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Resize all frames in video

        Args:
            video: Video tensor [T, C, H, W]
            target_size: Target size (height, width)

        Returns:
            Resized video tensor
        """
        if video.shape[2:] == target_size:
            return video

        try:
            import torch.nn.functional as F

            # Resize each frame
            T, C, H, W = video.shape
            resized_frames = []

            # Process in batches for efficiency
            batch_size = 16
            for i in range(0, T, batch_size):
                batch = video[i:i+batch_size]
                resized_batch = F.interpolate(
                    batch,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )
                resized_frames.append(resized_batch)

            resized_video = torch.cat(resized_frames, dim=0)
            logger.debug(f"Resized video from {(H, W)} to {target_size}")

            return resized_video

        except Exception as e:
            logger.error(f"Error resizing video: {e}")
            raise

    def normalize_video(
        self,
        video: torch.Tensor,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ) -> torch.Tensor:
        """
        Normalize video frames (ImageNet normalization by default)

        Args:
            video: Video tensor [T, C, H, W]
            mean: Mean for each channel
            std: Std for each channel

        Returns:
            Normalized video tensor
        """
        mean = torch.tensor(mean).view(1, 3, 1, 1).to(video.device)
        std = torch.tensor(std).view(1, 3, 1, 1).to(video.device)

        return (video - mean) / std

    def extract_frames_at_indices(
        self,
        video: torch.Tensor,
        indices: list
    ) -> torch.Tensor:
        """
        Extract specific frames from video

        Args:
            video: Video tensor [T, C, H, W]
            indices: List of frame indices to extract

        Returns:
            Extracted frames [N, C, H, W]
        """
        indices = [idx for idx in indices if 0 <= idx < video.shape[0]]
        return video[indices]

    def compute_optical_flow(
        self,
        video: torch.Tensor,
        method: str = 'simple'
    ) -> torch.Tensor:
        """
        Compute optical flow between consecutive frames

        Args:
            video: Video tensor [T, C, H, W]
            method: Flow computation method ('simple' for frame differencing)

        Returns:
            Flow tensor [T-1, 2, H, W] (x and y flow)
        """
        if method == 'simple':
            # Simple frame differencing as proxy for flow
            flow = video[1:] - video[:-1]
            # Return magnitude and direction as 2-channel output
            flow_magnitude = torch.norm(flow, dim=1, keepdim=True)
            flow_x = flow[:, 0:1]  # Red channel diff
            flow_y = flow[:, 1:2]  # Green channel diff

            optical_flow = torch.cat([flow_x, flow_y], dim=1)
            return optical_flow
        else:
            raise NotImplementedError(f"Flow method '{method}' not implemented")

    def detect_scene_changes(
        self,
        video: torch.Tensor,
        threshold: float = 0.3
    ) -> list:
        """
        Detect scene changes in video

        Args:
            video: Video tensor [T, C, H, W]
            threshold: Threshold for scene change detection

        Returns:
            List of frame indices where scene changes occur
        """
        # Compute frame differences
        diffs = torch.norm(video[1:] - video[:-1], dim=(1, 2, 3))
        diffs = diffs / (video.shape[1] * video.shape[2] * video.shape[3])

        # Find peaks above threshold
        scene_changes = [0]  # First frame is always a scene boundary
        for i, diff in enumerate(diffs):
            if diff > threshold:
                scene_changes.append(i + 1)

        logger.debug(f"Detected {len(scene_changes)} scene changes")
        return scene_changes


def extract_uniform_frames(video: torch.Tensor, num_frames: int) -> torch.Tensor:
    """
    Extract uniformly spaced frames from video

    Args:
        video: Video tensor [T, C, H, W]
        num_frames: Number of frames to extract

    Returns:
        Extracted frames [num_frames, C, H, W]
    """
    T = video.shape[0]
    if num_frames >= T:
        return video

    indices = torch.linspace(0, T - 1, num_frames).long()
    return video[indices]


def create_sliding_window_clips(
    video: torch.Tensor,
    clip_length: int,
    stride: int
) -> torch.Tensor:
    """
    Create overlapping clips from video

    Args:
        video: Video tensor [T, C, H, W]
        clip_length: Length of each clip
        stride: Stride between clips

    Returns:
        Clips tensor [N, clip_length, C, H, W]
    """
    T = video.shape[0]
    clips = []

    for start_idx in range(0, T - clip_length + 1, stride):
        clip = video[start_idx:start_idx + clip_length]
        clips.append(clip)

    if len(clips) == 0:
        # Video too short, return single clip (padded if needed)
        if T < clip_length:
            padding = torch.zeros(
                clip_length - T, *video.shape[1:],
                dtype=video.dtype, device=video.device
            )
            clip = torch.cat([video, padding], dim=0)
        else:
            clip = video[:clip_length]
        clips.append(clip)

    return torch.stack(clips, dim=0)
