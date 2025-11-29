"""
Universal Visual Encoder for VersaSound
Extracts visual features from video for audio generation
No hardcoded scenario assumptions - learns from data
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Optional, Tuple
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class UniversalVisualEncoder(nn.Module):
    """
    Extracts comprehensive visual features for any video content
    Designed to be versatile and data-driven

    Architecture:
    1. Pretrained backbone (frozen) for basic visual features
    2. Trainable motion encoder for motion analysis
    3. Temporal transformer for temporal modeling
    4. Audio cue extractor for audio-relevant features
    """

    def __init__(self, config: Dict):
        """
        Initialize Visual Encoder

        Args:
            config: Configuration dictionary with model parameters
        """
        super().__init__()

        self.config = config
        self.backbone_name = config.get('backbone_name', 'videomae_base')
        self.output_dim = config.get('output_dim', 768)
        self.freeze_backbone = config.get('freeze_backbone', True)

        # Expected input size for backbone (will resize to this)
        self.expected_size = config.get('input_size', 224)

        # Load pretrained backbone
        self.backbone, self.backbone_dim = self._load_backbone()

        # Feature projection
        self.feature_projector = nn.Sequential(
            nn.Linear(self.backbone_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Motion encoder
        motion_config = config.get('motion_encoder', {})
        self.motion_encoder = MotionEncoder(
            input_channels=3,
            hidden_dim=motion_config.get('hidden_dim', 256),
            num_layers=motion_config.get('num_layers', 3)
        )

        # Temporal transformer
        num_temporal_layers = config.get('num_temporal_layers', 4)
        num_heads = config.get('num_heads', 12)

        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.output_dim,
                nhead=num_heads,
                dim_feedforward=self.output_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=num_temporal_layers
        )

        # Audio cue extractor
        audio_cue_config = config.get('audio_cue_extractor', {})
        if audio_cue_config.get('enabled', True):
            self.audio_cue_extractor = AudioCueExtractor(
                visual_dim=self.output_dim,
                motion_dim=motion_config.get('hidden_dim', 256),
                hidden_dim=audio_cue_config.get('hidden_dim', 512)
            )
        else:
            self.audio_cue_extractor = None

        logger.info(f"Initialized UniversalVisualEncoder with backbone: {self.backbone_name}")

    def _load_backbone(self) -> Tuple[nn.Module, int]:
        """
        Load pretrained visual backbone

        Returns:
            Tuple of (backbone model, output dimension)
        """
        backbone_name = self.backbone_name.lower()

        if 'videomae' in backbone_name:
            backbone, dim = self._load_videomae_backbone()
        elif 'clip' in backbone_name:
            backbone, dim = self._load_clip_backbone()
        elif 'dinov2' in backbone_name:
            backbone, dim = self._load_dinov2_backbone()
        else:
            logger.warning(f"Unknown backbone: {backbone_name}, using ResNet3D")
            backbone, dim = self._load_resnet3d_backbone()

        # Freeze backbone if specified
        if self.freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
            backbone.eval()
            logger.info(f"Froze backbone parameters")

        return backbone, dim

    def _load_videomae_backbone(self) -> Tuple[nn.Module, int]:
        """Load VideoMAE backbone"""
        try:
            from transformers import VideoMAEModel

            if 'large' in self.backbone_name:
                model_name = "MCG-NJU/videomae-large"
                output_dim = 1024
            else:
                model_name = "MCG-NJU/videomae-base"
                output_dim = 768

            backbone = VideoMAEModel.from_pretrained(model_name)
            logger.info(f"Loaded VideoMAE backbone: {model_name}")

            return backbone, output_dim

        except ImportError:
            logger.error("transformers library required for VideoMAE")
            raise
        except Exception as e:
            logger.error(f"Error loading VideoMAE: {e}")
            raise

    def _load_clip_backbone(self) -> Tuple[nn.Module, int]:
        """Load CLIP vision backbone"""
        try:
            import clip

            model, _ = clip.load("ViT-B/16", device="cpu")
            backbone = model.visual
            output_dim = 512

            logger.info("Loaded CLIP ViT-B/16 backbone")
            return backbone, output_dim

        except ImportError:
            logger.error("clip library required for CLIP backbone")
            raise

    def _load_dinov2_backbone(self) -> Tuple[nn.Module, int]:
        """Load DINOv2 backbone"""
        try:
            backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            output_dim = 768

            logger.info("Loaded DINOv2 ViT-B/14 backbone")
            return backbone, output_dim

        except Exception as e:
            logger.error(f"Error loading DINOv2: {e}")
            raise

    def _load_resnet3d_backbone(self) -> Tuple[nn.Module, int]:
        """Fallback: Load ResNet3D backbone"""
        try:
            from torchvision.models.video import r3d_18

            backbone = r3d_18(pretrained=True)
            # Remove final classification layer
            backbone = nn.Sequential(*list(backbone.children())[:-1])
            output_dim = 512

            logger.info("Loaded ResNet3D-18 backbone")
            return backbone, output_dim

        except Exception as e:
            logger.error(f"Error loading ResNet3D: {e}")
            raise

    def _resize_video(self, video: torch.Tensor, target_size: int) -> torch.Tensor:
        """
        Resize video frames to target size

        Args:
            video: Video tensor [B, T, C, H, W]
            target_size: Target height/width

        Returns:
            Resized video [B, T, C, target_size, target_size]
        """
        B, T, C, H, W = video.shape

        # Check if resize needed
        if H == target_size and W == target_size:
            return video

        logger.debug(f"Resizing video from {H}x{W} to {target_size}x{target_size}")

        # Reshape for interpolation: [B*T, C, H, W]
        video_flat = video.reshape(B * T, C, H, W)

        # Resize using bilinear interpolation
        video_resized = torch.nn.functional.interpolate(
            video_flat,
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False
        )

        # Reshape back to [B, T, C, H, W]
        video_resized = video_resized.reshape(B, T, C, target_size, target_size)

        return video_resized

    def forward(
        self,
        video: torch.Tensor,
        return_intermediates: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Extract visual features from video

        Args:
            video: Video tensor [B, T, C, H, W] or [B, C, T, H, W]
            return_intermediates: Whether to return intermediate features

        Returns:
            Dictionary containing:
                - backbone: Backbone features [B, T, D]
                - motion: Motion features [B, T, M]
                - temporal: Temporally-modeled features [B, T, D]
                - audio_cues: Audio-relevant cues (dict)
                - intermediates: (optional) Intermediate features
        """
        # Ensure correct input format [B, T, C, H, W]
        if video.dim() == 5 and video.shape[2] == 3:
            # Already in correct format
            pass
        elif video.dim() == 5 and video.shape[1] == 3:
            # Convert [B, C, T, H, W] -> [B, T, C, H, W]
            video = video.permute(0, 2, 1, 3, 4)
        elif video.dim() == 4:
            # Single frame or image batch [B, C, H, W]
            video = video.unsqueeze(1)  # Add temporal dimension
        else:
            raise ValueError(f"Unexpected video shape: {video.shape}")

        B, T, C, H, W = video.shape

        # Resize to expected size for backbone
        video = self._resize_video(video, self.expected_size)

        # Extract backbone features
        backbone_features = self._extract_backbone_features(video)
        # backbone_features: [B, T, backbone_dim]

        # Project to output dimension
        projected = self.feature_projector(backbone_features)
        # projected: [B, T, output_dim]

        # Extract motion features
        motion_features = self.motion_encoder(video)
        # motion_features: [B, T, motion_dim]

        # Temporal modeling
        temporal_features = self.temporal_transformer(projected)
        # temporal_features: [B, T, output_dim]

        # Extract audio cues
        if self.audio_cue_extractor is not None:
            audio_cues = self.audio_cue_extractor(
                visual_features=temporal_features,
                motion_features=motion_features,
                video=video
            )
        else:
            audio_cues = {}

        output = {
            'backbone': projected,
            'motion': motion_features,
            'temporal': temporal_features,
            'audio_cues': audio_cues,
        }

        if return_intermediates:
            output['intermediates'] = {
                'raw_backbone': backbone_features,
                'pre_temporal': projected,
            }

        return output

    def _extract_backbone_features(self, video: torch.Tensor) -> torch.Tensor:
        """
        Extract features using backbone

        Args:
            video: [B, T, C, H, W]

        Returns:
            Features [B, T, D]
        """
        B, T, C, H, W = video.shape

        # Flatten batch and time for backbone processing
        video_flat = video.reshape(B * T, C, H, W)

        with torch.set_grad_enabled(not self.freeze_backbone):
            if 'videomae' in self.backbone_name.lower():
                # VideoMAE expects [B, C, T, H, W] per sample
                # Process in temporal chunks
                features = []
                chunk_size = min(16, T)  # Process 16 frames at a time

                for i in range(0, T, chunk_size):
                    end = min(i + chunk_size, T)
                    chunk = video[:, i:end]  # [B, chunk_T, C, H, W]

                    # Reshape for VideoMAE
                    chunk = chunk.permute(0, 2, 1, 3, 4)  # [B, C, chunk_T, H, W]

                    outputs = self.backbone(chunk, output_hidden_states=True)
                    # Get CLS token or mean pool
                    feat = outputs.last_hidden_state[:, 0]  # [B, D]

                    # Expand to match chunk size
                    feat = feat.unsqueeze(1).expand(-1, end - i, -1)
                    features.append(feat)

                features = torch.cat(features, dim=1)  # [B, T, D]

            elif 'clip' in self.backbone_name.lower():
                # CLIP processes frames independently
                features_flat = self.backbone(video_flat)  # [B*T, D]
                features = features_flat.reshape(B, T, -1)

            else:
                # Generic backbone
                features_flat = self.backbone(video_flat)  # [B*T, D, ...]

                # Handle different output formats
                if features_flat.dim() > 2:
                    features_flat = features_flat.mean(dim=list(range(2, features_flat.dim())))

                features = features_flat.reshape(B, T, -1)

        return features


class MotionEncoder(nn.Module):
    """
    Encode motion information from video
    Extracts motion magnitude, direction, and acceleration
    """

    def __init__(
        self,
        input_channels: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 3
    ):
        super().__init__()

        self.conv_layers = nn.ModuleList()
        in_channels = input_channels

        for i in range(num_layers):
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim // (2 ** i), kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim // (2 ** i)),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ))
            in_channels = hidden_dim // (2 ** i)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.output_dim = hidden_dim // (2 ** (num_layers - 1))

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Extract motion features

        Args:
            video: [B, T, C, H, W]

        Returns:
            Motion features [B, T, motion_dim]
        """
        B, T, C, H, W = video.shape

        # Compute frame differences (optical flow proxy)
        if T > 1:
            flow = video[:, 1:] - video[:, :-1]  # [B, T-1, C, H, W]
            # Pad first frame
            first_frame = torch.zeros_like(video[:, 0:1])
            flow = torch.cat([first_frame, flow], dim=1)  # [B, T, C, H, W]
        else:
            flow = torch.zeros_like(video)

        # Process flow through conv layers
        flow_flat = flow.reshape(B * T, C, H, W)

        features = flow_flat
        for conv_layer in self.conv_layers:
            features = conv_layer(features)

        # Global pooling
        features = self.global_pool(features)  # [B*T, C, 1, 1]
        features = features.reshape(B, T, -1)  # [B, T, motion_dim]

        return features


class AudioCueExtractor(nn.Module):
    """
    Extract audio-relevant cues from visual features
    Identifies events, intensity, periodicity for audio generation
    """

    def __init__(
        self,
        visual_dim: int,
        motion_dim: int,
        hidden_dim: int = 512
    ):
        super().__init__()

        combined_dim = visual_dim + motion_dim

        # Intensity envelope predictor
        self.intensity_predictor = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 0-1 intensity
        )

        # Contact/impact detector
        self.contact_detector = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Probability of contact
        )

        # Velocity estimator
        self.velocity_estimator = nn.Sequential(
            nn.Linear(motion_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # -1 to 1 velocity
        )

        # Periodicity detector
        self.periodicity_detector = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Periodicity score
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        motion_features: torch.Tensor,
        video: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract audio cues

        Args:
            visual_features: [B, T, visual_dim]
            motion_features: [B, T, motion_dim]
            video: Optional raw video [B, T, C, H, W]

        Returns:
            Dictionary of audio cues
        """
        # Combine visual and motion features
        combined = torch.cat([visual_features, motion_features], dim=-1)

        # Extract cues
        intensity = self.intensity_predictor(combined).squeeze(-1)  # [B, T]
        contact = self.contact_detector(combined).squeeze(-1)  # [B, T]
        velocity = self.velocity_estimator(motion_features).squeeze(-1)  # [B, T]
        periodicity = self.periodicity_detector(combined).squeeze(-1)  # [B, T]

        return {
            'intensity_envelope': intensity,
            'contact_detection': contact,
            'velocity_estimation': velocity,
            'periodic_motion': periodicity,
        }


def load_visual_encoder(
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None
) -> UniversalVisualEncoder:
    """
    Load visual encoder from config and checkpoint

    Args:
        config_path: Path to config YAML file
        checkpoint_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        Loaded UniversalVisualEncoder
    """
    # Load config
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "default_config.yaml"

    with open(config_path) as f:
        full_config = yaml.safe_load(f)
        config = full_config.get('visual_encoder', {})

    # Create model
    model = UniversalVisualEncoder(config)

    # Load checkpoint if provided
    if checkpoint_path is not None:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        logger.info("Checkpoint loaded successfully")

    # Move to device
    if device is not None:
        model = model.to(device)

    return model
