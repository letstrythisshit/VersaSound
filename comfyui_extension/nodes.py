"""
ComfyUI Nodes for VersaSound
Complete set of nodes for video-to-audio generation
"""

import torch
import logging
from typing import Dict, Tuple, Optional, Any
import numpy as np

# Import custom types and utilities
from .custom_types import (
    VisualFeatures, AudioData, SceneAnalysis, SyncReport,
    VISUAL_FEATURES_TYPE, AUDIO_DATA_TYPE, SCENE_ANALYSIS_TYPE, SYNC_REPORT_TYPE
)
from .utils import (
    get_device, VideoProcessor, AudioProcessor, blend_audio
)
from .utils.latent_utils import LatentProcessor
from .models import (
    load_visual_encoder, load_audio_generator, load_temporal_aligner
)
from .models.model_utils import model_cache

logger = logging.getLogger(__name__)


class VisualFeatureExtractor:
    """
    Extract visual features from video frames or latents

    INPUTS (choose ONE):
    - Connect 'video' for raw images/video (most common)
    - OR connect 'latents' + 'vae' for latent workflow

    Don't connect both! Use only one workflow.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (["versatile", "speech_optimized", "foley_optimized"],),
            },
            "optional": {
                "video": ("IMAGE",),  # Option 1: Raw video/images
                "latents": ("LATENT",),  # Option 2: Latents (requires VAE too)
                "vae": ("VAE",),  # Required if using latents
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
                "extract_motion": ("BOOLEAN", {"default": True}),
                "extract_semantic": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = (VISUAL_FEATURES_TYPE,)
    RETURN_NAMES = ("visual_features",)
    FUNCTION = "extract"
    CATEGORY = "VersaSound/Visual"

    def __init__(self):
        self.device = get_device()
        self.model = None
        self.video_processor = VideoProcessor()
        self.latent_processor = LatentProcessor()

    def extract(
        self,
        model_name: str,
        video: Optional[torch.Tensor] = None,
        latents: Optional[Dict] = None,
        vae: Optional[Any] = None,
        fps: float = 24.0,
        extract_motion: bool = True,
        extract_semantic: bool = True
    ) -> Tuple[VisualFeatures]:
        """
        Extract visual features from video OR latents (not both!)

        WORKFLOW OPTIONS:
        1. Video workflow: Connect 'video' input with raw images
        2. Latent workflow: Connect 'latents' + 'vae' inputs

        Args:
            model_name: Model variant to use
            video: (Option 1) Video frames [N, H, W, C] from ComfyUI
            latents: (Option 2) ComfyUI latent dictionary
            vae: (Option 2) VAE for decoding latents
            fps: Frame rate of video
            extract_motion: Extract motion features
            extract_semantic: Extract semantic features

        Returns:
            VisualFeatures object
        """
        try:
            # Load model if needed
            if self.model is None:
                logger.info(f"Loading visual encoder: {model_name}")
                self.model = load_visual_encoder()
                self.model = self.model.to(self.device)
                self.model.eval()
                model_cache.set('visual_encoder', self.model)

            # Determine input type and process
            if video is not None:
                # Process video frames
                # ComfyUI format: [B, H, W, C] or [N, H, W, C] where N is number of frames
                logger.info(f"Input video shape: {video.shape}, dtype: {video.dtype}")

                if video.dim() == 4:
                    # ComfyUI image batch: [N, H, W, C]
                    # Convert to [B=1, T=N, C, H, W] for model
                    N, H, W, C = video.shape

                    if C != 3:
                        raise ValueError(f"Expected 3 channels (RGB), got {C}. "
                                       f"Input shape: {video.shape}")

                    # Permute to [N, C, H, W]
                    video_tensor = video.permute(0, 3, 1, 2)  # [N, C, H, W]

                    # Add batch dimension: [1, N, C, H, W]
                    video_tensor = video_tensor.unsqueeze(0)  # [1, N, C, H, W]

                    logger.info(f"Converted to model format: {video_tensor.shape}")

                elif video.dim() == 5:
                    # Already in video format [B, T, C, H, W] or [B, T, H, W, C]
                    if video.shape[-1] == 3:
                        # Channels last: [B, T, H, W, C] -> [B, T, C, H, W]
                        video_tensor = video.permute(0, 1, 4, 2, 3)
                    else:
                        video_tensor = video
                else:
                    raise ValueError(f"Unexpected video shape: {video.shape}. "
                                   f"Expected 4D [N, H, W, C] or 5D [B, T, H, W, C]")

            elif latents is not None:
                # Process latents
                video_tensor = self.latent_processor.latent_to_visual_features(
                    latents, vae, decode_stride=8
                )

            else:
                raise ValueError(
                    "No input provided! You must connect ONE of the following:\n"
                    "  Option 1: Connect 'video' input (raw images/video)\n"
                    "  Option 2: Connect 'latents' + 'vae' inputs (latent workflow)\n"
                    "Choose one workflow and connect the appropriate inputs."
                )

            # Move to device
            video_tensor = video_tensor.to(self.device)

            # Extract features
            logger.info(f"=== Starting Visual Feature Extraction ===")
            logger.info(f"Input: {video_tensor.shape} on {video_tensor.device}")

            with torch.no_grad():
                features_dict = self.model(
                    video_tensor,
                    return_intermediates=False
                )

            logger.info(f"=== Feature Extraction Complete ===")
            logger.info(f"Backbone features: {features_dict['backbone'].shape} (VideoMAE output)")
            logger.info(f"Motion features: {features_dict['motion'].shape} (Optical flow-based)")
            logger.info(f"Temporal features: {features_dict['temporal'].shape} (After transformer)")

            # Log audio cues if available
            if 'audio_cues' in features_dict and features_dict['audio_cues']:
                logger.info(f"Audio cues extracted:")
                for cue_name, cue_value in features_dict['audio_cues'].items():
                    if isinstance(cue_value, torch.Tensor):
                        logger.info(f"  - {cue_name}: {cue_value.shape}, range=[{cue_value.min():.3f}, {cue_value.max():.3f}]")
                    else:
                        logger.info(f"  - {cue_name}: {cue_value}")

            # Create VisualFeatures object
            visual_features = VisualFeatures(
                backbone_features=features_dict['backbone'].cpu(),
                motion_features=features_dict['motion'].cpu() if extract_motion else None,
                semantic_features=features_dict.get('semantic', None),
                temporal_features=features_dict['temporal'].cpu(),
                audio_cues=features_dict.get('audio_cues', {}),
                metadata={
                    'fps': fps,
                    'duration': features_dict['temporal'].shape[1] / fps,
                    'num_frames': features_dict['temporal'].shape[1],
                    'model_name': model_name,
                }
            )

            logger.info(f"Created VisualFeatures: duration={visual_features.metadata['duration']:.2f}s, fps={fps}, frames={visual_features.metadata['num_frames']}")
            logger.info(f"====================================")

            return (visual_features,)

        except Exception as e:
            logger.error(f"Error extracting visual features: {e}")
            raise


class AudioGenerator:
    """
    Generate audio from visual features
    Core audio generation node
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "visual_features": (VISUAL_FEATURES_TYPE,),
                "audio_model": (["audioldm2"],),  # Only AudioLDM2 is currently implemented
            },
            "optional": {
                "text_prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "duration_override": ("FLOAT", {"default": -1.0}),  # -1 = use video duration
                "sample_rate": ("INT", {"default": 16000, "min": 8000, "max": 48000, "step": 1000}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0}),
                "num_inference_steps": ("INT", {"default": 50, "min": 10, "max": 200}),
                "seed": ("INT", {"default": -1}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "VersaSound/Audio"

    def __init__(self):
        self.device = get_device()
        self.model = None

    def generate(
        self,
        visual_features: VisualFeatures,
        audio_model: str,
        text_prompt: str = "",
        negative_prompt: str = "",
        duration_override: float = -1.0,
        sample_rate: int = 48000,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        seed: int = -1
    ) -> Tuple[Dict]:
        """
        Generate audio from visual features

        Returns:
            Audio dictionary compatible with ComfyUI audio nodes
        """
        try:
            # Load model if needed
            if self.model is None:
                logger.info(f"Loading audio generator: {audio_model}")
                self.model = load_audio_generator()
                self.model = self.model.to(self.device)
                model_cache.set('audio_generator', self.model)

            # Set seed if provided
            if seed >= 0:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)

            # Determine duration
            duration = duration_override if duration_override > 0 else visual_features.metadata.get('duration', 5.0)

            # Move features to device
            visual_features = visual_features.to(self.device)

            # Convert VisualFeatures to dict for model
            features_dict = {
                'backbone': visual_features.backbone,
                'motion': visual_features.motion,
                'temporal': visual_features.temporal,
                'audio_cues': visual_features.audio_cues,
                'metadata': visual_features.metadata
            }

            # Generate audio
            with torch.no_grad():
                audio_waveform = self.model(
                    visual_features=features_dict,
                    text_prompt=text_prompt if text_prompt else None,
                    duration=duration,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    sample_rate=sample_rate
                )

            # Ensure stereo format [2, samples]
            if audio_waveform.dim() == 1:
                audio_waveform = audio_waveform.unsqueeze(0).repeat(2, 1)
            elif audio_waveform.dim() == 2 and audio_waveform.shape[0] == 1:
                audio_waveform = audio_waveform.repeat(2, 1)

            # Create audio dict compatible with ComfyUI
            audio_dict = {
                "waveform": audio_waveform.cpu().unsqueeze(0),  # [B, C, samples]
                "sample_rate": sample_rate
            }

            logger.info(f"Generated audio: {audio_waveform.shape}, {sample_rate}Hz, {duration:.2f}s")

            return (audio_dict,)

        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            raise


class TemporalSynchronizer:
    """
    Ensure audio-video temporal alignment
    Critical for lip-sync, impacts, rhythmic content
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "visual_features": (VISUAL_FEATURES_TYPE,),
                "audio": ("AUDIO",),
                "sync_mode": (["automatic", "lip_sync", "impact_sync", "rhythm_sync"],),
            },
            "optional": {
                "alignment_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
                "allow_time_stretch": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("AUDIO", SYNC_REPORT_TYPE)
    RETURN_NAMES = ("synchronized_audio", "sync_report")
    FUNCTION = "synchronize"
    CATEGORY = "VersaSound/Synchronization"

    def __init__(self):
        self.device = get_device()
        self.model = None

    def synchronize(
        self,
        visual_features: VisualFeatures,
        audio: Dict,
        sync_mode: str,
        alignment_strength: float = 1.0,
        allow_time_stretch: bool = True
    ) -> Tuple[Dict, SyncReport]:
        """
        Synchronize audio with visual events
        """
        try:
            # Load model if needed
            if self.model is None:
                logger.info("Loading temporal aligner")
                self.model = load_temporal_aligner()
                self.model = self.model.to(self.device)
                model_cache.set('temporal_aligner', self.model)

            # Extract audio waveform
            waveform = audio["waveform"]  # [B, C, samples]
            sample_rate = audio["sample_rate"]

            # Convert to [B, samples]
            if waveform.dim() == 3:
                waveform = waveform.mean(dim=1)  # Average channels

            # Move to device
            waveform = waveform.to(self.device)
            visual_features = visual_features.to(self.device)

            # Convert features to dict
            features_dict = {
                'temporal': visual_features.temporal,
                'motion': visual_features.motion,
                'audio_cues': visual_features.audio_cues
            }

            # Perform synchronization
            with torch.no_grad():
                aligned_waveform, sync_report = self.model(
                    visual_features=features_dict,
                    audio=waveform,
                    sample_rate=sample_rate
                )

            # Convert back to ComfyUI audio format
            aligned_waveform = aligned_waveform.cpu()

            # Ensure stereo
            if aligned_waveform.dim() == 2 and aligned_waveform.shape[0] == 1:
                aligned_waveform = aligned_waveform.repeat(1, 2, 1)  # [B, 2, samples]
            elif aligned_waveform.dim() == 2:
                aligned_waveform = aligned_waveform.unsqueeze(1).repeat(1, 2, 1)

            aligned_audio = {
                "waveform": aligned_waveform,
                "sample_rate": sample_rate
            }

            logger.info(f"Synchronized audio: {sync_report}")

            return (aligned_audio, sync_report)

        except Exception as e:
            logger.error(f"Error synchronizing audio: {e}")
            # Return original audio with dummy report
            dummy_report = SyncReport(0.5, 0.1, 0, False, {})
            return (audio, dummy_report)


class AudioRefiner:
    """
    Post-processing and enhancement
    Fixes common issues, improves quality
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
            "optional": {
                "remove_noise": ("BOOLEAN", {"default": True}),
                "normalize": ("BOOLEAN", {"default": True}),
                "enhance_clarity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "add_reverb": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "eq_preset": (["none", "voice", "music", "sfx", "cinematic"],),
                "target_lufs": ("FLOAT", {"default": -16.0, "min": -30.0, "max": 0.0}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("refined_audio",)
    FUNCTION = "refine"
    CATEGORY = "VersaSound/Processing"

    def __init__(self):
        self.audio_processor = AudioProcessor()

    def refine(
        self,
        audio: Dict,
        remove_noise: bool = True,
        normalize: bool = True,
        enhance_clarity: float = 0.5,
        add_reverb: float = 0.0,
        eq_preset: str = "none",
        target_lufs: float = -16.0
    ) -> Tuple[Dict]:
        """
        Refine and enhance audio
        """
        try:
            waveform = audio["waveform"]  # [B, C, samples]
            sample_rate = audio["sample_rate"]

            # Convert to [C, samples] for processing
            if waveform.dim() == 3:
                waveform = waveform.squeeze(0)

            # Apply effects
            effects = {}

            if add_reverb > 0:
                effects['reverb'] = {'amount': add_reverb}

            if eq_preset != "none":
                effects['eq'] = {'preset': eq_preset}

            if remove_noise:
                effects['noise_reduction'] = {'amount': 0.5}

            if effects:
                waveform = self.audio_processor.apply_effects(waveform, effects)

            # Normalize
            if normalize:
                waveform = self.audio_processor.normalize(
                    waveform,
                    method='peak',
                    target_level=target_lufs / 10.0  # Approximate conversion
                )

            # Convert back to ComfyUI format
            refined_audio = {
                "waveform": waveform.unsqueeze(0),  # [B, C, samples]
                "sample_rate": sample_rate
            }

            logger.info("Refined audio")

            return (refined_audio,)

        except Exception as e:
            logger.error(f"Error refining audio: {e}")
            return (audio,)  # Return original on error


class LatentToVisualFeatures:
    """
    CRITICAL: Convert ComfyUI latents directly to visual features
    Bypass VAE decoding when possible for efficiency
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
            },
            "optional": {
                "vae": ("VAE",),
                "decode_samples": ("INT", {"default": 8}),  # Decode every Nth frame
                "fps": ("FLOAT", {"default": 24.0}),
            }
        }

    RETURN_TYPES = (VISUAL_FEATURES_TYPE,)
    RETURN_NAMES = ("visual_features",)
    FUNCTION = "convert"
    CATEGORY = "VersaSound/Visual"

    def __init__(self):
        self.device = get_device()
        self.latent_processor = LatentProcessor()
        self.model = None

    def convert(
        self,
        latent: Dict,
        vae: Optional[Any] = None,
        decode_samples: int = 8,
        fps: float = 24.0
    ) -> Tuple[VisualFeatures]:
        """
        Convert latent to visual features
        """
        try:
            # Load model if needed
            if self.model is None:
                self.model = load_visual_encoder()
                self.model = self.model.to(self.device)

            # Process latent
            video_tensor = self.latent_processor.latent_to_visual_features(
                latent, vae, decode_stride=decode_samples
            )

            video_tensor = video_tensor.to(self.device)

            # Extract features
            with torch.no_grad():
                features_dict = self.model(video_tensor)

            # Create VisualFeatures
            visual_features = VisualFeatures(
                backbone_features=features_dict['backbone'].cpu(),
                motion_features=features_dict['motion'].cpu(),
                temporal_features=features_dict['temporal'].cpu(),
                audio_cues=features_dict.get('audio_cues', {}),
                metadata={
                    'fps': fps,
                    'from_latent': True,
                }
            )

            return (visual_features,)

        except Exception as e:
            logger.error(f"Error converting latent: {e}")
            raise


class AudioBlender:
    """
    Mix multiple audio sources (layering)
    E.g., speech + background music + ambient sound
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_1": ("AUDIO",),
            },
            "optional": {
                "audio_2": ("AUDIO",),
                "audio_3": ("AUDIO",),
                "audio_4": ("AUDIO",),
                "weights": ("STRING", {"default": "1.0,0.5,0.3,0.2"}),
                "normalization": (["none", "peak", "rms"],),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("blended_audio",)
    FUNCTION = "blend"
    CATEGORY = "VersaSound/Processing"

    def blend(
        self,
        audio_1: Dict,
        audio_2: Optional[Dict] = None,
        audio_3: Optional[Dict] = None,
        audio_4: Optional[Dict] = None,
        weights: str = "1.0,0.5,0.3,0.2",
        normalization: str = "peak"
    ) -> Tuple[Dict]:
        """
        Blend multiple audio tracks
        """
        try:
            # Collect audio tracks
            audio_list = [audio_1]
            if audio_2 is not None:
                audio_list.append(audio_2)
            if audio_3 is not None:
                audio_list.append(audio_3)
            if audio_4 is not None:
                audio_list.append(audio_4)

            # Parse weights
            weight_list = [float(w.strip()) for w in weights.split(',')]
            weight_list = weight_list[:len(audio_list)]

            # Normalize weights
            weight_sum = sum(weight_list)
            weight_list = [w / weight_sum for w in weight_list]

            # Extract waveforms
            waveforms = []
            sample_rate = audio_list[0]["sample_rate"]

            for audio in audio_list:
                waveform = audio["waveform"]  # [B, C, samples]
                if waveform.dim() == 3:
                    waveform = waveform.squeeze(0)  # [C, samples]
                waveforms.append(waveform)

            # Blend
            blended = blend_audio(waveforms, weight_list, method='weighted_sum')

            # Normalize if requested
            if normalization != "none":
                processor = AudioProcessor()
                blended = processor.normalize(blended, method=normalization)

            # Convert to ComfyUI format
            blended_audio = {
                "waveform": blended.unsqueeze(0),  # [B, C, samples]
                "sample_rate": sample_rate
            }

            logger.info(f"Blended {len(audio_list)} audio tracks")

            return (blended_audio,)

        except Exception as e:
            logger.error(f"Error blending audio: {e}")
            return (audio_1,)


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "VersaSound_VisualFeatureExtractor": VisualFeatureExtractor,
    "VersaSound_AudioGenerator": AudioGenerator,
    "VersaSound_TemporalSynchronizer": TemporalSynchronizer,
    "VersaSound_AudioRefiner": AudioRefiner,
    "VersaSound_LatentToVisualFeatures": LatentToVisualFeatures,
    "VersaSound_AudioBlender": AudioBlender,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VersaSound_VisualFeatureExtractor": "Visual Feature Extractor",
    "VersaSound_AudioGenerator": "Audio Generator",
    "VersaSound_TemporalSynchronizer": "Temporal Synchronizer",
    "VersaSound_AudioRefiner": "Audio Refiner",
    "VersaSound_LatentToVisualFeatures": "Latent to Visual Features",
    "VersaSound_AudioBlender": "Audio Blender",
}
