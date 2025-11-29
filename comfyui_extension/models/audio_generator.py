"""
Universal Audio Generator for VersaSound
Generates audio conditioned on visual features
Works with pretrained audio diffusion models
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Optional, Tuple
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class UniversalAudioGenerator(nn.Module):
    """
    Generates audio conditioned on visual features
    Adapts visual features to pretrained audio models
    Supports multiple audio generation backends
    """

    def __init__(self, config: Dict):
        """
        Initialize Audio Generator

        Args:
            config: Configuration dictionary
        """
        super().__init__()

        self.config = config
        self.audio_model_name = config.get('audio_model_name', 'audioldm2')
        self.visual_dim = config.get('visual_dim', 768)
        self.use_lora = config.get('use_lora', False)

        # Load pretrained audio model
        self.audio_model, self.audio_latent_dim = self._load_audio_model()

        # Freeze audio model initially
        self._freeze_audio_model()

        # Visual to audio adapter
        adapter_layers = config.get('adapter_layers', 4)
        self.visual_adapter = VisualToAudioAdapter(
            visual_dim=self.visual_dim,
            audio_latent_dim=self.audio_latent_dim,
            num_layers=adapter_layers
        )

        # Temporal controller
        temporal_config = config.get('temporal_controller', {})
        if temporal_config.get('enabled', True):
            self.temporal_controller = TemporalController(
                visual_dim=self.visual_dim,
                hidden_dim=temporal_config.get('hidden_dim', 256)
            )
        else:
            self.temporal_controller = None

        # LoRA adapters (optional)
        if self.use_lora:
            self._add_lora_layers(config.get('lora_rank', 16))

        logger.info(f"Initialized UniversalAudioGenerator with {self.audio_model_name}")

    def _load_audio_model(self) -> Tuple[nn.Module, int]:
        """
        Load pretrained audio generation model

        Returns:
            Tuple of (model, latent_dimension)
        """
        model_name = self.audio_model_name.lower()

        if 'audioldm2' in model_name:
            return self._load_audioldm2()
        elif 'stable_audio' in model_name:
            return self._load_stable_audio()
        elif 'audiocraft' in model_name or 'audiogen' in model_name:
            return self._load_audiocraft()
        else:
            logger.warning(f"Unknown audio model: {model_name}, using dummy model")
            return self._create_dummy_model()

    def _load_audioldm2(self) -> Tuple[nn.Module, int]:
        """Load AudioLDM2 model"""
        try:
            from diffusers import AudioLDM2Pipeline

            model = AudioLDM2Pipeline.from_pretrained(
                "cvssp/audioldm2",
                torch_dtype=torch.float32
            )

            # Get latent dimension
            latent_dim = 768  # AudioLDM2 uses CLAP embeddings

            logger.info("Loaded AudioLDM2 model")
            return model, latent_dim

        except ImportError:
            logger.error("diffusers library required for AudioLDM2")
            raise
        except Exception as e:
            logger.warning(f"Error loading AudioLDM2: {e}, using dummy model")
            return self._create_dummy_model()

    def _load_stable_audio(self) -> Tuple[nn.Module, int]:
        """Load Stable Audio model"""
        try:
            from stable_audio_tools import get_pretrained_model

            model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
            latent_dim = 768

            logger.info("Loaded Stable Audio model")
            return model, latent_dim

        except ImportError:
            logger.warning("stable-audio-tools not available")
            return self._create_dummy_model()
        except Exception as e:
            logger.warning(f"Error loading Stable Audio: {e}")
            return self._create_dummy_model()

    def _load_audiocraft(self) -> Tuple[nn.Module, int]:
        """Load AudioCraft/AudioGen model"""
        try:
            from audiocraft.models import AudioGen

            model = AudioGen.get_pretrained("facebook/audiogen-medium")
            latent_dim = 768

            logger.info("Loaded AudioCraft model")
            return model, latent_dim

        except ImportError:
            logger.warning("audiocraft not available")
            return self._create_dummy_model()
        except Exception as e:
            logger.warning(f"Error loading AudioCraft: {e}")
            return self._create_dummy_model()

    def _create_dummy_model(self) -> Tuple[nn.Module, int]:
        """Create a dummy audio model for testing"""
        logger.warning("Using dummy audio model - audio quality will be poor")

        class DummyAudioModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.latent_dim = 768
                self.decoder = nn.Sequential(
                    nn.Linear(768, 2048),
                    nn.ReLU(),
                    nn.Linear(2048, 48000)  # 1 second at 48kHz
                )

            def forward(self, conditioning, duration=1.0):
                # Simple dummy generation
                batch_size = conditioning.shape[0]
                num_samples = int(duration * 48000)

                # Generate from conditioning mean
                cond_mean = conditioning.mean(dim=1)  # [B, D]
                audio = self.decoder(cond_mean)  # [B, 48000]

                # Tile to desired length
                if num_samples > 48000:
                    reps = (num_samples // 48000) + 1
                    audio = audio.repeat(1, reps)[:, :num_samples]

                return audio

        model = DummyAudioModel()
        return model, 768

    def _freeze_audio_model(self):
        """Freeze pretrained audio model parameters"""
        # Check if it's a diffusers Pipeline (which doesn't have .parameters() method)
        if hasattr(self.audio_model, 'components'):
            # It's a diffusers Pipeline - freeze its components
            for name, component in self.audio_model.components.items():
                if component is not None and isinstance(component, nn.Module):
                    for param in component.parameters():
                        param.requires_grad = False
            logger.debug("Froze audio model pipeline components")
        elif hasattr(self.audio_model, 'parameters'):
            # It's a regular nn.Module
            for param in self.audio_model.parameters():
                param.requires_grad = False
            logger.debug("Froze audio model parameters")
        else:
            logger.warning("Audio model has no parameters to freeze")

    def _add_lora_layers(self, rank: int):
        """Add LoRA adapters for fine-tuning"""
        try:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=rank,
                lora_alpha=rank * 2,
                target_modules=["to_q", "to_k", "to_v", "to_out"],
                lora_dropout=0.1,
                bias="none"
            )

            self.audio_model = get_peft_model(self.audio_model, lora_config)
            logger.info(f"Added LoRA adapters with rank {rank}")

        except ImportError:
            logger.warning("peft library not available, skipping LoRA")
        except Exception as e:
            logger.warning(f"Could not add LoRA: {e}")

    @torch.no_grad()
    def forward(
        self,
        visual_features: Dict[str, torch.Tensor],
        text_prompt: Optional[str] = None,
        duration: Optional[float] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        sample_rate: int = 48000,
        return_intermediates: bool = False
    ) -> torch.Tensor:
        """
        Generate audio from visual features

        Args:
            visual_features: Dict from VisualEncoder
            text_prompt: Optional text conditioning
            duration: Audio duration in seconds (None = use video duration)
            num_inference_steps: Number of diffusion steps
            guidance_scale: Classifier-free guidance scale
            sample_rate: Output sample rate
            return_intermediates: Return intermediate outputs

        Returns:
            Generated audio waveform [B, num_samples]
        """
        # Get metadata
        metadata = visual_features.get('metadata', {})
        if duration is None:
            duration = metadata.get('duration', 5.0)

        # Adapt visual features to audio conditioning
        audio_conditioning = self.visual_adapter(visual_features)
        # audio_conditioning: [B, T, latent_dim]

        # Get temporal control signals
        if self.temporal_controller is not None:
            temporal_controls = self.temporal_controller(visual_features)
        else:
            temporal_controls = None

        # Generate audio based on model type
        if 'audioldm2' in self.audio_model_name.lower():
            audio = self._generate_with_audioldm2(
                audio_conditioning,
                text_prompt,
                duration,
                num_inference_steps,
                guidance_scale,
                temporal_controls
            )

        elif 'stable_audio' in self.audio_model_name.lower():
            audio = self._generate_with_stable_audio(
                audio_conditioning,
                text_prompt,
                duration,
                temporal_controls
            )

        elif 'audiocraft' in self.audio_model_name.lower():
            audio = self._generate_with_audiocraft(
                audio_conditioning,
                text_prompt,
                duration,
                temporal_controls
            )

        else:
            # Dummy model
            audio = self.audio_model(audio_conditioning, duration)

        # Ensure correct sample rate
        if audio.shape[-1] != int(duration * sample_rate):
            # Resample if needed
            audio = self._resample_audio(audio, duration, sample_rate)

        return audio

    def _generate_with_audioldm2(
        self,
        conditioning: torch.Tensor,
        text_prompt: Optional[str],
        duration: float,
        num_steps: int,
        guidance_scale: float,
        temporal_controls: Optional[Dict]
    ) -> torch.Tensor:
        """Generate audio using AudioLDM2"""
        try:
            batch_size = conditioning.shape[0]

            # Prepare prompt
            if text_prompt is None:
                text_prompt = "high quality audio"

            prompts = [text_prompt] * batch_size

            # Generate
            audio_arrays = self.audio_model(
                prompt=prompts,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                audio_length_in_s=duration,
                num_waveforms_per_prompt=1
            ).audios

            # Convert to tensor
            audio = torch.from_numpy(audio_arrays).float()

            # Reshape to [B, samples]
            if audio.dim() == 3:
                audio = audio.squeeze(1)

            return audio

        except Exception as e:
            logger.error(f"Error generating with AudioLDM2: {e}")
            # Fallback to dummy generation
            return self._dummy_generate(conditioning, duration)

    def _generate_with_stable_audio(
        self,
        conditioning: torch.Tensor,
        text_prompt: Optional[str],
        duration: float,
        temporal_controls: Optional[Dict]
    ) -> torch.Tensor:
        """Generate audio using Stable Audio"""
        # Placeholder implementation
        logger.warning("Stable Audio generation not fully implemented")
        return self._dummy_generate(conditioning, duration)

    def _generate_with_audiocraft(
        self,
        conditioning: torch.Tensor,
        text_prompt: Optional[str],
        duration: float,
        temporal_controls: Optional[Dict]
    ) -> torch.Tensor:
        """Generate audio using AudioCraft"""
        # Placeholder implementation
        logger.warning("AudioCraft generation not fully implemented")
        return self._dummy_generate(conditioning, duration)

    def _dummy_generate(
        self,
        conditioning: torch.Tensor,
        duration: float
    ) -> torch.Tensor:
        """Dummy audio generation for testing"""
        batch_size = conditioning.shape[0]
        num_samples = int(duration * 48000)

        # Generate simple sine wave based on conditioning
        t = torch.linspace(0, duration, num_samples).to(conditioning.device)
        freq = 440.0 + conditioning.mean() * 100  # Vary frequency based on conditioning

        audio = torch.sin(2 * torch.pi * freq * t)
        audio = audio.unsqueeze(0).expand(batch_size, -1)

        return audio

    def _resample_audio(
        self,
        audio: torch.Tensor,
        duration: float,
        target_sr: int
    ) -> torch.Tensor:
        """Resample audio to target sample rate"""
        target_length = int(duration * target_sr)

        if audio.shape[-1] == target_length:
            return audio

        # Simple linear interpolation
        audio_resampled = torch.nn.functional.interpolate(
            audio.unsqueeze(1),
            size=target_length,
            mode='linear',
            align_corners=False
        ).squeeze(1)

        return audio_resampled


class VisualToAudioAdapter(nn.Module):
    """
    Adapts visual features to audio model conditioning
    Lightweight cross-modal projection layer
    """

    def __init__(
        self,
        visual_dim: int,
        audio_latent_dim: int,
        num_layers: int = 4
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(visual_dim, audio_latent_dim)

        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=audio_latent_dim,
                nhead=8,
                dim_feedforward=audio_latent_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        # Output normalization
        self.output_norm = nn.LayerNorm(audio_latent_dim)

    def forward(self, visual_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Project visual features to audio conditioning space

        Args:
            visual_features: Dict with 'temporal', 'backbone', 'motion' keys

        Returns:
            Audio conditioning [B, T, audio_latent_dim]
        """
        # Combine different visual features
        # Prioritize temporal features, but include motion and audio cues
        temporal = visual_features['temporal']
        motion = visual_features.get('motion', None)
        audio_cues = visual_features.get('audio_cues', {})

        # If motion features available, incorporate them
        if motion is not None:
            # Combine via concatenation and projection
            combined = torch.cat([temporal, motion], dim=-1)
            # Project back to visual_dim
            combined = nn.functional.linear(
                combined,
                weight=torch.nn.Parameter(torch.randn(temporal.shape[-1], combined.shape[-1]))
            )
        else:
            combined = temporal

        # Project to audio latent space
        x = self.input_proj(combined)

        # Refine with cross-attention layers
        for layer in self.cross_attention_layers:
            x = layer(x, x)

        # Normalize output
        x = self.output_norm(x)

        return x


class TemporalController(nn.Module):
    """
    Generates temporal control signals for audio synthesis
    Ensures synchronization with visual events
    """

    def __init__(
        self,
        visual_dim: int,
        hidden_dim: int = 256
    ):
        super().__init__()

        # Intensity envelope predictor
        self.intensity_predictor = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Event timing predictor
        self.event_predictor = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Rhythm predictor
        self.rhythm_predictor = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )

    def forward(self, visual_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extract temporal control signals

        Args:
            visual_features: Visual features dict

        Returns:
            Dict of control signals
        """
        temporal = visual_features['temporal']  # [B, T, D]

        intensity = self.intensity_predictor(temporal).squeeze(-1)  # [B, T]
        events = self.event_predictor(temporal).squeeze(-1)  # [B, T]
        rhythm = self.rhythm_predictor(temporal).squeeze(-1)  # [B, T]

        return {
            'intensity_envelope': intensity,
            'event_timings': events,
            'rhythm': rhythm
        }


def load_audio_generator(
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None
) -> UniversalAudioGenerator:
    """
    Load audio generator from config and checkpoint

    Args:
        config_path: Path to config YAML
        checkpoint_path: Path to checkpoint
        device: Device to load on

    Returns:
        Loaded UniversalAudioGenerator
    """
    # Load config
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "default_config.yaml"

    with open(config_path) as f:
        full_config = yaml.safe_load(f)
        config = full_config.get('audio_generator', {})

    # Create model
    model = UniversalAudioGenerator(config)

    # Load checkpoint
    if checkpoint_path is not None:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

    # Move to device
    if device is not None:
        model = model.to(device)

    model.eval()
    return model
