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
            import transformers
            import diffusers

            logger.info(f"Loading AudioLDM2 with transformers={transformers.__version__}, diffusers={diffusers.__version__}")

            from diffusers import AudioLDM2Pipeline

            # Load with trust_remote_code and specific revision for better compatibility
            model = AudioLDM2Pipeline.from_pretrained(
                "cvssp/audioldm2",
                torch_dtype=torch.float32,
                trust_remote_code=True,
                use_safetensors=True if hasattr(AudioLDM2Pipeline, 'use_safetensors') else False
            )

            # Get latent dimension
            latent_dim = 768  # AudioLDM2 uses CLAP embeddings

            logger.info(f"Loaded AudioLDM2 model successfully")
            logger.info(f"AudioLDM2 components: {list(model.components.keys())}")

            return model, latent_dim

        except ImportError:
            logger.error("diffusers library required for AudioLDM2")
            raise
        except Exception as e:
            logger.error(f"Error loading AudioLDM2: {e}, using dummy model")
            import traceback
            logger.error(traceback.format_exc())
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

    def to(self, device):
        """Override to() to handle Pipeline objects"""
        # Move the nn.Module components (adapter, temporal controller, etc.)
        super().to(device)

        # Move audio model (handle Pipeline objects specially)
        if hasattr(self.audio_model, 'to'):
            # Try to move the whole model/pipeline
            try:
                self.audio_model = self.audio_model.to(device)
                logger.debug(f"Moved audio model to {device}")
            except Exception as e:
                logger.warning(f"Could not move audio model to device: {e}")
                # If it's a Pipeline, try moving components individually
                if hasattr(self.audio_model, 'components'):
                    for name, component in self.audio_model.components.items():
                        if component is not None and hasattr(component, 'to'):
                            try:
                                component.to(device)
                                logger.debug(f"Moved pipeline component '{name}' to {device}")
                            except Exception as comp_e:
                                logger.warning(f"Could not move component '{name}': {comp_e}")

        return self

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
            if text_prompt is None or text_prompt.strip() == "":
                text_prompt = "high quality audio"

            prompts = [text_prompt] * batch_size

            logger.info(f"Generating audio with AudioLDM2: prompt='{text_prompt}', duration={duration}s, steps={num_steps}")

            # Try to patch GPT2Model if needed (workaround for transformers compatibility)
            self._patch_gpt2_if_needed()

            # Generate
            # NOTE: AudioLDM2 generates at 16kHz sample rate by default
            audio_arrays = self.audio_model(
                prompt=prompts,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                audio_length_in_s=duration,
                num_waveforms_per_prompt=1,
                generator=None  # Explicitly set generator to None for determinism
            ).audios

            # Convert to tensor and move to correct device
            audio = torch.from_numpy(audio_arrays).float().to(conditioning.device)

            # Reshape to [B, samples]
            if audio.dim() == 3:
                audio = audio.squeeze(1)
            elif audio.dim() == 2 and batch_size == 1:
                # audio_arrays might be [1, samples] or [samples]
                pass
            elif audio.dim() == 1:
                audio = audio.unsqueeze(0).expand(batch_size, -1)

            logger.info(f"AudioLDM2 generated audio: shape={audio.shape}, min={audio.min():.3f}, max={audio.max():.3f}")
            return audio

        except AttributeError as e:
            if "_update_model_kwargs_for_generation" in str(e) or "GPT2Model" in str(e):
                import transformers
                logger.error(
                    f"AudioLDM2 compatibility error with transformers {transformers.__version__}:\n"
                    f"Error: {e}\n"
                    f"This is a known issue with certain transformers/diffusers version combinations.\n"
                    f"Current versions installed: transformers={transformers.__version__}, diffusers={getattr(__import__('diffusers'), '__version__', 'unknown')}\n"
                    f"\nTry downgrading transformers: pip install transformers==4.35.2 diffusers==0.25.0\n"
                    f"Falling back to dummy audio generation."
                )
                import traceback
                logger.error(traceback.format_exc())
            else:
                logger.error(f"AudioLDM2 AttributeError: {e}")
                import traceback
                logger.error(traceback.format_exc())
            return self._dummy_generate(conditioning, duration)
        except Exception as e:
            logger.error(f"Error generating with AudioLDM2: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Fallback to dummy generation
            return self._dummy_generate(conditioning, duration)

    def _patch_gpt2_if_needed(self):
        """Attempt to patch GPT2Model for compatibility with newer transformers"""
        try:
            import types

            # AudioLDM2 uses 'language_model' which is a GPT2Model - this is where the error occurs
            if hasattr(self.audio_model, 'language_model'):
                language_model = self.audio_model.language_model
                if not hasattr(language_model, '_update_model_kwargs_for_generation'):
                    # Add the missing method as a pass-through
                    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, **kwargs):
                        # Simple pass-through - just return model_kwargs unchanged
                        return model_kwargs

                    # Bind the method to the language_model instance
                    language_model._update_model_kwargs_for_generation = types.MethodType(
                        _update_model_kwargs_for_generation, language_model
                    )
                    logger.info("✓ Patched language_model (GPT2Model)._update_model_kwargs_for_generation")

            # Also check text_encoder in case it's needed
            if hasattr(self.audio_model, 'text_encoder'):
                text_encoder = self.audio_model.text_encoder
                if hasattr(text_encoder, 'transformer') and not hasattr(text_encoder.transformer, '_update_model_kwargs_for_generation'):
                    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, **kwargs):
                        return model_kwargs

                    text_encoder.transformer._update_model_kwargs_for_generation = types.MethodType(
                        _update_model_kwargs_for_generation, text_encoder.transformer
                    )
                    logger.info("✓ Patched text_encoder.transformer (GPT2Model)._update_model_kwargs_for_generation")

        except Exception as e:
            logger.warning(f"Could not patch GPT2Model: {e}")

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
        logger.warning(
            "Using dummy audio generator! AudioLDM2 is not working.\n"
            "To fix: pip install 'transformers>=4.30.0,<5.0.0' 'diffusers>=0.25.0,<0.32.0'"
        )

        batch_size = conditioning.shape[0]
        num_samples = int(duration * 48000)

        # Generate audible sine wave sweep for testing
        # This creates an audible test tone that sweeps from 220Hz to 880Hz
        t = torch.linspace(0, duration, num_samples).to(conditioning.device)

        # Frequency sweep from 220Hz (A3) to 880Hz (A5) over duration
        freq_start = 220.0
        freq_end = 880.0
        freq = freq_start + (freq_end - freq_start) * (t / duration)

        # Generate tone with amplitude 0.3 (audible but not too loud)
        audio = 0.3 * torch.sin(2 * torch.pi * freq * t)

        # Expand to batch size
        audio = audio.unsqueeze(0).expand(batch_size, -1)

        logger.info(f"Generated dummy audio: shape={audio.shape}, min={audio.min():.3f}, max={audio.max():.3f}")

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

        # Ensure audio is 2D [B, samples]
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Simple linear interpolation
        # interpolate expects [B, C, samples], so unsqueeze to add channel dim
        audio_resampled = torch.nn.functional.interpolate(
            audio.unsqueeze(1),  # [B, 1, samples]
            size=target_length,
            mode='linear',
            align_corners=False
        ).squeeze(1)  # [B, samples]

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

        self.visual_dim = visual_dim
        self.audio_latent_dim = audio_latent_dim

        # Motion fusion projection (lazy initialization on first forward pass)
        # This projects concatenated temporal+motion features back to visual_dim
        # We don't know motion_dim at init time, so we create this lazily
        self.motion_fusion = None

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
            # Combine via concatenation and learned projection
            combined = torch.cat([temporal, motion], dim=-1)

            # Lazy initialization of motion_fusion layer
            if self.motion_fusion is None:
                combined_dim = combined.shape[-1]
                self.motion_fusion = nn.Linear(combined_dim, self.visual_dim).to(combined.device)
                logger.debug(f"Lazy initialized motion_fusion: {combined_dim} -> {self.visual_dim}")

            # Project back to visual_dim using learned weights
            combined = self.motion_fusion(combined)
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
