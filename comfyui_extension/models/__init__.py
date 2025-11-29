"""
Model modules for VersaSound ComfyUI Extension
"""

from .visual_encoder import (
    UniversalVisualEncoder,
    MotionEncoder,
    AudioCueExtractor,
    load_visual_encoder
)

from .audio_generator import (
    UniversalAudioGenerator,
    VisualToAudioAdapter,
    TemporalController,
    load_audio_generator
)

from .temporal_aligner import (
    TemporalAlignmentModule,
    EventDetector,
    AlignmentPredictor,
    AudioWarper,
    load_temporal_aligner
)

__all__ = [
    # Visual Encoder
    'UniversalVisualEncoder',
    'MotionEncoder',
    'AudioCueExtractor',
    'load_visual_encoder',

    # Audio Generator
    'UniversalAudioGenerator',
    'VisualToAudioAdapter',
    'TemporalController',
    'load_audio_generator',

    # Temporal Aligner
    'TemporalAlignmentModule',
    'EventDetector',
    'AlignmentPredictor',
    'AudioWarper',
    'load_temporal_aligner',
]
