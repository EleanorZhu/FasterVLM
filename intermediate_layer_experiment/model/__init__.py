"""
Intermediate Layer Experiment Model Module

This module contains the modified LLaVA model that extracts intermediate layer embeddings
and feeds them back as input to the LLM.
"""

from .llava_intermediate_layer import (
    LlavaIntermediateLayerForCausalLM,
    LlavaIntermediateLayerConfig,
)

__all__ = [
    'LlavaIntermediateLayerForCausalLM',
    'LlavaIntermediateLayerConfig',
]

# Fixed train/inference consistency version
from .llava_intermediate_layer_fixed import (
    LlavaIntermediateLayerForCausalLMFixed,
    LlavaIntermediateLayerConfigFixed,
)

__all__.extend([
    'LlavaIntermediateLayerForCausalLMFixed',
    'LlavaIntermediateLayerConfigFixed',
])
