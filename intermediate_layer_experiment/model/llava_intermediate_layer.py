#    Modified LLaVA model with intermediate layer extraction and feedback
#
#    Based on LLaVA (Copyright 2023 Haotian Liu)
#    Licensed under the Apache License, Version 2.0

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from llava.model.language_model.llava_llama import (
    LlavaLlamaForCausalLM,
    LlavaConfig,
    LlavaLlamaModel,
)
from llava.constants import IMAGE_TOKEN_INDEX


class LlavaIntermediateLayerConfig(LlavaConfig):
    """
    Configuration class for LlavaIntermediateLayerForCausalLM.
    
    Adds configuration for intermediate layer extraction.
    """
    model_type = "llava_intermediate_layer"
    
    def __init__(
        self,
        intermediate_layer_idx: int = 3,
        use_intermediate_feedback: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.intermediate_layer_idx = intermediate_layer_idx
        self.use_intermediate_feedback = use_intermediate_feedback


class LlavaIntermediateLayerForCausalLM(LlavaLlamaForCausalLM):
    """
    Modified LLaVA model that extracts the complete sequence from an intermediate layer
    and feeds it back as input to the LLM.

    Architecture flow:
    1. Image → Vision Encoder → Projector → Initial embeddings (Layer 0)
    2. First pass: LLM processes through layers 0-N, extract complete layer N hidden states
       - Extracts: [text_prefix_L3] + [vision_L3] + [text_suffix_L3]
    3. Second pass: Feed entire layer N sequence back to Layer 0, process through all layers
       - Both vision and text embeddings are now at the same semantic level (Layer N)
    4. Generate final output

    This approach avoids the semantic mismatch problem where vision embeddings from Layer N
    would be concatenated with text embeddings from Layer 0.
    """
    config_class = LlavaIntermediateLayerConfig

    def __init__(self, config, visual_token_num=None):
        super().__init__(config, visual_token_num)

        # Configuration for intermediate layer extraction
        self.intermediate_layer_idx = getattr(config, 'intermediate_layer_idx', 3)
        self.use_intermediate_feedback = getattr(config, 'use_intermediate_feedback', True)

        print(f"[IntermediateLayer] Initialized with layer_idx={self.intermediate_layer_idx}, "
              f"feedback={self.use_intermediate_feedback}")

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        """
        Override to track visual token positions.
        """
        # Call parent method
        result = super().prepare_inputs_labels_for_multimodal(
            input_ids, position_ids, attention_mask, past_key_values, labels,
            images, image_sizes
        )

        # Unpack result
        if len(result) == 8:
            input_ids_out, position_ids_out, attention_mask_out, past_key_values_out, \
                inputs_embeds, labels_out, v_token_num, cls_attn = result
        else:
            # Fallback for different return format
            return result

        # Track visual token positions
        # Find where IMAGE_TOKEN_INDEX was in the original input_ids
        if input_ids is not None and inputs_embeds is not None:
            # Find the position of visual tokens in the embedded sequence
            # In LLaVA, visual tokens replace IMAGE_TOKEN_INDEX
            # We need to find where they are in the final sequence

            # Simple approach: count tokens before IMAGE_TOKEN_INDEX
            batch_size = inputs_embeds.shape[0]
            if batch_size == 1:  # Single sample (typical for inference)
                # Find IMAGE_TOKEN_INDEX position in original input_ids
                original_input_ids = input_ids[0] if input_ids.dim() > 1 else input_ids

                # Count tokens before the image token
                # IMAGE_TOKEN_INDEX = -200
                image_token_mask = (original_input_ids == IMAGE_TOKEN_INDEX)
                if image_token_mask.any():
                    image_token_pos = image_token_mask.nonzero(as_tuple=True)[0][0].item()

                    # Visual tokens start at this position in the embedded sequence
                    # and span v_token_num tokens
                    start_idx = image_token_pos
                    end_idx = start_idx + v_token_num

                    self._current_visual_token_positions = (start_idx, end_idx)
                    self._current_v_token_num = v_token_num
                else:
                    # No image token found, use default
                    self._current_visual_token_positions = (0, v_token_num)
                    self._current_v_token_num = v_token_num
            else:
                # For batch processing, use a simple heuristic
                self._current_visual_token_positions = (20, 20 + v_token_num)
                self._current_v_token_num = v_token_num

        return result

    def _find_visual_token_positions(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
    ) -> Tuple[int, int]:
        """
        Find the positions of visual tokens in the input sequence.

        NOTE: This method is kept for debugging and logging purposes.
        In the new implementation, we use the entire intermediate sequence,
        so we don't need to extract visual tokens separately.

        In LLaVA, visual tokens are inserted at the position of IMAGE_TOKEN_INDEX.
        We need to find where they are in the final embedded sequence.

        Args:
            input_ids: Original input IDs (may be None if using inputs_embeds)
            inputs_embeds: Input embeddings [batch_size, seq_len, hidden_dim]

        Returns:
            Tuple of (start_idx, end_idx) for visual tokens
        """
        # For LLaVA, we need to track where visual tokens were inserted
        # This is stored during prepare_inputs_labels_for_multimodal
        # For now, we'll use a heuristic: visual tokens are typically at the beginning
        # after the system prompt tokens

        # This will be set by the modified prepare method
        if hasattr(self, '_current_visual_token_positions'):
            return self._current_visual_token_positions

        # Fallback: assume visual tokens are in the middle portion
        # This is a simplified approach - in practice, we track this explicitly
        seq_len = inputs_embeds.shape[1]
        # Rough estimate: visual tokens typically start after ~20 tokens and span ~576 tokens
        start_idx = 20
        end_idx = min(start_idx + 576, seq_len - 10)
        return (start_idx, end_idx)
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """
        Modified generate method with intermediate layer feedback.

        Process:
        1. First pass: Extract complete intermediate layer sequence (vision + text)
        2. Second pass: Use complete intermediate sequence as input to Layer 0
        3. Generate output

        This ensures both vision and text embeddings are at the same semantic level.
        """
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)

        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        # Prepare inputs with multimodal data
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                v_token_num,
                cls_attn
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
            v_token_num, cls_attn = 0, None

        # If intermediate feedback is enabled and we have visual tokens
        if self.use_intermediate_feedback and v_token_num > 0:
            # Stage 1: Extract complete intermediate layer sequence
            intermediate_sequence = self._extract_intermediate_embeddings(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

            # Stage 2: Use complete intermediate sequence as input
            if intermediate_sequence is not None:
                inputs_embeds = intermediate_sequence
                print(f"[IntermediateLayer] Using complete intermediate sequence as input")

        # Generate with (possibly modified) embeddings
        output = super(LlavaLlamaForCausalLM, self).generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

        return output, v_token_num, cls_attn
    
    def _extract_intermediate_embeddings(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Optional[torch.FloatTensor]:
        """
        Extract complete sequence from intermediate layer by running a forward pass.

        Args:
            inputs_embeds: Input embeddings [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask
            position_ids: Position IDs

        Returns:
            Complete intermediate layer hidden states (entire sequence including both vision and text)
            Shape: [batch_size, seq_len, hidden_dim]
        """
        # Run forward pass with output_hidden_states=True
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True,
        )

        # Extract hidden states from the specified intermediate layer
        # hidden_states is a tuple of (num_layers + 1) tensors
        # Index 0 is the embedding layer, index 1 is layer 0, etc.
        if outputs.hidden_states is None or len(outputs.hidden_states) <= self.intermediate_layer_idx + 1:
            print(f"[Warning] Cannot extract layer {self.intermediate_layer_idx}, "
                  f"only {len(outputs.hidden_states) if outputs.hidden_states else 0} layers available")
            return None

        # Get the intermediate layer hidden states (complete sequence)
        # +1 because index 0 is the embedding layer
        intermediate_hidden_states = outputs.hidden_states[self.intermediate_layer_idx + 1]

        print(f"[IntermediateLayer] Extracted complete sequence from layer {self.intermediate_layer_idx}, "
              f"shape: {intermediate_hidden_states.shape}")

        return intermediate_hidden_states


# Register the new model configuration and class
AutoConfig.register("llava_intermediate_layer", LlavaIntermediateLayerConfig)
AutoModelForCausalLM.register(LlavaIntermediateLayerConfig, LlavaIntermediateLayerForCausalLM)

