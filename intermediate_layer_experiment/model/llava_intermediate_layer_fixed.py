#    Modified LLaVA model with intermediate layer extraction and feedback (FIXED VERSION)
#
#    This version fixes the train/inference distribution mismatch:
#    - Training: Extract intermediate layer from question only (no answer)
#    - Then append answer embeddings for loss computation
#    - This ensures intermediate_seq is consistent between training and inference
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


class LlavaIntermediateLayerConfigFixed(LlavaConfig):
    """
    Configuration class for LlavaIntermediateLayerForCausalLMFixed.

    Adds configuration for intermediate layer extraction and prompt tuning.
    FIXED VERSION: Ensures train/inference consistency.
    """
    model_type = "llava_intermediate_layer_fixed"

    def __init__(
        self,
        intermediate_layer_idx: int = 3,
        use_intermediate_feedback: bool = True,
        use_prompt_tuning: bool = False,
        num_prompt_tokens: int = 10,
        prompt_init_method: str = "random",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.intermediate_layer_idx = intermediate_layer_idx
        self.use_intermediate_feedback = use_intermediate_feedback
        self.use_prompt_tuning = use_prompt_tuning
        self.num_prompt_tokens = num_prompt_tokens
        self.prompt_init_method = prompt_init_method


class LlavaIntermediateLayerForCausalLMFixed(LlavaLlamaForCausalLM):
    """
    Modified LLaVA model that extracts the complete sequence from an intermediate layer

    FIXED VERSION: Ensures train/inference consistency by:
    1. Extracting intermediate layer from question only (no answer)
    2. Appending answer embeddings after intermediate extraction (training only)
    3. This makes intermediate_seq identical between training and inference
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
    config_class = LlavaIntermediateLayerConfigFixed

    def __init__(self, config, visual_token_num=None):
        super().__init__(config, visual_token_num)

        # Configuration for intermediate layer extraction
        self.intermediate_layer_idx = getattr(config, 'intermediate_layer_idx', 3)
        self.use_intermediate_feedback = getattr(config, 'use_intermediate_feedback', True)

        # Configuration for prompt tuning
        self.use_prompt_tuning = getattr(config, 'use_prompt_tuning', False)
        self.num_prompt_tokens = getattr(config, 'num_prompt_tokens', 10)
        self.prompt_init_method = getattr(config, 'prompt_init_method', 'random')

        # Initialize learnable prompt tokens if enabled
        if self.use_prompt_tuning:
            hidden_size = config.hidden_size
            self.prompt_embeddings = nn.Parameter(
                torch.zeros(self.num_prompt_tokens, hidden_size)
            )
            self._initialize_prompt_embeddings()
            print(f"[PromptTuning] Initialized {self.num_prompt_tokens} prompt tokens "
                  f"with method '{self.prompt_init_method}'")

        print(f"[IntermediateLayer] Initialized with layer_idx={self.intermediate_layer_idx}, "
              f"feedback={self.use_intermediate_feedback}, "
              f"prompt_tuning={self.use_prompt_tuning}")

    def _initialize_prompt_embeddings(self):
        """
        Initialize prompt embeddings based on the specified method.
        """
        if self.prompt_init_method == "random":
            # Initialize with small random values
            nn.init.normal_(self.prompt_embeddings, mean=0.0, std=0.02)
        elif self.prompt_init_method == "from_vocab":
            # Initialize from random vocabulary embeddings
            vocab_size = self.get_model().embed_tokens.weight.shape[0]
            random_indices = torch.randint(0, vocab_size, (self.num_prompt_tokens,))
            with torch.no_grad():
                self.prompt_embeddings.data = self.get_model().embed_tokens.weight[random_indices].clone()
        else:
            raise ValueError(f"Unknown prompt_init_method: {self.prompt_init_method}")

    def _prepend_prompt_tokens(self, embeddings: torch.FloatTensor) -> torch.FloatTensor:
        """
        Prepend learnable prompt tokens to the input embeddings.

        Args:
            embeddings: Input embeddings [batch_size, seq_len, hidden_dim]

        Returns:
            Embeddings with prompt tokens prepended [batch_size, seq_len + num_prompt_tokens, hidden_dim]
        """
        batch_size = embeddings.shape[0]
        # Expand prompt embeddings to batch size and convert to same dtype/device as embeddings
        prompt_tokens = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        prompt_tokens = prompt_tokens.to(dtype=embeddings.dtype, device=embeddings.device)
        # Prepend to embeddings
        return torch.cat([prompt_tokens, embeddings], dim=1)

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

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        answer_input_ids: Optional[torch.LongTensor] = None,  # NEW: Answer tokens to append after intermediate extraction
        answer_labels: Optional[torch.LongTensor] = None,  # NEW: Answer labels (with proper padding handling)
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass with support for intermediate layer feedback and prompt tuning.

        FIXED VERSION: Two-stage training to ensure train/inference consistency.

        Args:
            answer_input_ids: Optional answer token IDs to append AFTER intermediate extraction.
                             Only used during training with prompt tuning.
                             Shape: [batch_size, answer_seq_len]

        Training flow:
            1. input_ids contains ONLY question (no answer)
            2. Extract intermediate layer from question
            3. Add prompt tokens
            4. Append answer_input_ids embeddings
            5. Compute loss on answer tokens

        Inference flow:
            1. input_ids contains question
            2. Extract intermediate layer
            3. Add prompt tokens
            4. Generate (no answer_input_ids)
        """
        # Prepare multimodal inputs
        if inputs_embeds is None:
            prepare_result = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
            # Parent may return 6 or more values (e.g., with extra visual metadata). Keep the first 6.
            if isinstance(prepare_result, (tuple, list)):
                input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels = prepare_result[:6]
            else:
                input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels = prepare_result

        # Apply intermediate layer feedback if enabled
        if self.use_intermediate_feedback and images is not None:
            # Extract intermediate embeddings
            # IMPORTANT: When prompt tuning is enabled, we MUST detach intermediate embeddings
            # to prevent gradient flow. This ensures gradients only update prompt tokens.
            intermediate_sequence = self._extract_intermediate_embeddings(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                detach=self.use_prompt_tuning,  # Detach for prompt tuning, allow grad otherwise
            )

            if intermediate_sequence is not None:
                inputs_embeds = intermediate_sequence

                # Add prompt tokens if enabled
                if self.use_prompt_tuning:
                    # At this point, inputs_embeds is detached (requires_grad=False)
                    # After prepending prompts, the result will have requires_grad=True
                    # but gradients will ONLY flow to prompt_embeddings, not to inputs_embeds
                    inputs_embeds = self._prepend_prompt_tokens(inputs_embeds)

                    # Adjust attention mask
                    batch_size = attention_mask.shape[0]
                    prompt_attention = torch.ones(
                        batch_size, self.num_prompt_tokens,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device
                    )
                    attention_mask = torch.cat([prompt_attention, attention_mask], dim=1)

                    # Adjust position IDs
                    # If position_ids is None, create it based on the sequence length
                    if position_ids is None:
                        # Create position IDs for the intermediate sequence (before adding prompts)
                        seq_len = inputs_embeds.shape[1] - self.num_prompt_tokens  # Length before prompts
                        position_ids = torch.arange(
                            seq_len,
                            dtype=torch.long,
                            device=inputs_embeds.device
                        ).unsqueeze(0).expand(batch_size, -1)

                    # Now prepend position IDs for prompt tokens
                    prompt_position_ids = torch.arange(
                        self.num_prompt_tokens,
                        dtype=position_ids.dtype,
                        device=position_ids.device
                    ).unsqueeze(0).expand(batch_size, -1)
                    position_ids = torch.cat([prompt_position_ids, position_ids + self.num_prompt_tokens], dim=1)

                    # Adjust labels - prepend IGNORE_INDEX for prompt tokens
                    if labels is not None:
                        from llava.constants import IGNORE_INDEX
                        prompt_labels = torch.full(
                            (batch_size, self.num_prompt_tokens),
                            IGNORE_INDEX,
                            dtype=labels.dtype,
                            device=labels.device
                        )
                        labels = torch.cat([prompt_labels, labels], dim=1)

                    # NEW: Append answer embeddings (during training AND validation)
                    # This is the KEY FIX: answer is added AFTER intermediate extraction
                    # so intermediate_seq doesn't contain answer (consistent with inference)
                    # Note: We append answer in both training and validation (for loss computation)
                    # but NOT during inference (when answer_input_ids is None)
                    if answer_input_ids is not None:
                        # Get answer embeddings
                        answer_embeds = self.get_model().embed_tokens(answer_input_ids)

                        # Append to inputs_embeds
                        inputs_embeds = torch.cat([inputs_embeds, answer_embeds], dim=1)

                        # Extend attention mask for answer tokens
                        answer_attention = torch.ones(
                            batch_size, answer_input_ids.shape[1],
                            dtype=attention_mask.dtype,
                            device=attention_mask.device
                        )
                        attention_mask = torch.cat([attention_mask, answer_attention], dim=1)

                        # Extend position IDs for answer tokens
                        current_seq_len = position_ids.shape[1]
                        answer_position_ids = torch.arange(
                            current_seq_len,
                            current_seq_len + answer_input_ids.shape[1],
                            dtype=position_ids.dtype,
                            device=position_ids.device
                        ).unsqueeze(0).expand(batch_size, -1)
                        position_ids = torch.cat([position_ids, answer_position_ids], dim=1)

                        # Extend labels for answer tokens
                        # Use provided answer_labels if available, otherwise clone from answer_input_ids
                        if labels is not None:
                            if answer_labels is None:
                                # Fallback: use answer_input_ids as labels
                                answer_labels_to_use = answer_input_ids.clone()
                            else:
                                # Use provided answer_labels (with proper padding handling)
                                answer_labels_to_use = answer_labels
                            labels = torch.cat([labels, answer_labels_to_use], dim=1)

        # Call parent forward
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

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

                # Stage 3: Add prompt tokens if enabled
                if self.use_prompt_tuning:
                    inputs_embeds = self._prepend_prompt_tokens(inputs_embeds)

                    # Adjust attention mask for prompt tokens
                    batch_size = inputs_embeds.shape[0]
                    if attention_mask is None:
                        # If no attention mask was provided (common in inference), create one for the full sequence
                        attention_mask = torch.ones(
                            batch_size, inputs_embeds.shape[1],
                            dtype=torch.bool,
                            device=inputs_embeds.device,
                        )
                    else:
                        prompt_attention = torch.ones(
                            batch_size, self.num_prompt_tokens,
                            dtype=attention_mask.dtype,
                            device=attention_mask.device
                        )
                        attention_mask = torch.cat([prompt_attention, attention_mask], dim=1)

                    # Adjust position IDs for prompt tokens
                    if position_ids is not None:
                        prompt_position_ids = torch.arange(
                            self.num_prompt_tokens,
                            dtype=position_ids.dtype,
                            device=position_ids.device
                        ).unsqueeze(0).expand(batch_size, -1)
                        position_ids = torch.cat([prompt_position_ids, position_ids + self.num_prompt_tokens], dim=1)

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
        detach: bool = True,
    ) -> Optional[torch.FloatTensor]:
        """
        Extract complete sequence from intermediate layer by running a forward pass.

        Args:
            inputs_embeds: Input embeddings [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask
            position_ids: Position IDs
            detach: If True, detach embeddings to prevent gradient flow (for prompt tuning).
                   When True, gradients will NOT flow back through the intermediate extraction.
                   This is essential for proper prompt tuning where only prompt tokens should be updated.

        Returns:
            Complete intermediate layer hidden states (entire sequence including both vision and text)
            Shape: [batch_size, seq_len, hidden_dim]
        """
        # For prompt tuning, we must extract intermediate embeddings WITHOUT gradients
        # This ensures gradients only flow to prompt tokens, not through the LLM layers
        if detach:
            with torch.no_grad():
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

            # Explicitly detach to ensure no gradient flow
            intermediate_hidden_states = intermediate_hidden_states.detach()
        else:
            # Original behavior: allow gradient flow (for non-prompt-tuning use cases)
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                return_dict=True,
            )

            # Extract hidden states from the specified intermediate layer
            if outputs.hidden_states is None or len(outputs.hidden_states) <= self.intermediate_layer_idx + 1:
                print(f"[Warning] Cannot extract layer {self.intermediate_layer_idx}, "
                      f"only {len(outputs.hidden_states) if outputs.hidden_states else 0} layers available")
                return None

            intermediate_hidden_states = outputs.hidden_states[self.intermediate_layer_idx + 1]

        return intermediate_hidden_states


# Register the new model configuration and class
AutoConfig.register("llava_intermediate_layer_fixed", LlavaIntermediateLayerConfigFixed)
AutoModelForCausalLM.register(LlavaIntermediateLayerConfigFixed, LlavaIntermediateLayerForCausalLMFixed)

