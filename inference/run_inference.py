"""
Run inference using precomputed MLP-aligned embeddings with LLaVA 13B LLM.

This script loads precomputed visual embeddings and uses them with the
LLaVA 13B language model to generate text responses based on per-image prompts.
"""

import os
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import sys
# Add repo root to path
PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from llava.model import LlavaLlamaForCausalLM
from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.mm_utils import tokenizer_image_token
from transformers import AutoTokenizer

from inference.cc_dataset import CCEmbeddingDataset


def load_llava_13b_llm(model_path: str, device, dtype):
    """
    Load LLaVA 13B model (primarily for the LLM component).
    
    Returns:
        model: The LLaVA 13B model
        tokenizer: The tokenizer
    """
    print(f"Loading LLaVA 13B from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    
    print("LLaVA 13B loaded successfully")
    return model, tokenizer


def prepare_inputs_with_embeddings(
    model,
    tokenizer,
    prompt: str,
    visual_embedding: torch.Tensor,
    conv_mode: str = "llava_v1",
    device=None,
):
    """
    Prepare inputs for generation using precomputed visual embeddings.
    
    Args:
        model: LLaVA model
        tokenizer: Tokenizer
        prompt: Text prompt
        visual_embedding: Precomputed visual embedding, shape (num_tokens, 5120)
        conv_mode: Conversation template mode
        device: Device to use
    
    Returns:
        inputs_embeds: Combined text + visual embeddings
        attention_mask: Attention mask
    """
    # Prepare conversation
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()

    # Tokenize with proper image token handling
    input_ids = tokenizer_image_token(prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(device)

    # Find IMAGE_TOKEN_INDEX position BEFORE embedding
    image_token_positions = (input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)

    if len(image_token_positions[1]) == 0:
        # No image token found, just embed text normally
        text_embeds = model.get_model().embed_tokens(input_ids)
        return text_embeds, torch.ones_like(input_ids)

    # Get the position of the image token
    image_token_idx = image_token_positions[1][0].item()

    # Split input_ids into before and after image token (excluding the IMAGE_TOKEN_INDEX itself)
    input_ids_before = input_ids[:, :image_token_idx]
    input_ids_after = input_ids[:, image_token_idx + 1:]

    # Embed the text parts (these don't contain IMAGE_TOKEN_INDEX)
    text_embeds_before = model.get_model().embed_tokens(input_ids_before) if input_ids_before.numel() > 0 else None
    text_embeds_after = model.get_model().embed_tokens(input_ids_after) if input_ids_after.numel() > 0 else None

    # Ensure visual embedding is on correct device and has batch dimension
    if visual_embedding.dim() == 2:
        visual_embedding = visual_embedding.unsqueeze(0)  # (1, num_tokens, 5120)
    visual_embedding = visual_embedding.to(device=device, dtype=model.dtype)

    # Concatenate: text_before + visual + text_after
    parts = []
    if text_embeds_before is not None:
        parts.append(text_embeds_before)
    parts.append(visual_embedding)
    if text_embeds_after is not None:
        parts.append(text_embeds_after)

    inputs_embeds = torch.cat(parts, dim=1)
    
    # Create attention mask
    attention_mask = torch.ones(
        inputs_embeds.shape[:2],
        dtype=torch.long,
        device=device
    )
    
    return inputs_embeds, attention_mask


def generate_response(
    model,
    tokenizer,
    prompt: str,
    visual_embedding: torch.Tensor,
    conv_mode: str = "llava_v1",
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.7,
    device=None,
):
    """
    Generate a response using the model with precomputed visual embeddings.

    Args:
        model: LLaVA model
        tokenizer: Tokenizer
        prompt: Text prompt
        visual_embedding: Precomputed visual embedding
        conv_mode: Conversation template mode
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        device: Device to use

    Returns:
        generated_text: The generated response
    """
    with torch.no_grad():
        inputs_embeds, attention_mask = prepare_inputs_with_embeddings(
            model, tokenizer, prompt, visual_embedding, conv_mode, device
        )

        # Generate using the parent class's generate method
        # We call super().generate() which is LlamaForCausalLM.generate()
        # This bypasses LLaVA's custom generate that rejects inputs_embeds
        output_ids = super(type(model), model).generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Decode
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Extract only the assistant's response
        # The output includes the full conversation, we want only the last response
        conv = conv_templates[conv_mode].copy()
        if conv.sep2 is not None:
            # For templates with two separators
            parts = generated_text.split(conv.sep2)
            if len(parts) > 1:
                generated_text = parts[-1].strip()
        else:
            # For templates with one separator
            parts = generated_text.split(conv.sep)
            if len(parts) > 1:
                generated_text = parts[-1].strip()

    return generated_text


def main():
    parser = argparse.ArgumentParser(description='Run inference with precomputed embeddings')
    
    # Model paths
    parser.add_argument('--model_13b', type=str, required=True,
                        help='Path to LLaVA v1.5 13B model')
    
    # Data paths
    parser.add_argument('--embeddings_dir', type=str, required=True,
                        help='Directory containing precomputed embeddings')
    parser.add_argument('--json_path', type=str, required=True,
                        help='Path to BLIP annotations JSON file')
    
    # Output
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to save inference results (JSON)')
    
    # Generation parameters
    parser.add_argument('--conv_mode', type=str, default='llava_v1',
                        help='Conversation template mode')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.2,
                        help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.7,
                        help='Nucleus sampling parameter')
    
    # Processing options
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (currently only supports 1)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (for testing)')
    parser.add_argument('--bf16', action='store_true',
                        help='Use bfloat16 precision')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    
    print(f"Device: {device}, dtype: {dtype}")
    
    # Load model
    model, tokenizer = load_llava_13b_llm(args.model_13b, device, dtype)
    
    # Create dataset
    dataset = CCEmbeddingDataset(
        embeddings_dir=args.embeddings_dir,
        json_path=args.json_path,
        max_samples=args.max_samples,
    )
    
    # Note: We use batch_size=1 for simplicity in this implementation
    # Batched inference would require padding embeddings to same length
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # Use 0 to avoid multiprocessing issues with embeddings
    )
    
    print(f"\nRunning inference on {len(dataset)} samples...")
    print(f"Results will be saved to {args.output_file}\n")
    
    # Run inference
    results = []
    total_time = 0
    for idx, batch in enumerate(tqdm(dataloader, desc="Generating responses")):
        t_start = time.time()

        embedding = batch['embedding'][0]  # Remove batch dimension
        image_id = batch['image_id'][0]
        prompt = batch['prompt'][0]
        answer = batch['answer'][0]

        # Generate response
        generated = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            visual_embedding=embedding,
            conv_mode=args.conv_mode,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )

        t_elapsed = time.time() - t_start
        total_time += t_elapsed

        if idx < 3:  # Print timing for first 3 samples
            print(f"\n  Sample {idx+1}: {t_elapsed:.2f}s")
        
        results.append({
            'image_id': image_id,
            'prompt': prompt,
            'ground_truth': answer,
            'generated': generated,
        })
    
    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    avg_time = total_time / len(results) if results else 0
    print(f"\nDone! Results saved to {args.output_file}")
    print(f"Processed {len(results)} samples")
    print(f"Average time per sample: {avg_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")


if __name__ == '__main__':
    main()

