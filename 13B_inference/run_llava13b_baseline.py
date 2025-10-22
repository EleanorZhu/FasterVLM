"""
Run baseline inference using full LLaVA 1.5-13B model (without MLP alignment).

This script processes images through the complete 13B pipeline:
  Image → 13B Vision Encoder → 13B Projector → 13B LLM → Generated Text
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import time

import torch
from PIL import Image

import sys
# Add repo root to path
PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates


def load_test_annotations(json_path: str) -> List[Dict]:
    """Load test annotations from JSON file."""
    print(f"Loading test annotations from {json_path}...")
    with open(json_path, 'r') as f:
        annotations = json.load(f)
    print(f"  Loaded {len(annotations)} test samples")
    return annotations


def run_inference_on_sample(
    model,
    tokenizer,
    image_processor,
    image_path: str,
    prompt: str,
    conv_mode: str = "llava_v1",
    max_new_tokens: int = 576,
    temperature: float = 0.0,
    top_p: float = None,
    num_beams: int = 1,
    device: str = "cuda",
) -> str:
    """
    Run inference on a single image using the full LLaVA 13B model.

    Args:
        model: LLaVA 13B model
        tokenizer: Tokenizer
        image_processor: Image processor
        image_path: Path to the image file
        prompt: Text prompt
        conv_mode: Conversation template mode
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 = greedy)
        top_p: Nucleus sampling parameter
        num_beams: Number of beams for beam search
        device: Device to use

    Returns:
        generated_text: The generated response
    """
    # Load and process image
    image = Image.open(image_path).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)
    # Use model's dtype to avoid dtype mismatches
    image_tensor = image_tensor.to(device, dtype=model.dtype)
    image_sizes = [image.size]
    
    # Prepare conversation prompt
    qs = prompt
    if model.config.mm_use_im_start_end:
        from llava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()
    
    # Tokenize
    input_ids = tokenizer_image_token(
        prompt_text, 
        tokenizer, 
        IMAGE_TOKEN_INDEX, 
        return_tensors='pt'
    ).unsqueeze(0).to(device)
    
    # Generate
    with torch.inference_mode():
        # LLaVA's generate returns (output_ids, v_token_num, cls_attn)
        result = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=True if temperature > 0 else False,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

        # Unpack the result tuple
        if isinstance(result, tuple):
            output_ids = result[0]
        else:
            output_ids = result

    # Decode output
    generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return generated_text


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline inference with full LLaVA 13B model"
    )
    parser.add_argument(
        "--model_13b",
        type=str,
        default="checkpoints/llava-v1.5-13b",
        help="Path or HF id of LLaVA v1.5 13B model",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="13B_inference/blip_laion_cc_200_test_samples.json",
        help="Path to test annotations JSON",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="playground/data/CC_200",
        help="Directory containing test images",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="13B_inference/results/llava13b_baseline_inference.json",
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--conv_mode",
        type=str,
        default="vicuna_v1",
        help="Conversation template mode",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=576,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy decoding)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Nucleus sampling parameter (default: None)",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Number of beams for beam search (default: 1)",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 precision",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of samples to process (for testing)",
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading LLaVA 13B model from {args.model_13b}...")
    model_name = get_model_name_from_path(args.model_13b)

    # Determine dtype
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    if args.bf16:
        print("  Using bfloat16 precision")

    # Load model with single GPU - pass device directly as device_map
    # This avoids multi-GPU device mismatch issues with FasterVLM's encode_images
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_13b,
        model_base=None,
        model_name=model_name,
        device_map=device,  # Use device string directly (not "auto")
        device=device,
        torch_dtype=dtype,
    )

    # Set visual_token_num to 576 (full tokens, no pruning) for baseline
    # This is required by FasterVLM's encode_images method
    model.visual_token_num = 576

    model.eval()
    print("  Model loaded successfully")
    print(f"  Visual token num: {model.visual_token_num} (full tokens, no pruning)")
    
    # Load test annotations
    annotations = load_test_annotations(args.json_path)
    
    if args.max_samples is not None:
        annotations = annotations[:args.max_samples]
        print(f"  Limited to {len(annotations)} samples for testing")
    
    # Process each sample
    results = []
    total_time = 0
    images_dir = Path(args.images_dir)
    
    print(f"\nProcessing {len(annotations)} samples...")
    for idx, ann in enumerate(tqdm(annotations, desc="Running inference")):
        t_start = time.time()
        
        # Extract information from annotation
        image_id = ann['id']
        image_filename = ann['image']
        image_path = images_dir / image_filename
        
        # Extract prompt and ground truth from conversations
        prompt = ""
        ground_truth = ""
        for conv in ann['conversations']:
            if conv['from'] == 'human':
                # Remove <image> token from prompt
                prompt = conv['value'].replace('<image>', '').strip()
            elif conv['from'] == 'gpt':
                ground_truth = conv['value']
        
        # Check if image exists
        if not image_path.exists():
            print(f"\n  Warning: Image not found: {image_path}")
            continue
        
        # Run inference
        try:
            generated = run_inference_on_sample(
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                image_path=str(image_path),
                prompt=prompt,
                conv_mode=args.conv_mode,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                device=device,
            )
        except Exception as e:
            import traceback
            print(f"\n  Error processing {image_id}: {e}")
            print(f"  Traceback:")
            traceback.print_exc()
            generated = f"[ERROR: {str(e)}]"
        
        t_elapsed = time.time() - t_start
        total_time += t_elapsed
        
        # Print timing for first few samples
        if idx < 3:
            print(f"\n  Sample {idx+1} ({image_id}): {t_elapsed:.2f}s")
            print(f"    Prompt: {prompt[:80]}...")
            print(f"    Generated: {generated[:80]}...")
        
        results.append({
            'image_id': image_id,
            'prompt': prompt,
            'ground_truth': ground_truth,
            'generated': generated,
        })
    
    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    avg_time = total_time / len(results) if results else 0
    print(f"\n{'='*60}")
    print("INFERENCE COMPLETE")
    print(f"{'='*60}")
    print(f"Total samples processed: {len(results)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per sample: {avg_time:.2f}s")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

