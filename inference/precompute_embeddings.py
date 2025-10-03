"""
Precompute MLP-aligned visual embeddings for inference.

This script processes images through:
1. LLaVA 7B vision encoder + projector (outputs 4096-dim embeddings)
2. MLP alignment layer (4096 -> 5120-dim)
3. Saves the aligned embeddings to disk for later inference

The precomputed embeddings can then be used directly with LLaVA 13B LLM
for text generation without recomputing visual features.
"""

import os
import argparse
from pathlib import Path
from typing import Optional
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import sys
# Add repo root to path
PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from llava.model import LlavaLlamaForCausalLM
from train_mlp.alignment import MLPAlignment
from inference.cc_dataset import CCImageDataset


def load_llava_7b_vision(
    model_path: str,
    vision_tower: str,
    select_layer: int,
    select_feature: str,
    device,
    dtype,
):
    """
    Load LLaVA 7B vision encoder and projector.
    
    Returns:
        vision_tower: The vision encoder
        projector: The vision projector (outputs 4096-dim)
    """
    print(f"Loading LLaVA 7B from {model_path}...")
    model = LlavaLlamaForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    
    # Initialize vision modules
    model_cfg = argparse.Namespace(
        vision_tower=vision_tower,
        mm_vision_select_layer=select_layer,
        mm_vision_select_feature=select_feature,
        pretrain_mm_mlp_adapter=None,
        mm_patch_merge_type='flat',
    )
    model.get_model().initialize_vision_modules(model_args=model_cfg, fsdp=None)
    
    # Extract vision tower and projector
    vt = model.get_vision_tower()
    vt.to(dtype=dtype, device=device)
    projector = model.get_model().mm_projector.to(dtype=dtype, device=device)
    
    # Set to eval mode and freeze
    vt.eval()
    projector.eval()
    for p in vt.parameters():
        p.requires_grad = False
    for p in projector.parameters():
        p.requires_grad = False
    
    print("LLaVA 7B vision modules loaded successfully")
    return vt, projector


def load_mlp_alignment(checkpoint_path: str, device, dtype):
    """
    Load the trained MLP alignment layer.
    
    Returns:
        mlp: The MLP alignment model
    """
    print(f"Loading MLP alignment from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract state dict and config
    state_dict = ckpt.get('state_dict', ckpt)
    ckpt_args = ckpt.get('args', {})
    
    # Build MLP with same architecture as training
    mlp = MLPAlignment(
        input_dim=ckpt_args.get('mlp_input_dim', 4096),
        hidden_dim=ckpt_args.get('mlp_hidden_dim', 5120),
        output_dim=ckpt_args.get('mlp_output_dim', 5120),
        dropout=ckpt_args.get('mlp_dropout', 0.0),
        num_blocks=ckpt_args.get('mlp_num_blocks', 0),
        bottleneck_dim=ckpt_args.get('mlp_bottleneck_dim', 1024),
        residual_scale_init=ckpt_args.get('mlp_residual_scale_init', 1e-3),
        activation=ckpt_args.get('mlp_activation', 'relu'),
        out_zero_init=ckpt_args.get('mlp_out_zero_init', True),
        final_layernorm=ckpt_args.get('mlp_final_layernorm', False),
        normalize_output=ckpt_args.get('mlp_normalize_output', False),
    ).to(device=device, dtype=dtype)
    
    mlp.load_state_dict(state_dict)
    mlp.eval()
    for p in mlp.parameters():
        p.requires_grad = False
    
    print("MLP alignment loaded successfully")
    return mlp


def main():
    parser = argparse.ArgumentParser(description='Precompute MLP-aligned embeddings for CC dataset')
    
    # Model paths
    parser.add_argument('--model_7b', type=str, required=True,
                        help='Path to LLaVA v1.5 7B model')
    parser.add_argument('--mlp_checkpoint', type=str, default='../checkpoints/contrastive-mlp-cc/best_mlp_alignment.pt',
                        help='Path to trained MLP alignment checkpoint (.pt file)')
    
    # Dataset paths
    parser.add_argument('--cc_root', type=str, default='../playground/data/CC',
                        help='Root directory of CC dataset (e.g., playground/data/CC)')
    parser.add_argument('--json_path', type=str, default='blip_laion_cc_sbu_558k.json',
                        help='Path to BLIP annotations JSON file')
    
    # Output
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save precomputed embeddings')
    
    # Vision tower config
    parser.add_argument('--vision_tower', type=str, default='openai/clip-vit-large-patch14-336',
                        help='Vision tower model name or path')
    parser.add_argument('--mm_vision_select_layer', type=int, default=-2,
                        help='Which layer to select from vision tower')
    parser.add_argument('--mm_vision_select_feature', type=str, default='patch',
                        help='Which feature to select (patch or cls_patch)')
    
    # Processing options
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for processing')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (for testing)')
    parser.add_argument('--bf16', action='store_true',
                        help='Use bfloat16 precision')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    
    print(f"Device: {device}, dtype: {dtype}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    vt, projector = load_llava_7b_vision(
        args.model_7b,
        args.vision_tower,
        args.mm_vision_select_layer,
        args.mm_vision_select_feature,
        device,
        dtype,
    )
    
    mlp = load_mlp_alignment(args.mlp_checkpoint, device, dtype)
    
    # Create dataset
    image_processor = vt.image_processor
    dataset = CCImageDataset(
        cc_root=args.cc_root,
        json_path=args.json_path,
        image_processor=image_processor,
        max_samples=args.max_samples,
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    print(f"\nProcessing {len(dataset)} images...")
    print(f"Saving embeddings to {args.output_dir}\n")
    
    # Process images
    processed_count = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Precomputing embeddings"):
            images = batch['image'].to(device=device, dtype=dtype)
            image_ids = batch['image_id']
            
            # Forward through vision encoder
            vision_features, _ = vt(images)  # Shape: (B, num_tokens, vision_dim)
            
            # Forward through projector
            projected = projector(vision_features)  # Shape: (B, num_tokens, 4096)
            
            # Forward through MLP alignment
            # Flatten batch and tokens for MLP
            B, N, C = projected.shape
            projected_flat = projected.reshape(B * N, C)
            aligned_flat = mlp(projected_flat)  # Shape: (B*N, 5120)
            aligned = aligned_flat.reshape(B, N, -1)  # Shape: (B, num_tokens, 5120)
            
            # Save each embedding
            for i in range(B):
                emb = aligned[i].cpu()  # Shape: (num_tokens, 5120)
                image_id = image_ids[i]
                
                save_path = os.path.join(args.output_dir, f"{image_id}.pt")
                torch.save({
                    'embedding': emb,
                    'image_id': image_id,
                }, save_path)
                
                processed_count += 1
    
    print(f"\nDone! Processed {processed_count} images")
    print(f"Embeddings saved to {args.output_dir}")


if __name__ == '__main__':
    main()

