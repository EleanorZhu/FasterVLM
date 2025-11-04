"""
Training script for prompt tuning on intermediate layer embeddings.

This script trains learnable prompt tokens to help the LLM better interpret
intermediate layer embeddings from the vision-language model.
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import json
import sys
from tqdm import tqdm
from PIL import Image
from typing import Dict, List
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import wandb

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
from llava import conversation as conversation_lib
from intermediate_layer_experiment.model.llava_intermediate_layer import (
    LlavaIntermediateLayerForCausalLM,
    LlavaIntermediateLayerConfig
)


class TextVQAPromptTuningDataset(Dataset):
    """
    Dataset for training prompt tuning on TextVQA.
    
    Loads questions and answers from TextVQA and formats them for training.
    """
    
    def __init__(
        self,
        question_file: str,
        annotation_file: str,
        image_folder: str,
        tokenizer,
        image_processor,
        conv_mode: str = "vicuna_v1",
        max_samples: int = None
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_folder = image_folder
        self.conv_mode = conv_mode
        
        # Load questions
        with open(question_file, 'r') as f:
            self.questions = [json.loads(line) for line in f]

        # Load annotations
        with open(annotation_file, 'r') as f:
            annotations_data = json.load(f)

        # Create mapping from image_id to answers
        # Note: The JSONL uses image_id as question_id, but annotation uses image_id separately
        self.answers = {}
        for item in annotations_data['data']:
            # Use image_id as the key (matches question_id in JSONL)
            image_id = item['image_id']
            # Use the first answer as ground truth
            self.answers[image_id] = item['answers'][0] if item['answers'] else ""

        # Filter to only samples with answers
        # The JSONL question_id corresponds to annotation image_id
        self.questions = [q for q in self.questions if self.answers.get(q['question_id'])]

        if max_samples:
            self.questions = self.questions[:max_samples]

        print(f"Loaded {len(self.questions)} samples from TextVQA")
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question_data = self.questions[idx]
        question_id = question_data['question_id']
        question_text = question_data['text']
        image_file = question_data['image']
        answer = self.answers[question_id]
        
        # Load and process image
        image_path = os.path.join(self.image_folder, image_file)
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        # Format conversation
        conv = conversation_lib.conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question_text)
        conv.append_message(conv.roles[1], answer)
        prompt = conv.get_prompt()
        
        # Tokenize
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        
        # Create labels - mask everything except the answer
        labels = input_ids.clone()
        
        # Find where the answer starts
        # The answer comes after the second role (ASSISTANT)
        sep = conv.sep + conv.roles[1] + ": "
        parts = prompt.split(sep)
        if len(parts) == 2:
            # Tokenize the question part to find where to start unmasking
            question_part = parts[0] + sep
            question_len = len(tokenizer_image_token(question_part, self.tokenizer, IMAGE_TOKEN_INDEX))
            # Mask the question part
            labels[:question_len] = IGNORE_INDEX
        else:
            # Fallback: mask everything (shouldn't happen)
            labels[:] = IGNORE_INDEX
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'image': image_tensor,
            'question_id': question_id
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    # Find max length
    max_len = max(item['input_ids'].shape[0] for item in batch)
    
    # Pad sequences
    input_ids_list = []
    labels_list = []
    images = []
    question_ids = []
    
    for item in batch:
        input_ids = item['input_ids']
        labels = item['labels']
        
        # Pad
        padding_len = max_len - input_ids.shape[0]
        if padding_len > 0:
            input_ids = torch.cat([
                input_ids,
                torch.full((padding_len,), 0, dtype=input_ids.dtype)  # pad_token_id = 0
            ])
            labels = torch.cat([
                labels,
                torch.full((padding_len,), IGNORE_INDEX, dtype=labels.dtype)
            ])
        
        input_ids_list.append(input_ids)
        labels_list.append(labels)
        images.append(item['image'])
        question_ids.append(item['question_id'])
    
    return {
        'input_ids': torch.stack(input_ids_list),
        'labels': torch.stack(labels_list),
        'images': torch.stack(images),
        'attention_mask': torch.stack(input_ids_list).ne(0),  # 0 is pad_token_id
        'question_ids': question_ids
    }


def convert_to_intermediate_layer_model(
    base_model,
    tokenizer,
    image_processor,
    intermediate_layer_idx: int,
    num_prompt_tokens: int,
    prompt_init_method: str,
    device: str,
    visual_token_num: int = 576,
):
    """
    Convert a base LLaVA model to use intermediate layer feedback with prompt tuning.
    """
    # Create config
    config = LlavaIntermediateLayerConfig.from_pretrained(
        base_model.config._name_or_path,
        intermediate_layer_idx=intermediate_layer_idx,
        use_intermediate_feedback=True,
        use_prompt_tuning=True,
        num_prompt_tokens=num_prompt_tokens,
        prompt_init_method=prompt_init_method
    )
    
    # Create new model (carry visual_token_num so pruning is well-defined)
    model = LlavaIntermediateLayerForCausalLM(config, visual_token_num=visual_token_num)

    # Copy weights from base model
    model.load_state_dict(base_model.state_dict(), strict=False)
    
    # Copy vision tower and projector
    model.get_model().vision_tower = base_model.get_model().vision_tower
    model.get_model().mm_projector = base_model.get_model().mm_projector

    # Ensure consistent dtype/device to avoid matmul dtype mismatch
    # Try to use fp16 for both; if the vision tower remains in another dtype, match projector to it
    vt = model.get_model().vision_tower
    try:
        vt.to(device=device, dtype=torch.float16)
    except TypeError:
        # Fallback for wrappers that don't accept device kwarg
        vt.to(dtype=torch.float16)
        vt.to(device)

    # Determine actual vision tower dtype and align projector with it
    try:
        vt_dtype = next(vt.parameters()).dtype
    except StopIteration:
        vt_dtype = torch.float16

    model.get_model().mm_projector.to(device=device, dtype=vt_dtype)

    # Match overall model dtype to base_model dtype (likely fp16)
    try:
        base_dtype = next(base_model.parameters()).dtype
    except StopIteration:
        base_dtype = torch.float16
    model = model.to(device=device, dtype=base_dtype)

    # IMPORTANT: Keep prompt_embeddings in fp32 for stable gradient updates
    # This prevents "Attempting to unscale FP16 gradients" error with GradScaler
    if hasattr(model, 'prompt_embeddings'):
        model.prompt_embeddings.data = model.prompt_embeddings.data.float()

    return model


def freeze_all_except_prompts(model):
    """Freeze all parameters except prompt embeddings."""
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze prompt embeddings
    if hasattr(model, 'prompt_embeddings'):
        model.prompt_embeddings.requires_grad = True
        print(f"Unfroze prompt_embeddings: {model.prompt_embeddings.shape}")
    else:
        raise ValueError("Model does not have prompt_embeddings!")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")


def verify_gradient_flow(model):
    """
    Verify that gradients only flow to prompt tokens, not to any other parameters.
    This is critical for proper prompt tuning methodology.
    """
    print("\n" + "="*70)
    print("GRADIENT FLOW VERIFICATION")
    print("="*70)

    # Check prompt embeddings
    prompt_has_grad = False
    if hasattr(model, 'prompt_embeddings'):
        if model.prompt_embeddings.grad is not None:
            grad_norm = model.prompt_embeddings.grad.norm().item()
            print(f"✓ Prompt embeddings have gradient (norm: {grad_norm:.6f})")
            prompt_has_grad = True
        else:
            print("✗ ERROR: Prompt embeddings have NO gradient!")
    else:
        print("✗ ERROR: Model does not have prompt_embeddings!")

    # Check that NO other parameters have gradients
    other_params_with_grad = []
    for name, param in model.named_parameters():
        if 'prompt_embeddings' not in name:
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                other_params_with_grad.append((name, grad_norm))

    if len(other_params_with_grad) == 0:
        print("✓ All non-prompt parameters have NO gradient (correct!)")
    else:
        print(f"✗ ERROR: {len(other_params_with_grad)} non-prompt parameters have gradients:")
        for name, grad_norm in other_params_with_grad[:5]:  # Show first 5
            print(f"  - {name}: grad_norm={grad_norm:.6f}")
        if len(other_params_with_grad) > 5:
            print(f"  ... and {len(other_params_with_grad) - 5} more")

    # Final verdict
    print("-"*70)
    if prompt_has_grad and len(other_params_with_grad) == 0:
        print("✓✓✓ GRADIENT FLOW IS CORRECT ✓✓✓")
        print("Only prompt tokens will be updated during optimization.")
    else:
        print("✗✗✗ GRADIENT FLOW IS INCORRECT ✗✗✗")
        print("Please check the implementation!")
    print("="*70 + "\n")

    return prompt_has_grad and len(other_params_with_grad) == 0


def train(args):
    # Initialize wandb
    wandb.init(
        project="fastervlm-prompt-tuning",
        name=f"layer{args.intermediate_layer_idx}_n{args.num_prompt_tokens}_lr{args.learning_rate}",
        config={
            "intermediate_layer_idx": args.intermediate_layer_idx,
            "num_prompt_tokens": args.num_prompt_tokens,
            "prompt_init_method": args.prompt_init_method,
            "visual_token_num": args.visual_token_num,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "conv_mode": args.conv_mode,
        }
    )

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load base model
    print(f"Loading base model from {args.model_path}...")
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, base_model, image_processor, context_len = load_pretrained_model(
        args.model_path, None, model_name, device_map=None
    )
    base_model = base_model.to(device)

    # Convert to intermediate layer model with prompt tuning
    print(f"Converting to intermediate layer model with prompt tuning...")
    model = convert_to_intermediate_layer_model(
        base_model,
        tokenizer,
        image_processor,
        args.intermediate_layer_idx,
        args.num_prompt_tokens,
        args.prompt_init_method,
        device,
        visual_token_num=args.visual_token_num,
    )

    # Freeze all except prompts
    freeze_all_except_prompts(model)
    
    # Create dataset
    print(f"Loading dataset...")
    dataset = TextVQAPromptTuningDataset(
        question_file=args.question_file,
        annotation_file=args.annotation_file,
        image_folder=args.image_folder,
        tokenizer=tokenizer,
        image_processor=image_processor,
        conv_mode=args.conv_mode,
        max_samples=args.max_samples
    )
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW([model.prompt_embeddings], lr=args.learning_rate, weight_decay=args.weight_decay)

    # Learning rate scheduler with warmup
    # Note: total_steps is based on optimizer steps (after gradient accumulation), not data batches
    total_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay after warmup
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP scaler for stability in fp16
    scaler = GradScaler(enabled=True)

    # Training loop
    print(f"\nStarting training for {args.num_epochs} epochs...")
    print(f"Batch size: {args.batch_size}, Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Total optimizer steps: {total_steps}, Warmup steps: {warmup_steps}")
    best_val_loss = float('inf')
    first_batch_done = False
    accumulated_loss = 0.0  # For tracking accumulated loss

    for epoch in range(args.num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]")

        for batch_idx, batch in enumerate(train_bar):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            images = batch['images'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Match image dtype to vision tower parameters to avoid dtype mismatch
            try:
                vt_dtype = next(model.get_model().vision_tower.parameters()).dtype
            except StopIteration:
                vt_dtype = images.dtype
            images = images.to(vt_dtype)

            # Forward pass with AMP for stability
            with autocast(dtype=torch.float16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    images=images,
                    use_cache=False
                )
                loss = outputs.loss
                # Scale loss by accumulation steps for correct gradient magnitude
                loss = loss / args.gradient_accumulation_steps

            # Skip batches with no supervised tokens or non-finite loss
            valid_tokens = (labels != IGNORE_INDEX).sum().item()
            if valid_tokens == 0 or not torch.isfinite(loss):
                train_bar.set_postfix({'loss': 'skip'})
                # Don't zero gradients here - we're accumulating
                continue

            # Backward pass with AMP scaler (accumulate gradients)
            scaler.scale(loss).backward()

            # Accumulate the original loss (before scaling) for logging
            accumulated_loss += loss.item() * args.gradient_accumulation_steps

            # Only update weights every gradient_accumulation_steps
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                # Clip gradients on prompts only
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([model.prompt_embeddings], max_norm=1.0)

                # Verify gradient flow on first batch (important correctness check)
                if not first_batch_done:
                    gradient_flow_correct = verify_gradient_flow(model)
                    if not gradient_flow_correct:
                        raise RuntimeError("Gradient flow verification failed! Only prompt tokens should have gradients.")
                    first_batch_done = True

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()  # Update learning rate
                optimizer.zero_grad(set_to_none=True)

                current_lr = scheduler.get_last_lr()[0]
                # Display average loss over accumulated batches
                display_loss = accumulated_loss / args.gradient_accumulation_steps
                train_bar.set_postfix({'loss': f'{display_loss:.3f}', 'lr': f'{current_lr:.2e}'})

                # Log to wandb
                wandb.log({
                    "train/loss": display_loss,
                    "train/lr": current_lr,
                    "train/step": epoch * (len(train_loader) // args.gradient_accumulation_steps) + (batch_idx // args.gradient_accumulation_steps),
                })

                train_loss += display_loss
                accumulated_loss = 0.0

        # Calculate average loss per batch (not per optimizer step)
        avg_train_loss = train_loss / (len(train_loader) // args.gradient_accumulation_steps)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Val]")
        
        with torch.no_grad():
            for batch in val_bar:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                images = batch['images'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                # Match image dtype to vision tower parameters
                try:
                    vt_dtype = next(model.get_model().vision_tower.parameters()).dtype
                except StopIteration:
                    vt_dtype = images.dtype
                images = images.to(vt_dtype)

                with autocast(dtype=torch.float16):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        images=images,
                        use_cache=False
                    )
                    loss = outputs.loss
                val_loss += loss.item()
                val_bar.set_postfix({'loss': loss.item()})
        
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{args.num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Log epoch metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train/epoch_loss": avg_train_loss,
            "val/epoch_loss": avg_val_loss,
        })

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(args.output_dir, exist_ok=True)
            output_path = os.path.join(
                args.output_dir,
                f"prompt_tokens_layer{args.intermediate_layer_idx}_n{args.num_prompt_tokens}.pt"
            )
            torch.save({
                'prompt_embeddings': model.prompt_embeddings.data.cpu(),
                'num_prompt_tokens': args.num_prompt_tokens,
                'intermediate_layer_idx': args.intermediate_layer_idx,
                'prompt_init_method': args.prompt_init_method,
                'epoch': epoch + 1,
                'val_loss': avg_val_loss
            }, output_path)
            print(f"Saved best model to {output_path}")

            # Log best model to wandb
            wandb.log({"val/best_loss": best_val_loss})

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train prompt tuning for intermediate layer embeddings")

    # Model arguments
    parser.add_argument("--model-path", type=str, required=True, help="Path to base LLaVA model")
    parser.add_argument("--intermediate-layer-idx", type=int, default=3, help="Which layer to extract embeddings from")
    parser.add_argument("--num-prompt-tokens", type=int, default=10, help="Number of learnable prompt tokens")
    parser.add_argument("--prompt-init-method", type=str, default="random", choices=["random", "from_vocab"],
                        help="How to initialize prompt embeddings")
    parser.add_argument("--visual-token-num", type=int, default=576,
                        help="Number of visual tokens")

    # Data arguments
    parser.add_argument("--question-file", type=str, required=True, help="Path to question JSONL file")
    parser.add_argument("--annotation-file", type=str, required=True, help="Path to annotation JSON file")
    parser.add_argument("--image-folder", type=str, required=True, help="Path to image folder")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1", help="Conversation mode")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to use")

    # Training arguments
    parser.add_argument("--output-dir", type=str, default="./intermediate_layer_experiment/checkpoints",
                        help="Output directory for checkpoints")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="Number of gradient accumulation steps (effective_batch_size = batch_size * gradient_accumulation_steps)")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Ratio of total steps for warmup")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    train(args)

