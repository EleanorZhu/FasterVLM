"""
Inference script for TextVQA evaluation with intermediate layer feedback.

This script is adapted from llava/eval/model_vqa_loader.py to work with
the modified LlavaIntermediateLayerForCausalLM model.
"""

import argparse
import torch
import os
import json
import sys
from tqdm import tqdm
import shortuuid

# Add parent directory to path to import our custom model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math

# Import our custom model
from intermediate_layer_experiment.model import LlavaIntermediateLayerForCausalLM


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def load_model_with_intermediate_layer(model_path, intermediate_layer_idx, visual_token_num=None, use_intermediate_feedback=True):
    """
    Load the LLaVA model and convert it to use intermediate layer feedback.

    Args:
        model_path: Path to the pretrained LLaVA model
        intermediate_layer_idx: Which layer to extract embeddings from
        visual_token_num: Number of visual tokens (for FasterVLM)
        use_intermediate_feedback: Whether to use intermediate layer feedback (default: True)

    Returns:
        tokenizer, model, image_processor, context_len
    """
    from transformers import AutoTokenizer
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path

    if use_intermediate_feedback:
        print(f"Loading LLaVA model with intermediate layer {intermediate_layer_idx}...")
    else:
        print(f"Loading LLaVA model without intermediate layer feedback (using projector output only)...")

    # Load the standard model first to get everything initialized properly
    # Use device="cuda:0" to ensure all components are on the same device
    model_name = get_model_name_from_path(model_path)
    tokenizer, base_model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name,
        visual_token_num=visual_token_num,
        device="cuda:0"
    )

    # Now we need to convert the base model to our custom model
    # The key is to preserve the vision tower and projector that are already initialized
    if use_intermediate_feedback:
        print(f"Converting to intermediate layer model (layer {intermediate_layer_idx})...")
    else:
        print(f"Using projector output directly (no intermediate layer extraction)...")

    # Get the base model's config and update it
    custom_config = base_model.config
    custom_config.intermediate_layer_idx = intermediate_layer_idx
    custom_config.use_intermediate_feedback = use_intermediate_feedback

    # Create our custom model with the same weights
    custom_model = LlavaIntermediateLayerForCausalLM(
        config=custom_config,
        visual_token_num=visual_token_num,
    )

    # Copy the state dict from base model
    custom_model.load_state_dict(base_model.state_dict(), strict=False)

    # Initialize vision modules using the same approach as the base model
    vision_tower = base_model.get_vision_tower()
    if vision_tower is not None:
        custom_model.get_model().vision_tower = vision_tower
        custom_model.get_model().mm_projector = base_model.get_model().mm_projector
        if hasattr(base_model.get_model(), 'image_newline'):
            custom_model.get_model().image_newline = base_model.get_model().image_newline

    # Move to GPU and set to eval mode
    # Use cuda:0 explicitly to avoid device mismatch
    device = 'cuda:0'

    # Move the entire model to the device
    # This should handle all submodules including vision tower
    custom_model = custom_model.to(device)

    custom_model.eval()

    print("Model loaded successfully!")

    return tokenizer, custom_model, image_processor, context_len


def eval_model(args):
    # Seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    
    print(f"Loading model with intermediate layer {args.intermediate_layer_idx}...")
    tokenizer, model, image_processor, context_len = load_model_with_intermediate_layer(
        model_path,
        args.intermediate_layer_idx,
        visual_token_num=args.visual_token_num,
        use_intermediate_feedback=args.use_intermediate_feedback
    )
    
    # Data
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    model_name = get_model_name_from_path(model_path)
    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'Auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    visual_token_nums = []
    data_bar = tqdm(zip(data_loader, questions), total=len(questions))
    
    for (input_ids, image_tensor, image_sizes), line in data_bar:
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids, v_token_num, image_attns = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)
        
        visual_token_nums.append(v_token_num)
        data_bar.set_postfix({"v_token_num": v_token_num})
        
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": cur_prompt,
            "text": outputs,
            "answer_id": ans_id,
            "model_id": model_name + f"_layer{args.intermediate_layer_idx}",
            "metadata": {
                "intermediate_layer": args.intermediate_layer_idx,
            }
        }) + "\n")
        ans_file.flush()
    
    ans_file.close()
    
    avg_token_num = sum(visual_token_nums) / len(visual_token_nums) if visual_token_nums else 0
    print(f"Average number of visual tokens: {avg_token_num}")
    print(f"Results saved to {answers_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--visual-token-num", type=int, default=None)
    parser.add_argument("--intermediate-layer-idx", type=int, default=3,
                        help="Which intermediate layer to extract embeddings from (0-indexed)")
    parser.add_argument("--use-intermediate-feedback", action="store_true", default=False,
                        help="Use intermediate layer feedback. If False, only use projector output.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    eval_model(args)

