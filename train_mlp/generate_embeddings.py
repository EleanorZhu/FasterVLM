import os
import argparse
from typing import List, Optional

import torch
from PIL import Image

import sys
from pathlib import Path
# Add repo root so `import llava` works when running from train_mlp/
PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from llava.model import LlavaLlamaForCausalLM

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def find_images(root: str) -> List[str]:
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if os.path.splitext(f.lower())[1] in IMG_EXTS:
                paths.append(os.path.join(dirpath, f))
    return paths


def load_stage(stage_model_path: str, vision_tower: str, select_layer: int, select_feature: str, device, dtype):
    model = LlavaLlamaForCausalLM.from_pretrained(stage_model_path, torch_dtype=dtype)
    stage_cfg = argparse.Namespace(
        vision_tower=vision_tower,
        mm_vision_select_layer=select_layer,
        mm_vision_select_feature=select_feature,
        pretrain_mm_mlp_adapter=None,
        mm_patch_merge_type='flat',
    )
    model.get_model().initialize_vision_modules(model_args=stage_cfg, fsdp=None)
    vt = model.get_vision_tower()
    vt.to(dtype=dtype, device=device)
    projector = model.get_model().mm_projector.to(dtype=dtype, device=device)
    vt.eval(); projector.eval()
    for p in vt.parameters():
        p.requires_grad = False
    for p in projector.parameters():
        p.requires_grad = False
    return vt, projector


def main():
    parser = argparse.ArgumentParser(description='Precompute (X, y) embedding pairs for contrastive MLP alignment.')
    parser.add_argument('--model_7b', type=str, required=True, help='Path or HF id of LLaVA v1.5 7B')
    parser.add_argument('--model_13b', type=str, required=True, help='Path or HF id of LLaVA v1.5 13B')
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--vision_tower', type=str, default='openai/clip-vit-large-patch14-336')
    parser.add_argument('--mm_vision_select_layer', type=int, default=-2)
    parser.add_argument('--mm_vision_select_feature', type=str, default='patch')
    parser.add_argument('--bf16', type=lambda x: str(x).lower() == 'true', default=False)
    parser.add_argument('--max_images', type=int, default=None)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    pairs_dir = os.path.join(args.output_dir, 'pairs')
    os.makedirs(pairs_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    vt7, proj7 = load_stage(args.model_7b, args.vision_tower, args.mm_vision_select_layer, args.mm_vision_select_feature, device, dtype)
    vt13, proj13 = load_stage(args.model_13b, args.vision_tower, args.mm_vision_select_layer, args.mm_vision_select_feature, device, dtype)

    image_paths = find_images(args.images_dir)
    if args.max_images:
        image_paths = image_paths[:args.max_images]

    processor = vt7.image_processor

    for i, path in enumerate(image_paths):
        img = Image.open(path).convert('RGB')
        pixel_values = processor.preprocess(img, return_tensors='pt')['pixel_values'].to(device=device, dtype=dtype)
        with torch.no_grad():
            f7, _ = vt7(pixel_values)
            f13, _ = vt13(pixel_values)
            x = proj7(f7).cpu()  # (1, N, 4096)
            y = proj13(f13).cpu()  # (1, N, 5120)
        x = x.squeeze(0)
        y = y.squeeze(0)
        base = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(pairs_dir, f'{base}.pt')
        torch.save({'x': x, 'y': y, 'image_path': path}, out_path)
        if (i + 1) % 10 == 0:
            print(f'[{i+1}/{len(image_paths)}] saved {out_path}')

    print(f'Done. Saved {len(image_paths)} pairs under {pairs_dir}')


if __name__ == '__main__':
    main()

