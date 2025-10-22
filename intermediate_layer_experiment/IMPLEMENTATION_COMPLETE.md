## 📊 Architecture Comparison

### Before: Semantic Mismatch
```
[text_L0] + [vision_L3] + [text_L0]
    ↑           ↑             ↑
 Layer 0     Layer 3       Layer 0  ❌ Inconsistent!
```

### After: Semantic Consistency
```
[text_L3] + [vision_L3] + [text_L3]
    ↑           ↑             ↑
 Layer 3     Layer 3       Layer 3  ✅ Consistent!
```

---

## 🚀 How to Use

### Basic Usage (Same as Before)

```bash
CUDA_VISIBLE_DEVICES=0 python intermediate_layer_experiment/scripts/model_vqa_intermediate_layer.py \
    --model-path /home/qingchan/project/FasterVLM/checkpoints/llava-v1.5-7b \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./results/output.jsonl \
    --intermediate-layer-idx 3 \
    --visual-token-num 576 \
    --use-intermediate-feedback \
    --temperature 0 \
    --conv-mode vicuna_v1
```

### Using the Evaluation Script

```bash
# With intermediate feedback (new implementation)
bash intermediate_layer_experiment/scripts/eval_textvqa_intermediate.sh 3 576 true

# Without intermediate feedback (baseline)
bash intermediate_layer_experiment/scripts/eval_textvqa_intermediate.sh 3 576 false
```