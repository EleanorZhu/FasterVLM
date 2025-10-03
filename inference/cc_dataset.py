"""
Dataset loader for CC (Conceptual Captions) dataset with BLIP annotations.

This module provides dataset classes for loading images from the CC dataset
and pairing them with prompts/annotations from the BLIP LAION CC SBU 558k JSON file.
"""

import json
from typing import Dict, Optional
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image


class CCImageDataset(Dataset):
    """
    Dataset for loading CC images with their corresponding prompts from JSON annotations.

    The CC dataset is organized in nested folders (00000/, 00001/, etc.), and the
    JSON file contains image paths relative to the CC root and conversation-style annotations.

    Args:
        cc_root: Root directory of the CC dataset (e.g., 'playground/data/CC')
        json_path: Path to the BLIP annotations JSON file
        image_processor: Optional image processor/transform to apply to images
        max_samples: Optional limit on number of samples to load
    """

    def __init__(
        self,
        cc_root: str,
        json_path: str,
        image_processor=None,
        max_samples: Optional[int] = None,
    ):
        self.cc_root = Path(cc_root)
        self.image_processor = image_processor

        # Load JSON annotations
        print(f"Loading annotations from {json_path}...")
        with open(json_path, 'r') as f:
            self.annotations = json.load(f)

        if max_samples is not None:
            self.annotations = self.annotations[:max_samples]

        print(f"Loaded {len(self.annotations)} annotations")

        # Build image path mapping and validate existence
        self.valid_samples = []
        missing_count = 0

        for idx, ann in enumerate(self.annotations):
            # Image path is relative to CC root (e.g., "00453/004539375.jpg")
            rel_path = ann['image']
            full_path = self.cc_root / rel_path

            if full_path.exists():
                self.valid_samples.append({
                    'index': idx,
                    'image_path': str(full_path),
                    'image_id': ann['id'],
                    'conversations': ann['conversations'],
                })
            else:
                missing_count += 1
                if missing_count <= 5:  # Only print first few missing files
                    print(f"Warning: Image not found: {full_path}")

        if missing_count > 0:
            print(f"Warning: {missing_count} images not found out of {len(self.annotations)}")

        print(f"Valid samples: {len(self.valid_samples)}")

    def __len__(self) -> int:
        return len(self.valid_samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a dictionary containing:
            - image: PIL Image or processed tensor (if image_processor is provided)
            - image_path: str, full path to the image file
            - image_id: str, unique identifier from JSON
            - prompt: str, the human prompt from conversations
            - answer: str, the expected answer from conversations
        """
        sample = self.valid_samples[idx]

        # Load image
        image = Image.open(sample['image_path']).convert('RGB')

        # Apply image processor if provided
        if self.image_processor is not None:
            image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        # Extract prompt and answer from conversations
        # Format: [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]
        conversations = sample['conversations']
        prompt = ""
        answer = ""

        for conv in conversations:
            if conv['from'] == 'human':
                # Remove <image> token from prompt for now
                prompt = conv['value'].replace('<image>', '').strip()
            elif conv['from'] == 'gpt':
                answer = conv['value']

        return {
            'image': image,
            'image_path': sample['image_path'],
            'image_id': sample['image_id'],
            'prompt': prompt,
            'answer': answer,
        }

    def get_image_path(self, idx: int) -> str:
        """Get the full image path for a given index."""
        return self.valid_samples[idx]['image_path']

    def get_image_id(self, idx: int) -> str:
        """Get the image ID for a given index."""
        return self.valid_samples[idx]['image_id']


class CCEmbeddingDataset(Dataset):
    """
    Dataset for loading precomputed MLP-aligned embeddings with their prompts.

    This dataset is used for the inference stage, where embeddings have been
    precomputed and saved to disk.

    Args:
        embeddings_dir: Directory containing .pt files with precomputed embeddings
        json_path: Path to the BLIP annotations JSON file
        max_samples: Optional limit on number of samples to load
    """

    def __init__(
        self,
        embeddings_dir: str,
        json_path: str,
        max_samples: Optional[int] = None,
    ):
        self.embeddings_dir = Path(embeddings_dir)

        # Load JSON annotations
        print(f"Loading annotations from {json_path}...")
        with open(json_path, 'r') as f:
            self.annotations = json.load(f)

        if max_samples is not None:
            self.annotations = self.annotations[:max_samples]

        # Build mapping from image_id to embedding file
        self.valid_samples = []
        missing_count = 0

        for ann in self.annotations:
            image_id = ann['id']
            # Embedding files are named by image_id
            emb_path = self.embeddings_dir / f"{image_id}.pt"

            if emb_path.exists():
                self.valid_samples.append({
                    'embedding_path': str(emb_path),
                    'image_id': image_id,
                    'conversations': ann['conversations'],
                })
            else:
                missing_count += 1
                if missing_count <= 5:
                    print(f"Warning: Embedding not found: {emb_path}")

        if missing_count > 0:
            print(f"Warning: {missing_count} embeddings not found out of {len(self.annotations)}")

        print(f"Valid samples with embeddings: {len(self.valid_samples)}")

    def __len__(self) -> int:
        return len(self.valid_samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a dictionary containing:
            - embedding: torch.Tensor, shape (num_tokens, 5120)
            - image_id: str, unique identifier
            - prompt: str, the human prompt from conversations
            - answer: str, the expected answer from conversations
        """
        sample = self.valid_samples[idx]

        # Load precomputed embedding
        emb_data = torch.load(sample['embedding_path'], map_location='cpu')
        embedding = emb_data['embedding']  # Shape: (num_tokens, 5120)

        # Extract prompt and answer from conversations
        conversations = sample['conversations']
        prompt = ""
        answer = ""

        for conv in conversations:
            if conv['from'] == 'human':
                prompt = conv['value'].replace('<image>', '').strip()
            elif conv['from'] == 'gpt':
                answer = conv['value']

        return {
            'embedding': embedding,
            'image_id': sample['image_id'],
            'prompt': prompt,
            'answer': answer,
        }
