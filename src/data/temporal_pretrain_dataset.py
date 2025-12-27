"""
Temporal Pretraining Dataset.

PyTorch Dataset for training the Temporal Encoder with LLM.
Generates synthetic (Resource_Timeline, Text_Description) pairs.
"""

import logging
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .resource_curve_synthesizer import ResourceCurveSynthesizer
from .text_description_generator import TextDescriptionGenerator

logger = logging.getLogger(__name__)


class TemporalPretrainDataset(Dataset):
    """
    Dataset for Temporal Encoder pretraining.
    
    Generates synthetic resource timelines with text descriptions
    for three task types: Trend (A), Bottleneck (B), Feasibility (C).
    """
    
    def __init__(
        self,
        num_samples: int,
        tokenizer: AutoTokenizer,
        type_distribution: Optional[Dict[str, float]] = None,
        max_length: int = 256,
        time_granularity_ms: int = 100,
        noise_level: float = 0.05,
        spike_probability: float = 0.3,
        seed: Optional[int] = None
    ):
        """
        Initialize dataset.
        
        Args:
            num_samples: Total number of samples
            tokenizer: Qwen tokenizer for text encoding
            type_distribution: Task type distribution {"A": 0.4, "B": 0.3, "C": 0.3}
            max_length: Maximum text length in tokens
            time_granularity_ms: Time step granularity
            noise_level: Noise level for curve generation
            spike_probability: Probability of spike injection
            seed: Random seed
        """
        self.num_samples = num_samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if type_distribution is None:
            type_distribution = {"A": 0.4, "B": 0.3, "C": 0.3}
        
        # Validate distribution
        total = sum(type_distribution.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"type_distribution must sum to 1.0, got {total}")
        
        self.type_distribution = type_distribution
        
        # Initialize synthesizers
        self.curve_synthesizer = ResourceCurveSynthesizer(
            time_granularity_ms=time_granularity_ms,
            noise_level=noise_level,
            spike_probability=spike_probability,
            seed=seed
        )
        
        self.text_generator = TextDescriptionGenerator(
            time_granularity_ms=time_granularity_ms,
            seed=seed
        )
        
        # Pre-generate task type assignments
        import random
        if seed is not None:
            random.seed(seed)
        
        self.task_types = self._generate_task_assignments()
        
        logger.info(f"Initialized TemporalPretrainDataset with {num_samples} samples")
        logger.info(f"Task distribution: {self._count_task_types()}")
    
    def _generate_task_assignments(self) -> List[str]:
        """Generate task type for each sample based on distribution."""
        import random
        
        task_types = []
        for _ in range(self.num_samples):
            r = random.random()
            cumsum = 0
            for task_type, prob in self.type_distribution.items():
                cumsum += prob
                if r < cumsum:
                    task_types.append(task_type)
                    break
        
        return task_types
    
    def _count_task_types(self) -> Dict[str, int]:
        """Count samples per task type."""
        from collections import Counter
        return dict(Counter(self.task_types))
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            {
                'curve': torch.Tensor,        # (num_timesteps, 4)
                'prompt_ids': torch.Tensor,   # (prompt_len,)
                'target_ids': torch.Tensor,   # (target_len,)
                'attention_mask': torch.Tensor,
                'labels': torch.Tensor,       # For loss computation
                'task_type': str              # "A", "B", or "C"
            }
        """
        task_type = self.task_types[idx]
        
        # Generate timeline based on task type
        if task_type == "B":
            # Type B needs spikes
            timeline, metadata = self.curve_synthesizer.generate_full_timeline(
                inject_spikes=True
            )
            spike_info = metadata.get('spikes', [])
        else:
            # Type A and C don't require spikes
            timeline, metadata = self.curve_synthesizer.generate_full_timeline(
                inject_spikes=False
            )
            spike_info = None
        
        # Generate text description
        if task_type == "B" and spike_info:
            prompt, target = self.text_generator.generate_type_b(
                timeline, metadata, spike_info=spike_info
            )
        else:
            prompt, target = self.text_generator.generate(
                task_type, timeline, metadata
            )
        
        # Tokenize
        # Prompt only (for input)
        prompt_encoded = self.tokenizer(
            prompt,
            max_length=self.max_length // 2,
            truncation=True,
            padding=False,
            return_tensors='pt'
        )
        prompt_ids = prompt_encoded['input_ids'].squeeze(0)
        
        # Target only (for loss computation)
        target_encoded = self.tokenizer(
            target,
            max_length=self.max_length // 2,
            truncation=True,
            padding=False,
            return_tensors='pt'
        )
        target_ids = target_encoded['input_ids'].squeeze(0)
        
        # Combined for attention mask
        combined_text = prompt + " " + target
        combined_encoded = self.tokenizer(
            combined_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Create labels: -100 for prompt, target_ids for target
        # Note: In forward pass, we'll prepend temporal embedding
        # So actual sequence is: [temporal_token] + prompt + target
        labels = torch.full_like(combined_encoded['input_ids'], -100)
        prompt_len = prompt_ids.size(0)
        labels[:, prompt_len:prompt_len + target_ids.size(0)] = target_ids
        
        return {
            'curve': timeline,
            'prompt_ids': prompt_ids,
            'target_ids': target_ids,
            'attention_mask': combined_encoded['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0),
            'task_type': task_type,
            'metadata': {
                'prompt': prompt,
                'target': target,
                'num_timesteps': metadata['num_timesteps'],
                'duration_s': metadata['duration_s']
            }
        }
    
    @classmethod
    def create_train_val_split(
        cls,
        num_train: int,
        num_val: int,
        tokenizer: AutoTokenizer,
        **kwargs
    ):
        """
        Create train and validation datasets.
        
        Args:
            num_train: Number of training samples
            num_val: Number of validation samples
            tokenizer: Tokenizer instance
            **kwargs: Additional arguments for dataset initialization
        
        Returns:
            (train_dataset, val_dataset)
        """
        train_dataset = cls(
            num_samples=num_train,
            tokenizer=tokenizer,
            seed=kwargs.pop('train_seed', 42),
            **kwargs
        )
        
        val_dataset = cls(
            num_samples=num_val,
            tokenizer=tokenizer,
            seed=kwargs.pop('val_seed', 43),
            **kwargs
        )
        
        return train_dataset, val_dataset


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    
    Handles variable-length curves by padding.
    
    Args:
        batch: List of samples from __getitem__
    
    Returns:
        Batched tensors
    """
    # Find max timesteps in batch
    max_timesteps = max(item['curve'].size(0) for item in batch)
    batch_size = len(batch)
    
    # Pad curves to max_timesteps
    curves = torch.zeros(batch_size, max_timesteps, 4)
    for i, item in enumerate(batch):
        T = item['curve'].size(0)
        curves[i, :T, :] = item['curve']
    
    # Find max prompt length
    max_prompt_len = max(item['prompt_ids'].size(0) for item in batch)
    prompt_ids = torch.zeros(batch_size, max_prompt_len, dtype=torch.long)
    for i, item in enumerate(batch):
        L = item['prompt_ids'].size(0)
        prompt_ids[i, :L] = item['prompt_ids']
    
    # Find max target length
    max_target_len = max(item['target_ids'].size(0) for item in batch)
    target_ids = torch.zeros(batch_size, max_target_len, dtype=torch.long)
    for i, item in enumerate(batch):
        L = item['target_ids'].size(0)
        target_ids[i, :L] = item['target_ids']
    
    # Attention masks (already padded to max_length)
    attention_masks = torch.stack([item['attention_mask'] for item in batch])
    
    # Labels (already padded)
    labels = torch.stack([item['labels'] for item in batch])
    
    # Task types
    task_types = [item['task_type'] for item in batch]
    
    return {
        'curve': curves,
        'prompt_ids': prompt_ids,
        'target_ids': target_ids,
        'attention_mask': attention_masks,
        'labels': labels,
        'task_type': task_types
    }
