"""
PyTorch Dataset for naive encoder pretraining.

This dataset:
1. Loads training data from JSONL
2. Tokenizes using extended tokenizer
3. Implements label masking (user_prompt = -100, response = token_ids)
4. Returns batches ready for training
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NaivePretrainDataset(Dataset):
    """
    Dataset for naive encoder pretraining.
    
    Implements teacher forcing with label masking:
    - user_prompt tokens are masked with -100 (not used in loss)
    - response tokens are kept for learning
    
    This teaches the model to generate combined_token from description.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        cache_encodings: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to naive_pretrain_data.jsonl
            tokenizer: Extended tokenizer with combined tokens
            max_length: Maximum sequence length
            cache_encodings: Whether to cache tokenized data
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_encodings = cache_encodings
        
        # Load data
        self.examples = self._load_data()
        
        # Cache for tokenized data
        self._cache = {} if cache_encodings else None
        
        logger.info(f"Initialized dataset with {len(self.examples)} examples")
        logger.info(f"Max length: {max_length}")
        logger.info(f"Caching: {cache_encodings}")
    
    def _load_data(self) -> List[Dict]:
        """Load training examples from JSONL file."""
        logger.info(f"Loading data from: {self.data_path}")
        
        examples = []
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line.strip())
                    examples.append(entry)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping line {line_num}: {e}")
                    continue
        
        logger.info(f"Loaded {len(examples)} examples")
        return examples
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.examples)
    
    def _create_labels(
        self,
        full_input_ids: torch.Tensor,
        user_prompt_length: int
    ) -> torch.Tensor:
        """
        Create labels with masking.
        
        Args:
            full_input_ids: Full sequence token IDs
            user_prompt_length: Length of user prompt (to be masked)
            
        Returns:
            Labels tensor with user_prompt masked as -100
        """
        labels = full_input_ids.clone()
        
        # Mask user_prompt tokens (they don't contribute to loss)
        labels[:user_prompt_length] = -100
        
        return labels
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example.
        
        Args:
            idx: Example index
            
        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        # Check cache
        if self._cache is not None and idx in self._cache:
            return self._cache[idx]
        
        example = self.examples[idx]
        
        # Extract components
        user_prompt = example['user_prompt']
        response = example['response']
        full_prompt = example['full_prompt']
        
        # Tokenize full prompt
        full_encoding = self.tokenizer(
            full_prompt,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Tokenize user_prompt to get its length
        user_encoding = self.tokenizer(
            user_prompt,
            add_special_tokens=False,  # Don't add special tokens for length calc
            truncation=False,
        )
        user_length = len(user_encoding['input_ids'])
        
        # Create labels (mask user_prompt, keep response)
        labels = self._create_labels(
            full_encoding['input_ids'].squeeze(0),
            user_length
        )
        
        # Prepare output
        output = {
            'input_ids': full_encoding['input_ids'].squeeze(0),
            'attention_mask': full_encoding['attention_mask'].squeeze(0),
            'labels': labels
        }
        
        # Cache if enabled
        if self._cache is not None:
            self._cache[idx] = output
        
        return output
    
    def get_example_info(self, idx: int) -> Dict:
        """
        Get human-readable info for an example.
        
        Args:
            idx: Example index
            
        Returns:
            Dictionary with example details
        """
        example = self.examples[idx]
        data = self[idx]
        
        # Decode tokens
        input_text = self.tokenizer.decode(data['input_ids'])
        
        # Find non-masked tokens in labels
        non_masked_indices = (data['labels'] != -100).nonzero(as_tuple=True)[0]
        response_tokens = data['input_ids'][non_masked_indices]
        response_text = self.tokenizer.decode(response_tokens)
        
        return {
            'index': idx,
            'combined_token': example['combined_token'],
            'user_prompt': example['user_prompt'],
            'response': example['response'],
            'full_prompt': example['full_prompt'],
            'input_ids_length': len(data['input_ids']),
            'num_masked_tokens': (data['labels'] == -100).sum().item(),
            'num_response_tokens': len(non_masked_indices),
            'decoded_input': input_text,
            'decoded_response': response_text
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of examples from dataset
        
    Returns:
        Batched tensors
    """
    # Stack tensors
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


def test_dataset():
    """Test the dataset implementation."""
    from transformers import AutoTokenizer
    
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent
    
    data_path = project_root / "data" / "generated" / "naive_pretrain_data.jsonl"
    tokenizer_path = project_root / "data" / "generated" / "extended_tokenizer"
    
    logger.info("=" * 80)
    logger.info("Testing NaivePretrainDataset")
    logger.info("=" * 80)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        trust_remote_code=True
    )
    
    # Create dataset
    dataset = NaivePretrainDataset(
        data_path=str(data_path),
        tokenizer=tokenizer,
        max_length=64
    )
    
    # Test a few examples
    logger.info("\n" + "=" * 80)
    logger.info("Sample Examples")
    logger.info("=" * 80)
    
    for i in range(min(3, len(dataset))):
        info = dataset.get_example_info(i)
        
        logger.info(f"\nExample {i + 1}:")
        logger.info(f"  Combined token: {info['combined_token']}")
        logger.info(f"  User prompt: {info['user_prompt'][:60]}...")
        logger.info(f"  Response: {info['response']}")
        logger.info(f"  Input length: {info['input_ids_length']}")
        logger.info(f"  Masked tokens: {info['num_masked_tokens']}")
        logger.info(f"  Response tokens: {info['num_response_tokens']}")
        logger.info(f"  Decoded response: {info['decoded_response']}")
    
    # Test DataLoader
    logger.info("\n" + "=" * 80)
    logger.info("Testing DataLoader")
    logger.info("=" * 80)
    
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    batch = next(iter(dataloader))
    
    logger.info(f"Batch shapes:")
    logger.info(f"  input_ids: {batch['input_ids'].shape}")
    logger.info(f"  attention_mask: {batch['attention_mask'].shape}")
    logger.info(f"  labels: {batch['labels'].shape}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ Dataset test completed successfully")
    logger.info("=" * 80)


if __name__ == "__main__":
    test_dataset()
