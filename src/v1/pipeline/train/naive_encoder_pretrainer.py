"""
NaiveEncoderPretrainer

Training pipeline for NaiveEncoderForPretraining using HuggingFace Trainer API.
Supports:
- Lightweight checkpoint saving (only new parameters)
- Resume from checkpoint
- W&B integration
- Mixed precision training
"""

import os
import logging
from typing import Dict, Optional
from pathlib import Path

import yaml
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from transformers.trainer_utils import get_last_checkpoint

from ...model.tool_encoder import NaiveEncoderForPretraining
from ...data.naive_pretrain_dataset import NaivePretrainDataset, collate_fn
from ...utils.checkpoint_utils import (
    save_new_parameters,
    load_new_parameters,
    find_latest_checkpoint,
    get_new_params_path,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NewParamsSaveCallback(TrainerCallback):
    """
    Callback to save only the new parameters at each checkpoint.
    """
    
    def __init__(self, model: NaiveEncoderForPretraining, checkpoint_dir: str):
        self.model = model
        self.checkpoint_dir = checkpoint_dir
    
    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Save new parameters when a checkpoint is saved.
        """
        # Get the current step
        step = state.global_step
        
        # Determine save path
        save_path = get_new_params_path(self.checkpoint_dir, step)
        
        # Save new parameters
        metadata = {
            "step": step,
            "epoch": state.epoch,
            "loss": state.log_history[-1].get("loss", None) if state.log_history else None,
        }
        
        save_new_parameters(
            model=self.model,
            save_path=save_path,
            original_vocab_size=self.model.original_vocab_size,
            num_new_tokens=self.model.num_new_tokens,
            metadata=metadata,
        )
        
        logger.info(f"Saved new parameters at step {step}")
        
        return control


class NaiveEncoderPretrainer:
    """
    Training pipeline for NaiveEncoderForPretraining.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the trainer.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.model = None
        self.dataset = None
        self.trainer = None
        
        # Setup
        self._setup_wandb()
        self._setup_model()
        self._setup_dataset()
        self._setup_trainer()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        logger.info(f"Loading configuration from: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _setup_wandb(self):
        """Setup W&B logging if enabled."""
        if self.config.get("wandb", {}).get("enabled", False):
            try:
                import wandb
                
                wandb_config = self.config["wandb"]
                
                # Initialize W&B
                wandb.init(
                    project=wandb_config.get("project", "tomas-llm"),
                    name=wandb_config.get("name", "naive_encoder_pretrain"),
                    tags=wandb_config.get("tags", []),
                    notes=wandb_config.get("notes", ""),
                    config=self.config,
                )
                
                logger.info("W&B logging enabled")
            except ImportError:
                logger.warning("wandb not installed. Disabling W&B logging.")
                self.config["training"]["report_to"] = []
        else:
            logger.info("W&B logging disabled")
            self.config["training"]["report_to"] = []
    
    def _setup_model(self):
        """Initialize the NaiveEncoderForPretraining model."""
        logger.info("Initializing NaiveEncoderForPretraining...")
        
        model_config = self.config["model"]
        
        self.model = NaiveEncoderForPretraining(
            llm_model_name=model_config["llm_model_name"],
            extended_tokenizer_path=model_config.get("extended_tokenizer_path"),
            combined_tokens_path=model_config.get("combined_tokens_path"),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        
        # Load previously saved new parameters if requested
        resume_config = self.config.get("resume", {})
        if resume_config.get("load_new_params", False):
            checkpoint_dir = self.config["paths"]["checkpoint_dir"]
            new_params_path = self.config["paths"].get("new_params_save_path")
            
            if new_params_path and os.path.exists(new_params_path):
                logger.info(f"Loading previously saved new parameters from: {new_params_path}")
                load_new_parameters(
                    model=self.model,
                    load_path=new_params_path,
                    device=self.model.device,
                )
        
        logger.info("Model initialization complete")
    
    def _setup_dataset(self):
        """Initialize the dataset."""
        logger.info("Initializing dataset...")
        
        data_config = self.config["data"]
        
        self.dataset = NaivePretrainDataset(
            data_path=data_config["train_data_path"],
            tokenizer=self.model.tokenizer,
            max_length=data_config.get("max_length", 512),
        )
        
        logger.info(f"Dataset loaded: {len(self.dataset)} samples")
    
    def _setup_trainer(self):
        """Initialize the HuggingFace Trainer."""
        logger.info("Initializing Trainer...")
        
        training_config = self.config["training"]
        
        # Create TrainingArguments
        training_args = TrainingArguments(
            output_dir=training_config["output_dir"],
            per_device_train_batch_size=training_config.get("per_device_train_batch_size", 8),
            gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 1),
            num_train_epochs=training_config.get("num_train_epochs", 3),
            learning_rate=training_config.get("learning_rate", 5e-5),
            weight_decay=training_config.get("weight_decay", 0.01),
            warmup_steps=training_config.get("warmup_steps", 100),
            max_grad_norm=training_config.get("max_grad_norm", 1.0),
            lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
            logging_dir=training_config.get("logging_dir", "logs"),
            logging_steps=training_config.get("logging_steps", 10),
            logging_first_step=training_config.get("logging_first_step", True),
            save_strategy=training_config.get("save_strategy", "steps"),
            save_steps=training_config.get("save_steps", 500),
            save_total_limit=training_config.get("save_total_limit", 3),
            eval_strategy=training_config.get("eval_strategy", "no"),
            fp16=training_config.get("fp16", True) and torch.cuda.is_available(),
            dataloader_pin_memory=training_config.get("dataloader_pin_memory", True),
            dataloader_num_workers=training_config.get("dataloader_num_workers", 4),
            remove_unused_columns=training_config.get("remove_unused_columns", False),
            report_to=training_config.get("report_to", []),
            seed=training_config.get("seed", 42),
        )
        
        # Create Trainer
        self.trainer = Trainer(
            model=self.model.llm_model,  # Pass the underlying LLM model
            args=training_args,
            train_dataset=self.dataset,
        )
            # data_collator=collate_fn
        
        # Add custom callback to save new parameters
        new_params_callback = NewParamsSaveCallback(
            model=self.model,
            checkpoint_dir=training_config["output_dir"],
        )
        self.trainer.add_callback(new_params_callback)
        
        logger.info("Trainer initialization complete")
    
    def train(self, resume_from_checkpoint: Optional[str] = None):
        """
        Start training.
        
        Args:
            resume_from_checkpoint: Path to checkpoint to resume from, or "latest"
        """
        logger.info("=" * 80)
        logger.info("Starting Training")
        logger.info("=" * 80)
        
        # Handle resume from checkpoint
        if resume_from_checkpoint == "latest":
            checkpoint_dir = self.config["training"]["output_dir"]
            resume_from_checkpoint = find_latest_checkpoint(checkpoint_dir)
            if resume_from_checkpoint:
                logger.info(f"Resuming from latest checkpoint: {resume_from_checkpoint}")
            else:
                logger.info("No checkpoint found. Starting from scratch.")
                resume_from_checkpoint = None
        elif resume_from_checkpoint:
            logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
        
        # Start training
        try:
            self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            logger.info("Training completed successfully!")
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        
        # Save final new parameters
        logger.info("Saving final new parameters...")
        final_save_path = self.config["paths"]["new_params_save_path"]
        
        save_new_parameters(
            model=self.model,
            save_path=final_save_path,
            original_vocab_size=self.model.original_vocab_size,
            num_new_tokens=self.model.num_new_tokens,
            metadata={
                "final": True,
                "total_steps": self.trainer.state.global_step,
                "total_epochs": self.trainer.state.epoch,
            },
        )
        
        logger.info(f"Final parameters saved to: {final_save_path}")
        logger.info("=" * 80)
    
    def save_model(self, output_dir: Optional[str] = None):
        """
        Save the complete model and tokenizer.
        
        Args:
            output_dir: Directory to save the model (default: from config)
        """
        if output_dir is None:
            output_dir = os.path.join(
                self.config["training"]["output_dir"],
                "final_model"
            )
        
        logger.info(f"Saving complete model to: {output_dir}")
        
        # Save the model
        self.trainer.save_model(output_dir)
        
        # Save the tokenizer
        self.model.tokenizer.save_pretrained(output_dir)
        
        logger.info("Model saved successfully")


def main():
    """
    Main training script entry point.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Train NaiveEncoderForPretraining")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pretrain_naive_encoder.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint (path or 'latest')",
    )
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = NaiveEncoderPretrainer(config_path=args.config)
    
    # Start training
    trainer.train(resume_from_checkpoint=args.resume)
    
    # Save final model
    trainer.save_model()
    
    logger.info("Training pipeline completed!")


if __name__ == "__main__":
    main()
