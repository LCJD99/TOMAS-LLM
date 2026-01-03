import argparse
import yaml
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_config_with_args(config: Dict, args: argparse.Namespace) -> Dict:
    merged = config.copy()
    args_dict = vars(args)
    
    if args_dict.get('batch_size') is not None:
        merged['training']['batch_size'] = args_dict['batch_size']
    if args_dict.get('num_epochs') is not None:
        merged['training']['num_epochs'] = args_dict['num_epochs']
    if args_dict.get('lr') is not None:
        merged['training']['learning_rate'] = args_dict['lr']
    if args_dict.get('device') is not None:
        merged['training']['device'] = args_dict['device']
    if args_dict.get('output_dir') is not None:
        merged['output']['output_dir'] = args_dict['output_dir']
    if args_dict.get('log_wandb') is not None:
        merged['logging']['wandb']['enabled'] = args_dict['log_wandb']
    
    return merged