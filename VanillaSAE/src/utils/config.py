import yaml
import argparse


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Vanilla SAE Training")
    parser.add_argument('--config', type=str, default='configs/default.yaml', 
                       help='Path to config file')
    parser.add_argument('--no-wandb', action='store_true', 
                       help='Disable wandb logging')
    return parser.parse_args()