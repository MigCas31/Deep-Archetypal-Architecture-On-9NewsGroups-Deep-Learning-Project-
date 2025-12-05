import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from models.vanilla_sae import VanillaSAE
from data.dataset import create_dataloaders
from utils.config import load_config, parse_args
from utils.wandb_logger import WandbLogger
from utils.trainer import train_model


def main():
    # Parse arguments
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path=config['data']['data_path'],
        batch_size=config['training']['batch_size'],
        test_size=config['data'].get('test_size', 0.2),
        val_size=config['data'].get('val_size', 0.1),
        random_state=config['data'].get('random_state', 42)
    )
    
    print(f"Data loaded - Train: {len(train_loader.dataset)}, "
          f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Get input dimension from first batch
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch.shape[1]
    
    # Create model
    model = VanillaSAE(
        input_dim=input_dim,
        hidden_dim=config['model']['hidden_dim'],
        activation=config['model']['activation'],
        sparsity_penalty=config['model']['sparsity_penalty'],
        tie_weights=config['model'].get('tie_weights', False),
        dropout_rate=config['model'].get('dropout_rate', 0.0)
    )
    
    print(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Initialize wandb logger
    wandb_logger = None
    if config.get('logging', {}).get('use_wandb', True) and not args.no_wandb:
        wandb_logger = WandbLogger(
            project_name=config['experiment']['project'],
            experiment_name=config['experiment']['name'],
            config=config
        )
        print("Initialized wandb logging")
    
    # Train model
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=config['training']['epochs'],
        lr=config['training']['learning_rate'],
        wandb_logger=wandb_logger,
        save_dir=config['experiment']['save_dir']
    )
    
    # Save final model in results directory
    from pathlib import Path
    save_path = Path(config['experiment']['save_dir'])
    save_path.mkdir(parents=True, exist_ok=True)
    final_model_path = save_path / 'final_model.pt'
    torch.save(trained_model.state_dict(), final_model_path)
    print(f"Training completed! Final model saved as '{final_model_path}'")
    
    # Finish wandb
    if wandb_logger:
        wandb_logger.finish()


if __name__ == "__main__":
    main()