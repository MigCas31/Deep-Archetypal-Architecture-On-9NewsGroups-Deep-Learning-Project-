import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')

from src.models.vanilla_sae import VanillaSAE
from src.data.dataset import create_dataloaders
from src.utils.config import load_config

def load_model_from_state_dict(state_dict_path, config_path, device):
    config = load_config(config_path)
    
    # Get input dimension from data
    train_loader, _, _ = create_dataloaders(
        data_path=config['data']['data_path'],
        batch_size=1,
        test_size=config['data'].get('test_size', 0.2),
        val_size=config['data'].get('val_size', 0.1),
        random_state=config['data'].get('random_state', 42)
    )
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
    
    # Load state dict
    model.load_state_dict(torch.load(state_dict_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, config

def evaluate_and_plot(model, data_loader, device, save_dir):
    model.eval()
    
    all_inputs = []
    all_reconstructions = []
    all_hidden = []
    all_losses = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i > 50: 
                break
                
            x = batch.to(device)
            reconstruction, hidden = model(x)
            loss_dict = model.compute_loss(x, reconstruction, hidden)
            
            all_inputs.append(x.cpu())
            all_reconstructions.append(reconstruction.cpu())
            all_hidden.append(hidden.cpu())
            all_losses.append(loss_dict['total_loss'].item())
    
    # Concatenate results
    inputs = torch.cat(all_inputs, dim=0)
    reconstructions = torch.cat(all_reconstructions, dim=0)
    hidden = torch.cat(all_hidden, dim=0)
    
    print(f"Data shapes: Input {inputs.shape}, Hidden {hidden.shape}")
    
    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    # Sample reconstructions
    plt.figure(figsize=(12, 8))
    n_samples = min(3, len(inputs))
    for i in range(n_samples):
        plt.plot(inputs[i][:50].numpy(), label=f'Original {i+1}', alpha=0.7)
        plt.plot(reconstructions[i][:50].numpy(), '--', label=f'Recon {i+1}', alpha=0.7)
    plt.title('Sample Reconstructions (first 50 features)', fontsize=14)
    plt.xlabel('Feature Index')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / 'sample_reconstructions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Reconstruction error distribution
    plt.figure(figsize=(10, 6))
    errors = (inputs - reconstructions).pow(2).mean(dim=1).numpy()
    plt.hist(errors, bins=30, alpha=0.7, color='skyblue')
    plt.title('Reconstruction Error Distribution', fontsize=14)
    plt.xlabel('MSE')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(save_dir / 'reconstruction_error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Sparsity pattern
    plt.figure(figsize=(10, 6))
    sparsity_per_sample = (torch.abs(hidden) < 0.01).float().mean(dim=1).numpy()
    plt.hist(sparsity_per_sample, bins=30, alpha=0.7, color='coral')
    plt.title('Sparsity Distribution', fontsize=14)
    plt.xlabel('Fraction of Near-Zero Activations')
    plt.ylabel('Number of Samples')
    plt.tight_layout()
    plt.savefig(save_dir / 'sparsity_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature activation heatmap
    plt.figure(figsize=(12, 8))
    n_samples_show = min(50, hidden.shape[0])
    n_features_show = min(50, hidden.shape[1])
    heatmap_data = hidden[:n_samples_show, :n_features_show].numpy()
    plt.imshow(heatmap_data.T, aspect='auto', cmap='viridis')
    plt.title('Hidden Activations Heatmap (subset)', fontsize=14)
    plt.xlabel('Sample Index')
    plt.ylabel('Feature Index')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_dir / 'feature_activation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    
    # Reconstruction quality vs sparsity
    plt.figure(figsize=(10, 6))
    plt.scatter(sparsity_per_sample, errors, alpha=0.5)
    plt.title('Reconstruction Error vs Sparsity', fontsize=14)
    plt.xlabel('Sparsity Ratio')
    plt.ylabel('Reconstruction Error')
    plt.tight_layout()
    plt.savefig(save_dir / 'reconstruction_vs_sparsity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Loss over batches
    plt.figure(figsize=(10, 6))
    plt.plot(all_losses, linewidth=2)
    plt.title('Loss per Batch', fontsize=14)
    plt.xlabel('Batch Index')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig(save_dir / 'loss_per_batch.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Activation statistics
    plt.figure(figsize=(10, 6))
    activation_stats = [
        hidden.mean().item(),
        hidden.std().item(), 
        hidden.min().item(),
        hidden.max().item()
    ]
    plt.bar(['Mean', 'Std', 'Min', 'Max'], activation_stats, color=['skyblue', 'lightgreen', 'coral', 'gold'])
    plt.title('Hidden Activation Statistics', fontsize=14)
    plt.ylabel('Value')
    for i, v in enumerate(activation_stats):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(save_dir / 'activation_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Summary metrics
    plt.figure(figsize=(10, 6))
    mse = errors.mean()
    mae = torch.abs(inputs - reconstructions).mean().item()
    sparsity = sparsity_per_sample.mean()
    
    metrics = [mse, mae, sparsity]
    labels = ['MSE', 'MAE', 'Sparsity']
    colors = ['red', 'orange', 'blue']
    
    bars = plt.bar(labels, metrics, color=colors, alpha=0.7)
    plt.title('Summary Metrics', fontsize=14)
    plt.ylabel('Value')
    
    # Add value labels on bars
    for bar, metric in zip(bars, metrics):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{metric:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'summary_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
 
    
    # Print summary
    # Save summary to text file
    summary_path = save_dir / 'evaluation_summary.txt'
    with open(summary_path, 'w') as f:
         f.write("EVALUATION SUMMARY\n")
         f.write("="*50 + "\n")
         f.write(f"Number of samples: {len(inputs)}\n")
         f.write(f"Input dimension: {inputs.shape[1]}\n")
         f.write(f"Hidden dimension: {hidden.shape[1]}\n")
         f.write(f"Compression ratio: {inputs.shape[1] / hidden.shape[1]:.2f}\n")
         f.write(f"Average loss: {np.mean(all_losses):.6f}\n")
         f.write(f"Reconstruction MSE: {mse:.6f}\n")
         f.write(f"Reconstruction MAE: {mae:.6f}\n")
         f.write(f"Mean correlation: {corr:.4f}\n")
         f.write(f"Mean sparsity: {sparsity:.4f}\n")
         f.write("="*50 + "\n")
         
        #print("\n" + "="*50)
        #print("EVALUATION SUMMARY")
        #print("="*50)
        #print(f"Number of samples: {len(inputs)}")
        #print(f"Input dimension: {inputs.shape[1]}")
        #print(f"Hidden dimension: {hidden.shape[1]}")
        #print(f"Compression ratio: {inputs.shape[1] / hidden.shape[1]:.2f}")
        #print(f"Average loss: {np.mean(all_losses):.6f}")
        #print(f"Reconstruction MSE: {mse:.6f}")
        #print(f"Reconstruction MAE: {mae:.6f}")
        #print(f"Mean correlation: {corr:.4f}")
        #print(f"Mean sparsity: {sparsity:.4f}")
        #print("="*50)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test different models
    models_to_test = [
        ('results_DL/best_model.pt', 'Default Model (Best)'),
        ('results_DL/final_model.pt', 'Default Model (Final)'),
        ('results_DL/high_sparsity/best_model.pt', 'High Sparsity (Best)'),
        ('results_DL/large_model/best_model.pt', 'Large Model (Best)')
    ]
    
    config_path = 'configs/default.yaml'
    
    for model_path, model_name in models_to_test:
        if Path(model_path).exists():
            print(f"Evaluating {model_name}: {model_path}")
            
            try:
                # Load model
                model, config = load_model_from_state_dict(model_path, config_path, device)
                
                # Create data loader
                _, _, test_loader = create_dataloaders(
                    data_path=config['data']['data_path'],
                    batch_size=32,
                    test_size=config['data'].get('test_size', 0.2),
                    val_size=config['data'].get('val_size', 0.1),
                    random_state=config['data'].get('random_state', 42)
                )
                
                # Create save directory
                save_dir = Path('evaluation_results') / model_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
                
                # Evaluate
                evaluate_and_plot(model, test_loader, device, save_dir)
                
            except Exception as e:
                print(f"Failed to evaluate {model_name}: {e}")
        else:
            print(f"Model not found: {model_path}")

if __name__ == "__main__":
    main()