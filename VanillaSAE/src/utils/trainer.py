import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from tqdm import tqdm
from pathlib import Path
import time

from .wandb_logger import WandbLogger


def train_model(
    model: torch.nn.Module,
    train_loader,
    val_loader, 
    device: torch.device,
    epochs: int = 50,
    lr: float = 0.001,
    wandb_logger=None,
    save_dir: str = "results"
):
    """
    Simple training function for Vanilla SAE.
    
    Args:
        model: SAE model to train
        train_loader: Training data loader
        val_loader: Validation data loader 
        device: Device to train on
        epochs: Number of epochs
        lr: Learning rate
        wandb_logger: Optional wandb logger
        save_dir: Directory to save models (default: "results")
        
    Returns:
        Trained model
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Create save directory if it doesn't exist
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    best_model_path = save_path / "best_model.pt"
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = batch.to(device)
            
            optimizer.zero_grad()
            reconstruction, hidden = model(batch)
            loss_dict = model.compute_loss(batch, reconstruction, hidden)
            loss = loss_dict['total_loss']
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    reconstruction, hidden = model(batch)
                    loss_dict = model.compute_loss(batch, reconstruction, hidden)
                    val_loss += loss_dict['total_loss'].item()
            
            val_loss /= len(val_loader)
            
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # Log to wandb
            if wandb_logger:
                wandb_logger.log_metrics({
                    'epoch': epoch,
                    'train/loss': train_loss,
                    'val/loss': val_loss
                }, step=epoch)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved to {best_model_path}")
        else:
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}")
            
            # Log to wandb
            if wandb_logger:
                wandb_logger.log_metrics({
                    'epoch': epoch,
                    'train/loss': train_loss
                }, step=epoch)
    
    print(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    
    # Load best model
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from {best_model_path}")
    
    return model
