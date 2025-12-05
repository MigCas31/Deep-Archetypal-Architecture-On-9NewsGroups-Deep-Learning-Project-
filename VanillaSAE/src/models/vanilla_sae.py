import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VanillaSAE(nn.Module):
    """
    Vanilla Sparse Autoencoder with L1 sparsity penalty.
    
    Architecture:
    - Encoder: Linear transformation with optional activation
    - Latent layer: Sparse representation
    - Decoder: Linear transformation to reconstruct input
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        activation: str = 'relu',
        sparsity_penalty: float = 0.01,
        tie_weights: bool = False,
        dropout_rate: float = 0.0,
        bias: bool = True
    ):
        """
        Initialize Vanilla SAE.
        
        Args:
            input_dim: Dimensionality of input data
            hidden_dim: Dimensionality of hidden/latent representation
            activation: Activation function ('relu', 'sigmoid', 'tanh', 'leaky_relu')
            sparsity_penalty: Weight for L1 sparsity penalty
            tie_weights: Whether to tie encoder and decoder weights (decoder = encoder^T)
            dropout_rate: Dropout rate for regularization
            bias: Whether to use bias terms
        """
        super(VanillaSAE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_penalty = sparsity_penalty
        self.tie_weights = tie_weights
        self.dropout_rate = dropout_rate
        
        # Encoder
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=bias)
        
        # Decoder
        if tie_weights:
            # Decoder uses transpose of encoder weights
            self.decoder_bias = nn.Parameter(torch.zeros(input_dim)) if bias else None
        else:
            self.decoder = nn.Linear(hidden_dim, input_dim, bias=bias)
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation: str):
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(), 
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
        }
        
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")
        
        return activations[activation]
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier/Glorot initialization."""
        # Initialize encoder weights
        nn.init.xavier_uniform_(self.encoder.weight)
        if self.encoder.bias is not None:
            nn.init.zeros_(self.encoder.bias)
        
        # Initialize decoder weights (if not tied)
        if not self.tie_weights:
            nn.init.xavier_uniform_(self.decoder.weight)
            if self.decoder.bias is not None:
                nn.init.zeros_(self.decoder.bias)
        else:
            if self.decoder_bias is not None:
                nn.init.zeros_(self.decoder_bias)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to hidden representation.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Hidden representation of shape (batch_size, hidden_dim)
        """
        hidden = self.encoder(x)
        hidden = self.activation(hidden)
        
        if self.dropout is not None and self.training:
            hidden = self.dropout(hidden)
            
        return hidden
    
    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Decode hidden representation to reconstruct input.
        
        Args:
            hidden: Hidden representation of shape (batch_size, hidden_dim)
            
        Returns:
            Reconstructed input of shape (batch_size, input_dim)
        """
        if self.tie_weights:
            # Use transpose of encoder weights for decoder
            reconstruction = F.linear(hidden, self.encoder.weight.t(), self.decoder_bias)
        else:
            reconstruction = self.decoder(hidden)
        
        return reconstruction
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (reconstruction, hidden_representation)
        """
        hidden = self.encode(x)
        reconstruction = self.decode(hidden)
        
        return reconstruction, hidden
    
    def compute_loss(
        self, 
        x: torch.Tensor, 
        reconstruction: torch.Tensor, 
        hidden: torch.Tensor,
        reduction: str = 'mean'
    ) -> dict:
        """
        Compute autoencoder loss with sparsity penalty.
        
        Args:
            x: Original input
            reconstruction: Reconstructed input
            hidden: Hidden representation
            reduction: How to reduce the loss ('mean', 'sum', 'none')
            
        Returns:
            Dictionary containing loss components
        """
        # Reconstruction loss (MSE)
        reconstruction_loss = F.mse_loss(reconstruction, x, reduction=reduction)
        
        # Sparsity penalty (L1 norm of hidden activations)
        sparsity_loss = torch.mean(torch.abs(hidden))
        
        # Total loss
        total_loss = reconstruction_loss + self.sparsity_penalty * sparsity_loss
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'sparsity_loss': sparsity_loss
        }
    
    def get_sparsity_stats(self, hidden: torch.Tensor) -> dict:
        """
        Compute sparsity statistics for hidden representations.
        
        Args:
            hidden: Hidden representation tensor
            
        Returns:
            Dictionary with sparsity statistics
        """
        with torch.no_grad():
            # Fraction of activations that are effectively zero (< 0.01)
            sparsity_ratio = (torch.abs(hidden) < 0.01).float().mean().item()
            
            # L1 norm (average absolute activation)
            l1_norm = torch.abs(hidden).mean().item()
            
            # L2 norm
            l2_norm = torch.norm(hidden, dim=-1).mean().item()
            
            # Maximum activation
            max_activation = torch.abs(hidden).max().item()
            
            # Standard deviation of activations
            activation_std = hidden.std().item()
            
        return {
            'sparsity_ratio': sparsity_ratio,
            'l1_norm': l1_norm,
            'l2_norm': l2_norm,
            'max_activation': max_activation,
            'activation_std': activation_std
        }
    
    def get_reconstruction_stats(self, x: torch.Tensor, reconstruction: torch.Tensor) -> dict:
        """
        Compute reconstruction quality statistics.
        
        Args:
            x: Original input
            reconstruction: Reconstructed input
            
        Returns:
            Dictionary with reconstruction statistics
        """
        with torch.no_grad():
            mse = F.mse_loss(reconstruction, x).item()
            mae = F.l1_loss(reconstruction, x).item()
            
            # Correlation coefficient
            x_flat = x.view(-1)
            recon_flat = reconstruction.view(-1)
            correlation = torch.corrcoef(torch.stack([x_flat, recon_flat]))[0, 1].item()
            
            # Relative error
            relative_error = (torch.norm(x - reconstruction) / torch.norm(x)).item()
            
        return {
            'mse': mse,
            'mae': mae,
            'correlation': correlation,
            'relative_error': relative_error
        }