import torch
import numpy as np
import anndata as ad
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Tuple


class NewsGroupsDataset(Dataset):
    
    def __init__(self, data: np.ndarray):
        """
        Args:
            data: preprocessed data array (already normalized)
        """
        self.data = torch.from_numpy(data.astype(np.float32))
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def load_data(data_path: str) -> np.ndarray:
    adata = ad.read_h5ad(data_path)
    
    # Get data matrix
    if hasattr(adata.X, 'toarray'):
        data = adata.X.toarray()  
    else:
        data = adata.X
    
    print(f"Loaded data with shape: {data.shape}")
    print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")
    print(f"Data mean: {data.mean():.3f}, std: {data.std():.3f}")
    
    return data


def create_dataloaders(
    data_path: str,
    batch_size: int = 64,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    num_workers: int = 0,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_path: Path to the h5ad file
        batch_size: Batch size
        test_size: Proportion for test set
        val_size: Proportion of remaining data for validation
        random_state: Random seed
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Load preprocessed data
    data = load_data(data_path)
    
    # Split data
    train_val_data, test_data = train_test_split(
        data, test_size=test_size, random_state=random_state
    )
    
    train_data, val_data = train_test_split(
        train_val_data, test_size=val_size, random_state=random_state
    )
    
    print(f"Split sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Create datasets
    train_dataset = NewsGroupsDataset(train_data)
    val_dataset = NewsGroupsDataset(val_data)
    test_dataset = NewsGroupsDataset(test_data)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader