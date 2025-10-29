"""
DDPM wrapper for Edge of Stochastic Stability (EoSS) framework.

This module provides a wrapper around the DDPM model that exposes the loss computation
in a way that's compatible with EoSS measurements (λ_max, batch sharpness, GNI, etc.).

The key insight is that EoSS needs access to the computational graph and gradients
to compute Hessian-based measurements, which requires bypassing the DDPM Trainer's
internal loss computation and backward pass.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import sys
import os

# Add the DDPM package to the path
ddpm_path = os.path.join(os.path.dirname(__file__), '..', 'denoising-diffusion-pytorch')
if ddpm_path not in sys.path:
    sys.path.append(ddpm_path)

try:
    from denoising_diffusion_pytorch import Unet, GaussianDiffusion
except ImportError:
    raise ImportError("Could not import DDPM modules. Make sure denoising-diffusion-pytorch is properly installed.")


class DDPMWrapper(nn.Module):
    """
    Wrapper around DDPM model that exposes loss computation for EoSS measurements.
    
    This wrapper allows the EoSS framework to:
    1. Compute the DDPM loss with create_graph=True for Hessian measurements
    2. Access gradients before they're consumed by the optimizer
    3. Maintain the computational graph needed for λ_max, batch sharpness, etc.
    """
    
    def __init__(
        self,
        input_dim: Tuple[int, int, int],  # (C, H, W) for images
        output_dim: int,  # Not used for DDPM, but required by EoSS interface
        dim: int = 64,
        dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
        channels: int = 3,
        image_size: Tuple[int, int] = (32, 32),
        timesteps: int = 1000,
        beta_schedule: str = 'cosine',
        objective: str = 'pred_noise',
        **kwargs
    ):
        super().__init__()
        
        # Store configuration
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.channels = channels
        self.image_size = image_size
        self.timesteps = timesteps
        
        # Create UNet backbone
        self.unet = Unet(
            dim=dim,
            dim_mults=dim_mults,
            channels=channels,
            out_dim=channels,  # DDPM predicts noise/channels
            **kwargs
        )
        
        # Create diffusion model
        self.diffusion = GaussianDiffusion(
            self.unet,
            image_size=image_size,
            timesteps=timesteps,
            beta_schedule=beta_schedule,
            objective=objective
        )
        
        # Store the diffusion model's loss computation method
        self._p_losses = self.diffusion.p_losses
        
    def forward(self, x: torch.Tensor, return_loss_only: bool = True) -> torch.Tensor:
        """
        Forward pass that computes DDPM loss with computational graph preserved.
        
        Args:
            x: Input images of shape (batch_size, channels, height, width)
            return_loss_only: If True, return only the loss tensor. If False, return (loss, predictions)
            
        Returns:
            Loss tensor with computational graph preserved for EoSS measurements
        """
        # Ensure input is properly normalized for DDPM
        if x.min() >= 0 and x.max() <= 1:
            # Convert from [0, 1] to [-1, 1] if needed
            x = x * 2 - 1
        
        # Use the diffusion model's forward method which handles timestep sampling internally
        # This preserves the computational graph needed for EoSS measurements
        loss = self.diffusion(x)
        
        if return_loss_only:
            return loss
        else:
            # For cases where we need predictions (though DDPM doesn't really have "predictions")
            # We could return the model's noise prediction, but for EoSS we mainly need the loss
            return loss, None
    
    def sample(self, batch_size: int = 1) -> torch.Tensor:
        """
        Generate samples from the DDPM model.
        
        Args:
            batch_size: Number of samples to generate
            
        Returns:
            Generated images
        """
        return self.diffusion.sample(batch_size=batch_size)
    
    def get_model_parameters(self) -> list:
        """
        Get list of model parameters for EoSS measurements.
        
        Returns:
            List of parameter tensors
        """
        return list(self.parameters())
    
    def compute_loss_with_graph(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute loss with computational graph preserved for EoSS measurements.
        
        This is the key method that allows EoSS to compute Hessian-based measurements.
        
        Args:
            x: Input images
            
        Returns:
            Loss tensor with create_graph=True
        """
        return self.forward(x, return_loss_only=True)


class DDPMImageDataset:
    """
    Dataset wrapper for DDPM training with EoSS framework.
    
    This handles the conversion between image data and the format expected by EoSS.
    """
    
    def __init__(
        self,
        images: torch.Tensor,
        normalize_to_neg_one: bool = True
    ):
        """
        Initialize dataset with image data.
        
        Args:
            images: Tensor of shape (N, C, H, W) containing images
            normalize_to_neg_one: Whether to normalize images to [-1, 1] range
        """
        self.images = images
        
        if normalize_to_neg_one:
            # Normalize from [0, 1] to [-1, 1] if needed
            if images.min() >= 0 and images.max() <= 1:
                self.images = images * 2 - 1
        
        # Create dummy targets (DDPM doesn't use targets, but EoSS expects them)
        self.targets = torch.zeros(len(images), 1)
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.targets[idx]
    
    def get_data_tuple(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return data in the format expected by EoSS: (X_train, Y_train, X_test, Y_test)
        
        Returns:
            Tuple of (X_train, Y_train, X_test, Y_test) where test data is the same as train
        """
        return self.images, self.targets, self.images, self.targets


def create_ddpm_model(
    input_dim: Tuple[int, int, int],
    output_dim: int,
    **ddpm_kwargs
) -> DDPMWrapper:
    """
    Factory function to create a DDPM model compatible with EoSS.
    
    Args:
        input_dim: Input dimensions (C, H, W)
        output_dim: Output dimensions (not used for DDPM)
        **ddpm_kwargs: Additional arguments for DDPM model
        
    Returns:
        DDPMWrapper instance
    """
    return DDPMWrapper(
        input_dim=input_dim,
        output_dim=output_dim,
        **ddpm_kwargs
    )


def prepare_ddpm_dataset(
    images: torch.Tensor,
    train_split: float = 0.8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare image dataset for DDPM training with EoSS.
    
    Args:
        images: Tensor of shape (N, C, H, W) containing images
        train_split: Fraction of data to use for training
        
    Returns:
        Tuple of (X_train, Y_train, X_test, Y_test) in EoSS format
    """
    # Normalize images to [-1, 1] range
    if images.min() >= 0 and images.max() <= 1:
        images = images * 2 - 1
    
    # Split into train/test
    n_train = int(len(images) * train_split)
    X_train = images[:n_train]
    X_test = images[n_train:]
    
    # Create dummy targets (DDPM doesn't use targets)
    Y_train = torch.zeros(len(X_train), 1)
    Y_test = torch.zeros(len(X_test), 1)
    
    return X_train, Y_train, X_test, Y_test
