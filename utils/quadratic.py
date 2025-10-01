"""
Quadratic approximation utilities for switching from true NN dynamics to 
quadratic Taylor approximation dynamics.

Based on the procedure outlined in docs/quadratic_approx.md
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from contextlib import contextmanager
from utils.nets import SquaredLoss


def flatten_params(model: nn.Module) -> torch.Tensor:
    """Flatten all model parameters into a single vector."""
    return torch.cat([p.data.reshape(-1) for p in model.parameters() if p.requires_grad])


def unflatten_params(flat_params: torch.Tensor, model: nn.Module) -> List[torch.Tensor]:
    """Unflatten a parameter vector into shapes matching the model parameters."""
    chunks = []
    start_idx = 0
    for p in model.parameters():
        if p.requires_grad:
            numel = p.numel()
            chunks.append(flat_params[start_idx:start_idx + numel].view_as(p))
            start_idx += numel
    return chunks


def set_model_params(model: nn.Module, flat_params: torch.Tensor) -> None:
    """Set model parameters from a flattened parameter vector."""
    start_idx = 0
    for p in model.parameters():
        if p.requires_grad:
            numel = p.numel()
            p.data.copy_(flat_params[start_idx:start_idx + numel].view_as(p))
            start_idx += numel


def compute_per_sample_jacobian(
    net: nn.Module,
    X: torch.Tensor,
    anchor_params: torch.Tensor
) -> torch.Tensor:
    """
    Compute per-sample per-output Jacobian of the model outputs w.r.t. parameters.
    
    Args:
        net: Neural network model
        X: Input data [batch_size, input_dim]
        anchor_params: Anchor parameters θ_0
        
    Returns:
        Jacobian tensor of shape [batch_size, num_outputs, num_params]
        For CIFAR-10, this would be [batch_size, 10, num_params]
    """
    with temporarily_set_params(net, anchor_params):
        batch_size = X.size(0)
        outputs = net(X)  # Shape: [batch_size, num_outputs]
        num_outputs = outputs.size(1)
        
        params = [p for p in net.parameters() if p.requires_grad]
        num_params = sum(p.numel() for p in params)
        
        # Initialize Jacobian tensor
        jacobian = torch.zeros(batch_size, num_outputs, num_params, device=X.device)
        
        # For each sample and each output, compute gradient w.r.t. parameters
        for i in range(batch_size):
            for j in range(num_outputs):
                # Zero gradients
                for p in params:
                    if p.grad is not None:
                        p.grad.zero_()
                
                # Compute gradient of outputs[i, j] w.r.t. parameters
                outputs[i, j].backward(retain_graph=True)
                
                # Collect gradients into jacobian
                start_idx = 0
                for p in params:
                    if p.grad is not None:
                        grad_flat = p.grad.reshape(-1)
                        end_idx = start_idx + grad_flat.numel()
                        jacobian[i, j, start_idx:end_idx] = grad_flat.detach()
                        start_idx = end_idx
        
        return jacobian.detach()


def compute_gauss_newton_hvp_on_the_fly(
    net: nn.Module,
    X: torch.Tensor,
    anchor_params: torch.Tensor,
    delta: torch.Tensor
) -> torch.Tensor:
    """
    Compute Gauss-Newton matrix-vector product by computing Jacobian on-the-fly.
    
    For MSE loss: GN = (1/batch_size) * sum_i J_i^T J_i
    GN * delta = (1/batch_size) * sum_i J_i^T (J_i * delta)
    
    Args:
        net: Neural network model
        X: Input batch [batch_size, input_dim]
        anchor_params: Anchor parameters θ_0
        delta: Parameter displacement vector
        
    Returns:
        Gauss-Newton matrix-vector product
    """    
    with temporarily_set_params(net, anchor_params):
        batch_size = X.size(0)
        outputs = net(X)  # Shape: [batch_size, num_outputs]
        num_outputs = outputs.size(1)
        
        params = [p for p in net.parameters() if p.requires_grad]
        num_params = sum(p.numel() for p in params)
        
        # Compute GN * delta on-the-fly without storing full Jacobian
        gn_delta = torch.zeros(num_params, device=X.device)
        
        # For each sample in the batch
        for i in range(batch_size):
            # For each output component
            for j in range(num_outputs):
                # Zero gradients
                for p in params:
                    if p.grad is not None:
                        p.grad.zero_()
                
                # Compute gradient of outputs[i, j] w.r.t. parameters
                outputs[i, j].backward(retain_graph=True)
                
                # Collect gradients into jacobian row
                jacobian_row = torch.cat([p.grad.reshape(-1) for p in params if p.grad is not None])
                
                # Compute J_{i,j} * delta (scalar)
                jv = torch.dot(jacobian_row, delta)
                
                # Add J_{i,j}^T * (J_{i,j} * delta) to accumulator
                gn_delta += jv * jacobian_row.detach()
        
        # Average over batch
        return gn_delta / batch_size


def compute_batch_jacobian_and_gnvp(
    net: nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    loss_fn: nn.Module,
    anchor_params: torch.Tensor,
    delta: torch.Tensor,
    full_jacobian: torch.Tensor = None,
    batch_indices: torch.Tensor = None,
    use_gauss_newton: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute batch gradient and Gauss-Newton vector product or Hessian-vector product.
    
    Args:
        net: Neural network model
        X: Input batch
        Y: Target batch  
        loss_fn: Loss function
        anchor_params: Flattened anchor parameters θ_0
        delta: Current displacement Δ = θ - θ_0
        full_jacobian: Pre-computed Jacobian for full dataset (deprecated, ignored)
        batch_indices: Indices of current batch samples (deprecated, ignored)
        use_gauss_newton: Whether to use Gauss-Newton instead of Hessian
        
    Returns:
        Tuple of (g_B, GN_B * delta) or (g_B, H_B * delta)
    """
    # Verify we're using MSE loss when using Gauss-Newton
    if use_gauss_newton and not isinstance(loss_fn, SquaredLoss):
        raise ValueError("Gauss-Newton approximation only supported for SquaredLoss (MSE)")
    
    with temporarily_set_params(net, anchor_params):
        batch_size = X.size(0)
        
        # Forward pass and loss computation
        output = net(X)
        loss = loss_fn(output, Y)
        
        # Get parameters that require gradients
        params = [p for p in net.parameters() if p.requires_grad]
        
        # Compute gradient g_B
        grad_list = torch.autograd.grad(loss, params, create_graph=not use_gauss_newton)
        g_B = torch.cat([g.reshape(-1) for g in grad_list])
        
        if use_gauss_newton:
            # Use Gauss-Newton approximation computed on-the-fly
            gnvp = compute_gauss_newton_hvp_on_the_fly(net, X, anchor_params, delta)
            
            return g_B.detach(), gnvp.detach()
        else:
            # Compute Hessian-vector product H_B * delta (original implementation)
            delta_list = unflatten_params(delta, net)
            
            # Compute dot product of gradients with delta for second-order derivative
            dot_product = sum((g * d).sum() for g, d in zip(grad_list, delta_list))
            
            # Compute second derivatives (Hessian-vector product)
            hvp_list = torch.autograd.grad(dot_product, params, retain_graph=False)
            H_B_delta = torch.cat([h.reshape(-1) for h in hvp_list])
            
            return g_B.detach(), H_B_delta.detach()


@contextmanager
def temporarily_set_params(model: nn.Module, temp_params: torch.Tensor):
    """
    Context manager to temporarily set model parameters and automatically restore them.
    
    Args:
        model: Neural network model
        temp_params: Temporary flattened parameters to set
        
    Usage:
        with temporarily_set_params(model, anchor_params):
            # model now has anchor_params
            output = model(X)
        # model automatically restored to original parameters
    """
    # Save current parameters
    original_params = flatten_params(model).detach().clone()
    
    try:
        # Set temporary parameters
        set_model_params(model, temp_params)
        yield
    finally:
        # Always restore original parameters
        set_model_params(model, original_params)


def compute_batch_grad_and_hvp(
    net: nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    loss_fn: nn.Module,
    anchor_params: torch.Tensor,
    delta: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DEPRECATED: Use compute_batch_jacobian_and_gnvp instead.
    
    Compute batch gradient g_B and Hessian-vector product H_B * delta at anchor point.
    Model parameters are automatically restored after computation.
    
    Args:
        net: Neural network model
        X: Input batch
        Y: Target batch  
        loss_fn: Loss function
        anchor_params: Flattened anchor parameters θ_0
        delta: Current displacement Δ = θ - θ_0
        
    Returns:
        Tuple of (g_B, H_B * delta) both as flattened tensors
    """
    # Use new function with Hessian mode
    return compute_batch_jacobian_and_gnvp(
        net, X, Y, loss_fn, anchor_params, delta, 
        full_jacobian=None, batch_indices=None, use_gauss_newton=False
    )


def compute_quadratic_loss(
    full_gradient: torch.Tensor,
    full_hessian_delta: torch.Tensor,
    delta: torch.Tensor,
    anchor_loss: float
) -> float:
    """
    Compute quadratic loss: L_quad(θ) = L(θ_0) + g^T * Δ + 0.5 * Δ^T * H * Δ
    
    Args:
        full_gradient: Full-batch gradient g at anchor point
        full_hessian_delta: Full-batch Hessian-vector product H * Δ
        delta: Current displacement Δ = θ - θ_0
        anchor_loss: Loss value L(θ_0) at anchor point
        
    Returns:
        Quadratic loss value
    """
    linear_term = torch.dot(full_gradient, delta).item()
    quadratic_term = 0.5 * torch.dot(delta, full_hessian_delta).item()
    return anchor_loss + linear_term + quadratic_term


def compute_quadratic_loss_gauss_newton_on_the_fly(
    net: nn.Module,
    X: torch.Tensor,
    anchor_params: torch.Tensor,
    full_gradient: torch.Tensor,
    delta: torch.Tensor,
    anchor_loss: float
) -> float:
    """
    Compute quadratic loss using Gauss-Newton approximation computed on-the-fly: 
    L_quad(θ) = L(θ_0) + g^T * Δ + 0.5 * Δ^T * GN * Δ
    
    where GN = (1/N) * sum_i J_i^T J_i is the Gauss-Newton matrix.
    
    Args:
        net: Neural network model
        X: Full dataset inputs
        anchor_params: Anchor parameters θ_0
        full_gradient: Full-batch gradient g at anchor point
        delta: Current displacement Δ = θ - θ_0
        anchor_loss: Loss value L(θ_0) at anchor point
        
    Returns:
        Quadratic loss value using Gauss-Newton approximation
    """
    linear_term = torch.dot(full_gradient, delta).item()
    
    # Compute GN * delta on-the-fly to save memory
    gn_delta = compute_gauss_newton_hvp_on_the_fly(net, X, anchor_params, delta)
    
    quadratic_term = 0.5 * torch.dot(delta, gn_delta).item()
    return anchor_loss + linear_term + quadratic_term


class QuadraticApproximation:
    """
    Manager class for quadratic approximation dynamics.
    
    Handles the switching from true NN dynamics to quadratic Taylor approximation
    at a specified anchor point θ_0. Supports both Hessian and Gauss-Newton approximations.
    """
    
    def __init__(
        self,
        net: nn.Module,
        loss_fn: nn.Module,
        device: torch.device,
        anchor_step: int,
        use_gauss_newton: bool = False
    ):
        self.net = net
        self.loss_fn = loss_fn
        self.device = device
        self.anchor_step = anchor_step
        self.use_gauss_newton = use_gauss_newton
        
        # State variables
        self.is_active = False
        self.anchor_params = None
        self.anchor_loss = None
        self.delta = None
        
        # Gauss-Newton specific state (no longer storing pre-computed Jacobian)
        
    def initialize_anchor(self, current_step: int, anchor_loss: float, 
                         full_dataset: Tuple[torch.Tensor, torch.Tensor] = None) -> bool:
        """
        Initialize the anchor point if we've reached the switch step.
        
        Args:
            current_step: Current training step
            anchor_loss: Loss value at current step (deprecated, will be computed later)
            full_dataset: (X, Y) tuple for full dataset (no longer used, kept for compatibility)
            
        Returns:
            True if anchor was initialized, False otherwise
        """
        if current_step >= self.anchor_step and not self.is_active:
            # Freeze anchor parameters θ_0
            self.anchor_params = flatten_params(self.net).detach().clone()
            self.anchor_loss = None # will be computed later
            
            # Initialize delta to zero
            self.delta = torch.zeros_like(self.anchor_params)
            
            self.is_active = True
            approximation_type = "Gauss-Newton (on-the-fly)" if self.use_gauss_newton else "Hessian"
            print(f"{approximation_type} quadratic approximation activated at step {current_step}")
            return True
        return False
        
    def compute_full_gradient_and_hvp(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """
        DEPRECATED: Use compute_quadratic_loss_for_logging instead.
        This method is kept for backward compatibility but should not be used.
        """
        # This method is now deprecated since we compute everything directly
        # in compute_quadratic_loss_for_logging to ensure mathematical correctness
        # Parameters are kept for backward compatibility
        _ = X, Y  # Suppress unused parameter warnings
    
    def compute_quadratic_gradient(self, X: torch.Tensor, Y: torch.Tensor, 
                                  batch_indices: torch.Tensor = None) -> torch.Tensor:
        """
        Compute the quadratic approximation gradient: g_B + H_B * Δ or g_B + GN_B * Δ
        
        Args:
            X: Input batch
            Y: Target batch
            batch_indices: Indices of current batch samples (deprecated, ignored)
            
        Returns:
            Quadratic gradient as flattened tensor
        """
        if not self.is_active:
            raise RuntimeError("Quadratic approximation not active")
            
        g_B, matrix_delta = compute_batch_jacobian_and_gnvp(
            self.net, X, Y, self.loss_fn, 
            self.anchor_params, self.delta, 
            None, None, self.use_gauss_newton
        )
        
        return g_B + matrix_delta
    
    def update_delta(self, learning_rate: float, quad_gradient: torch.Tensor) -> None:
        """
        Update delta using SGD: Δ_{t+1} = Δ_t - η * (g_B + H_B * Δ_t)
        
        Args:
            learning_rate: Learning rate η
            quad_gradient: Quadratic gradient g_B + H_B * Δ
        """
        if not self.is_active:
            raise RuntimeError("Quadratic approximation not active")
            
        self.delta = self.delta - learning_rate * quad_gradient
    
    def get_current_params(self) -> torch.Tensor:
        """Get current parameter vector θ = θ_0 + Δ"""
        if not self.is_active:
            return flatten_params(self.net)
        return self.anchor_params + self.delta
    
    def compute_quadratic_loss_for_logging(self, X: torch.Tensor, Y: torch.Tensor) -> Optional[float]:
        """
        Compute the full quadratic loss for logging: L_quad(θ) = L(θ_0) + g^T*Δ + 0.5*Δ^T*H*Δ
        where H is the Hessian matrix at the anchor point θ_0.
        Model parameters are automatically restored after computation.
        
        Args:
            X: Full dataset inputs  
            Y: Full dataset targets
            
        Returns:
            Quadratic loss value using Hessian, or None if not active
        """
        if not self.is_active:
            return None
            
        # Compute gradient at anchor point
        with temporarily_set_params(self.net, self.anchor_params):
            output = self.net(X)
            loss = self.loss_fn(output, Y)

            if self.anchor_loss is None:
                self.anchor_loss = loss.item()

            params = [p for p in self.net.parameters() if p.requires_grad]
            
            # Compute full gradient at anchor
            grad_list = torch.autograd.grad(loss, params, create_graph=True)
            full_gradient = torch.cat([g.reshape(-1) for g in grad_list])
            
            # Compute Hessian-vector product H*Δ at anchor point
            delta_list = unflatten_params(self.delta, self.net)
            dot_product = sum((g * d).sum() for g, d in zip(grad_list, delta_list))
            hvp_list = torch.autograd.grad(dot_product, params, retain_graph=False)
            hessian_delta = torch.cat([h.reshape(-1) for h in hvp_list])
            
            # Compute quadratic approximation using Hessian
            return compute_quadratic_loss(
                full_gradient, hessian_delta, self.delta, self.anchor_loss
            )
    
    def compute_quadratic_loss_gauss_newton_for_logging(self, X: torch.Tensor, Y: torch.Tensor) -> Optional[float]:
        """
        Compute the full quadratic loss for logging using Gauss-Newton approximation: 
        L_quad(θ) = L(θ_0) + g^T*Δ + 0.5*Δ^T*GN*Δ
        where GN is the Gauss-Newton matrix at the anchor point θ_0.
        The GN matrix is computed on-the-fly without constructing the full matrix.
        Model parameters are automatically restored after computation.
        
        Args:
            X: Full dataset inputs  
            Y: Full dataset targets
            
        Returns:
            Quadratic loss value using Gauss-Newton approximation, or None if not active
        """
        if not self.is_active:
            return None
            
        # Verify we're using MSE loss when using Gauss-Newton
        if not isinstance(self.loss_fn, SquaredLoss):
            raise ValueError("Gauss-Newton approximation only supported for SquaredLoss (MSE)")
            
        # Compute gradient at anchor point
        with temporarily_set_params(self.net, self.anchor_params):
            output = self.net(X)
            loss = self.loss_fn(output, Y)

            if self.anchor_loss is None:
                self.anchor_loss = loss.item()

            params = [p for p in self.net.parameters() if p.requires_grad]
            
            # Compute full gradient at anchor
            grad_list = torch.autograd.grad(loss, params, create_graph=False)
            full_gradient = torch.cat([g.reshape(-1) for g in grad_list])
            
            # Use Gauss-Newton approximation for quadratic loss computed on-the-fly
            return compute_quadratic_loss_gauss_newton_on_the_fly(
                self.net, X, self.anchor_params, full_gradient, self.delta, self.anchor_loss
            )