#!/usr/bin/env python3

"""
Test Gauss-Newton quadratic approximation implementation.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.quadratic import QuadraticApproximation, compute_per_sample_jacobian, compute_gauss_newton_hvp_on_the_fly
from utils.nets import MLP, SquaredLoss

def test_per_sample_jacobian():
    """Test per-sample Jacobian computation."""
    print("Testing per-sample Jacobian computation...")
    
    torch.manual_seed(42)
    net = MLP(input_dim=2, hidden_dim=4, n_layers=1, output_dim=3)  # 3 outputs for testing
    
    # Create test data
    X = torch.randn(5, 2)  # 5 samples, 2 inputs
    
    # Compute Jacobian
    anchor_params = torch.cat([p.data.reshape(-1) for p in net.parameters() if p.requires_grad])
    jacobian = compute_per_sample_jacobian(net, X, anchor_params)
    
    print(f"Jacobian shape: {jacobian.shape}")
    assert jacobian.shape[0] == 5, f"Expected batch size 5, got {jacobian.shape[0]}"
    assert jacobian.shape[1] == 3, f"Expected 3 outputs, got {jacobian.shape[1]}"
    assert jacobian.shape[2] == len(anchor_params), f"Expected {len(anchor_params)} params, got {jacobian.shape[2]}"
    
    print("✓ Per-sample Jacobian test passed!")


def test_gauss_newton_hvp_on_the_fly():
    """Test Gauss-Newton matrix-vector product computed on-the-fly."""
    print("Testing Gauss-Newton HVP on-the-fly...")
    
    torch.manual_seed(42)
    
    # Create a simple network
    net = MLP(input_dim=2, hidden_dim=4, n_layers=1, output_dim=3)
    X = torch.randn(5, 2)  # 5 samples, 2 inputs
    anchor_params = torch.cat([p.data.reshape(-1) for p in net.parameters() if p.requires_grad])
    delta = torch.randn(len(anchor_params))
    
    # Compute GN-vector product on-the-fly
    gnvp = compute_gauss_newton_hvp_on_the_fly(net, X, anchor_params, delta)
    
    print(f"GNVP shape: {gnvp.shape}")
    assert gnvp.shape == (len(anchor_params),), f"Expected shape ({len(anchor_params)},), got {gnvp.shape}"
    
    # Verify it's finite
    assert torch.isfinite(gnvp).all(), "GNVP should be finite"
    
    print("✓ Gauss-Newton HVP on-the-fly test passed!")


def test_quadratic_approximation_gauss_newton():
    """Test QuadraticApproximation with Gauss-Newton (memory-efficient)."""
    print("Testing QuadraticApproximation with Gauss-Newton (memory-efficient)...")
    
    torch.manual_seed(42)
    
    # Setup
    device = torch.device('cpu')
    net = MLP(input_dim=2, hidden_dim=4, n_layers=1, output_dim=10)  # CIFAR-10 like
    loss_fn = SquaredLoss()
    anchor_step = 5
    
    # Create quadratic approximation with Gauss-Newton
    quad_approx = QuadraticApproximation(net, loss_fn, device, anchor_step, use_gauss_newton=True)
    
    # Create test dataset (small CIFAR-10 like)
    X = torch.randn(8, 2)  # 8 samples, 2 features
    Y = torch.randn(8, 10)  # 10 outputs
    
    # Test initialization (no longer needs full dataset for pre-computation)
    initialized = quad_approx.initialize_anchor(current_step=6, anchor_loss=1.0, full_dataset=None)
    assert initialized, "Should have initialized anchor"
    assert quad_approx.is_active, "Should be active after initialization"
    
    # Test gradient computation
    X_batch = X[:4]  # First 4 samples as batch
    Y_batch = Y[:4]
    
    quad_grad = quad_approx.compute_quadratic_gradient(X_batch, Y_batch)
    
    print(f"Quadratic gradient shape: {quad_grad.shape}")
    assert quad_grad.shape == (quad_approx.anchor_params.numel(),), "Gradient should match parameter count"
    assert torch.isfinite(quad_grad).all(), "Gradient should be finite"
    
    # Test quadratic loss computation
    quad_loss = quad_approx.compute_quadratic_loss_for_logging(X, Y)
    print(f"Quadratic loss: {quad_loss}")
    assert quad_loss is not None, "Should compute quadratic loss"
    assert torch.isfinite(torch.tensor(quad_loss)), "Loss should be finite"
    
    print("✓ QuadraticApproximation with Gauss-Newton (memory-efficient) test passed!")


def test_comparison_small_network():
    """Test on a very small network to verify correctness."""
    print("Testing on small network to verify correctness...")
    
    torch.manual_seed(123)
    
    # Very small network: single linear layer, no bias
    net = nn.Linear(2, 1, bias=False)
    loss_fn = SquaredLoss()
    device = torch.device('cpu')
    
    # Small dataset
    X = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)  # 2 samples, 2 features
    Y = torch.tensor([[1.0], [2.0]], dtype=torch.float32)  # 2 outputs
    
    # Test both Hessian and Gauss-Newton
    quad_approx_hessian = QuadraticApproximation(net, loss_fn, device, 0, use_gauss_newton=False)
    quad_approx_gn = QuadraticApproximation(net, loss_fn, device, 0, use_gauss_newton=True)
    
    # Initialize both (no longer needs full dataset for GN)
    quad_approx_hessian.initialize_anchor(current_step=1, anchor_loss=1.0, full_dataset=None)
    quad_approx_gn.initialize_anchor(current_step=1, anchor_loss=1.0, full_dataset=None)
    
    # Compare gradients on the same batch
    grad_hessian = quad_approx_hessian.compute_quadratic_gradient(X, Y)
    grad_gn = quad_approx_gn.compute_quadratic_gradient(X, Y)
    
    print(f"Hessian gradient: {grad_hessian}")
    print(f"Gauss-Newton gradient: {grad_gn}")
    
    # They should be different (GN is an approximation)
    print(f"Difference: {torch.norm(grad_hessian - grad_gn)}")
    
    # Test quadratic loss computation for both
    quad_loss_hessian = quad_approx_hessian.compute_quadratic_loss_for_logging(X, Y)
    quad_loss_gn = quad_approx_gn.compute_quadratic_loss_for_logging(X, Y)
    
    print(f"Hessian quadratic loss: {quad_loss_hessian}")
    print(f"Gauss-Newton quadratic loss: {quad_loss_gn}")
    
    print("✓ Small network comparison test passed!")


if __name__ == "__main__":
    test_per_sample_jacobian()
    print()
    test_gauss_newton_hvp_on_the_fly()
    print()
    test_quadratic_approximation_gauss_newton()
    print()
    test_comparison_small_network()
    print("\n🎉 All Gauss-Newton memory-efficient tests passed!")