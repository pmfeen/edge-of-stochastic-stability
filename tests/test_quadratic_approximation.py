"""
Test for quadratic approximation functionality.

This test verifies that the quadratic approximation works correctly by:
1. Training a simple model for a few steps 
2. Switching to quadratic approximation
3. Verifying the quadratic dynamics behave as expected
"""

import torch
import torch.nn as nn
import numpy as np
import os
from pathlib import Path
import sys

# Add parent directory to path to import training modules
sys.path.append(str(Path(__file__).parent.parent))

from utils.quadratic import QuadraticApproximation, flatten_params, set_model_params, compute_batch_grad_and_hvp, temporarily_set_params
from utils.nets import MLP, SquaredLoss
from utils.data import prepare_dataset


def test_quadratic_approximation_basic():
    """Test basic quadratic approximation functionality."""
    print("Testing basic quadratic approximation functionality...")
    
    # Setup
    device = torch.device('cpu')  # Use CPU for testing
    batch_size = 4
    input_dim = 2
    output_dim = 1
    
    # Create simple synthetic data
    torch.manual_seed(42)
    X = torch.randn(batch_size * 3, input_dim)
    Y = torch.randn(batch_size * 3, output_dim)
    
    # Create simple MLP
    net = MLP(input_dim=input_dim, hidden_dim=8, n_layers=2, output_dim=output_dim)
    loss_fn = SquaredLoss()
    
    # Initialize quadratic approximation
    anchor_step = 5
    quad_approx = QuadraticApproximation(net, loss_fn, device, anchor_step)
    
    # Simulate training steps
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    
    print("Running initial training steps...")
    for step in range(10):
        print(f"Step {step}, anchor_step: {anchor_step}, is_active: {quad_approx.is_active}")
        # Get batch
        start_idx = (step * batch_size) % len(X)
        end_idx = start_idx + batch_size
        if end_idx > len(X):
            end_idx = len(X)
            start_idx = end_idx - batch_size
        
        X_batch = X[start_idx:end_idx]
        Y_batch = Y[start_idx:end_idx]
        
        if step <= anchor_step:
            # Normal training
            optimizer.zero_grad()
            preds = net(X_batch)
            loss = loss_fn(preds, Y_batch)
            loss.backward()
            optimizer.step()
            
            # Check if we should initialize anchor
            if step == anchor_step:
                print(f"Attempting to initialize at step {step}")
            result = quad_approx.initialize_anchor(step, loss.item())
            if result:
                print(f"Successfully initialized at step {step}")
                
        else:
            # Quadratic approximation
            if not quad_approx.is_active:
                # Should have been initialized at step 5
                raise AssertionError("Quadratic approximation should be active by now")
            
            # Test quadratic gradient computation
            quad_gradient = quad_approx.compute_quadratic_gradient(X_batch, Y_batch)
            
            # Update delta
            current_lr = optimizer.param_groups[0]['lr']
            old_delta_norm = torch.norm(quad_approx.delta).item()
            quad_approx.update_delta(current_lr, quad_gradient)
            new_delta_norm = torch.norm(quad_approx.delta).item()
            
            print(f"Step {step}: Delta norm changed from {old_delta_norm:.6f} to {new_delta_norm:.6f}")
            
            # Set model to quadratic position
            current_params = quad_approx.get_current_params()
            set_model_params(net, current_params)
    
    print("✓ Basic quadratic approximation test passed")


def test_quadratic_loss_computation():
    """Test quadratic loss computation including the quadratic term."""
    print("\nTesting quadratic loss computation...")
    
    device = torch.device('cpu')
    
    # Create simple model and data
    torch.manual_seed(42)
    net = MLP(input_dim=2, hidden_dim=4, n_layers=1, output_dim=1)
    loss_fn = SquaredLoss()
    X = torch.randn(8, 2)
    Y = torch.randn(8, 1)
    
    # Initialize quadratic approximation
    quad_approx = QuadraticApproximation(net, loss_fn, device, anchor_step=0)
    
    # Get initial loss
    with torch.no_grad():
        initial_preds = net(X)
        initial_loss = loss_fn(initial_preds, Y).item()
    
    # Initialize anchor
    quad_approx.initialize_anchor(0, initial_loss)
    
    # Compute full gradient and HVP (with delta=0 initially)
    quad_approx.compute_full_gradient_and_hvp(X, Y)
    
    # At the anchor point (delta=0), quadratic loss should equal the original loss
    quad_loss_at_anchor = quad_approx.compute_quadratic_loss_for_logging(X, Y)
    
    print(f"Original loss: {initial_loss:.6f}")
    print(f"Quadratic loss at anchor (delta=0): {quad_loss_at_anchor:.6f}")
    
    # They should be very close (within numerical precision)
    assert abs(initial_loss - quad_loss_at_anchor) < 1e-5, f"Losses should match at anchor point: {initial_loss} vs {quad_loss_at_anchor}"
    
    # Now set a non-zero delta and test that quadratic term is included
    quad_approx.delta = torch.randn_like(quad_approx.delta) * 0.1
    
    quad_loss_with_delta = quad_approx.compute_quadratic_loss_for_logging(X, Y)
    
    # The quadratic loss should be different from the original loss
    # since we now have non-zero delta
    print(f"Quadratic loss with delta: {quad_loss_with_delta:.6f}")
    
    # Verify that the loss is different (since we have a non-zero delta)
    assert abs(quad_loss_with_delta - initial_loss) > 1e-6, "Quadratic loss should differ from original when delta != 0"
    
    print("✓ Quadratic loss computation test passed (including quadratic term)")


def test_parameter_preservation():
    """Test that model parameters are properly preserved during quadratic approximation computations."""
    print("\nTesting parameter preservation...")
    
    # Create model and data
    net = MLP(input_dim=10, hidden_dim=20, output_dim=2)
    X = torch.randn(32, 10)
    Y = torch.randint(0, 2, (32,))
    loss_fn = nn.CrossEntropyLoss()
    
    # Save original parameters
    original_params = flatten_params(net).detach().clone()
    
    # Initialize quadratic approximation
    quad_approx = QuadraticApproximation(net, loss_fn, torch.device('cpu'), anchor_step=0)
    
    # Compute initial loss
    output = net(X)
    initial_loss = loss_fn(output, Y).item()
    
    # Initialize anchor
    quad_approx.initialize_anchor(0, initial_loss)
    
    # Set a non-zero delta
    quad_approx.delta = torch.randn_like(quad_approx.delta) * 0.1
    
    # Compute quadratic gradient (should preserve parameters)
    quad_gradient = quad_approx.compute_quadratic_gradient(X[:8], Y[:8])  # Use batch
    
    # Check that model parameters are unchanged
    current_params = flatten_params(net).detach()
    params_unchanged = torch.allclose(original_params, current_params, atol=1e-10)
    
    print(f"Parameters unchanged after compute_quadratic_gradient: {params_unchanged}")
    assert params_unchanged, "Model parameters should be preserved after compute_quadratic_gradient"
    
    # Compute quadratic loss (should preserve parameters)
    quad_loss = quad_approx.compute_quadratic_loss_for_logging(X, Y)
    
    # Check that model parameters are still unchanged
    current_params = flatten_params(net).detach()
    params_unchanged = torch.allclose(original_params, current_params, atol=1e-10)
    
    print(f"Parameters unchanged after compute_quadratic_loss_for_logging: {params_unchanged}")
    assert params_unchanged, "Model parameters should be preserved after compute_quadratic_loss_for_logging"
    
    # Test the context manager directly
    anchor_params = quad_approx.anchor_params
    with temporarily_set_params(net, anchor_params):
        # Inside context: parameters should be anchor parameters
        inside_params = flatten_params(net).detach()
        anchor_match = torch.allclose(inside_params, anchor_params, atol=1e-10)
        print(f"Parameters match anchor inside context: {anchor_match}")
        assert anchor_match, "Parameters should match anchor inside context manager"
    
    # Outside context: parameters should be restored
    restored_params = flatten_params(net).detach()
    restored_match = torch.allclose(restored_params, original_params, atol=1e-10)
    print(f"Parameters properly restored after context: {restored_match}")
    assert restored_match, "Parameters should be restored after context manager exits"
    
    print("✓ Parameter preservation test passed")


def test_parameter_flattening():
    """Test parameter flattening and unflattening utilities."""
    print("\nTesting parameter flattening utilities...")
    
    # Create model
    net = MLP(input_dim=3, hidden_dim=5, n_layers=2, output_dim=2)
    
    # Get original parameters
    original_params = [p.data.clone() for p in net.parameters()]
    
    # Flatten and unflatten
    flat_params = flatten_params(net)
    print(f"Flattened parameter vector length: {flat_params.numel()}")
    
    # Modify the flattened parameters
    modified_flat = flat_params + 0.1
    
    # Set back to model
    set_model_params(net, modified_flat)
    
    # Check that all parameters were modified
    for orig_p, current_p in zip(original_params, net.parameters()):
        diff = torch.norm(current_p.data - orig_p).item()
        assert diff > 1e-6, "Parameters should have been modified"
    
    # Test round-trip
    new_flat = flatten_params(net)
    assert torch.allclose(new_flat, modified_flat), "Round-trip should preserve values"
    
    print("✓ Parameter flattening test passed")


def test_hvp_computation():
    """Test Hessian-vector product computation."""
    print("\nTesting Hessian-vector product computation...")
    
    # Simple quadratic function: f(x) = 0.5 * x^T A x + b^T x + c
    # where Hessian is A
    net = nn.Linear(2, 1, bias=False)
    
    # Set specific weights for predictable Hessian
    with torch.no_grad():
        net.weight.data = torch.tensor([[1.0, 2.0]])  # This affects the Hessian structure
    
    loss_fn = SquaredLoss()
    
    # Simple data
    X = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # Identity-like data
    Y = torch.tensor([[0.0], [0.0]])  # Target zeros
    
    # Get anchor params
    anchor_params = flatten_params(net)
    
    # Test vector
    delta = torch.tensor([0.1, 0.2])
    
    # Compute HvP
    g_B, H_B_delta = compute_batch_grad_and_hvp(net, X, Y, loss_fn, anchor_params, delta)
    
    print(f"Gradient shape: {g_B.shape}")
    print(f"HvP shape: {H_B_delta.shape}")
    print(f"Gradient: {g_B}")
    print(f"HvP: {H_B_delta}")
    
    # Basic sanity checks
    assert g_B.shape == anchor_params.shape, "Gradient should have same shape as parameters"
    assert H_B_delta.shape == anchor_params.shape, "HvP should have same shape as parameters"
    
    print("✓ HvP computation test passed")


if __name__ == "__main__":
    print("Running quadratic approximation tests...")
    print("=" * 50)
    
    try:
        test_parameter_flattening()
        test_hvp_computation()
        test_quadratic_loss_computation()
        test_quadratic_approximation_basic()
        
        print("\n" + "=" * 50)
        print("✅ All quadratic approximation tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)