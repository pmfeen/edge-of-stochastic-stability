"""
Comprehensive test for quadratic approximation implementation with explicit Hessian validation.

This test validates the quadratic approximation dynamics by:
1. Creating small networks where we can compute exact per-sample Hessians
2. Comparing the quadratic dynamics against explicitly computed gradients and Hessians
3. Testing both individual components and full dynamics over several steps
4. Investigating the negative loss issue with mathematical rigor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.quadratic import (
    QuadraticApproximation, 
    flatten_params, 
    unflatten_params,
    compute_batch_grad_and_hvp,
    compute_quadratic_loss,
    set_model_params,
    temporarily_set_params
)


class TinyMLP(nn.Module):
    """Tiny MLP for explicit Hessian computation."""
    def __init__(self, input_dim: int = 2, hidden_dim: int = 3, output_dim: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=True)
        
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc2(h)


class LinearNet(nn.Module):
    """Simple linear network for easier explicit computation."""
    def __init__(self, input_dim: int = 2, output_dim: int = 1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)
        
    def forward(self, x):
        return self.linear(x)


def compute_explicit_hessian(model: nn.Module, X: torch.Tensor, Y: torch.Tensor, 
                           loss_fn: nn.Module) -> torch.Tensor:
    """
    Compute the exact Hessian matrix explicitly using double differentiation.
    Only works for small networks due to computational cost.
    """
    model.zero_grad()
    
    # Get flattened parameters
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    
    # Compute loss
    output = model(X)
    loss = loss_fn(output, Y)
    
    # Compute first derivatives
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat_grad = torch.cat([g.reshape(-1) for g in grads])
    
    # Compute Hessian
    hessian = torch.zeros(n_params, n_params)
    
    for i in range(n_params):
        # Compute second derivatives with respect to i-th parameter
        grad2 = torch.autograd.grad(flat_grad[i], params, retain_graph=True, allow_unused=True)
        
        hessian_row = []
        for g in grad2:
            if g is not None:
                hessian_row.append(g.reshape(-1))
            else:
                # Handle unused parameters - find the corresponding parameter size
                for p in params:
                    if p.grad is None or not torch.allclose(p.grad, torch.zeros_like(p.grad)):
                        continue
                    hessian_row.append(torch.zeros(p.numel()))
        
        if hessian_row:
            hessian[i] = torch.cat(hessian_row)
    
    return hessian


def compute_explicit_per_sample_gradients_and_hessians(
    model: nn.Module, X: torch.Tensor, Y: torch.Tensor, loss_fn: nn.Module
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Compute per-sample gradients and Hessians explicitly.
    Returns lists of gradients and Hessians for each sample.
    """
    per_sample_grads = []
    per_sample_hessians = []
    
    batch_size = X.size(0)
    
    for i in range(batch_size):
        x_i = X[i:i+1]  # Keep batch dimension
        y_i = Y[i:i+1]
        
        # Compute gradient for sample i
        model.zero_grad()
        output = model(x_i)
        loss = loss_fn(output, y_i)
        
        params = [p for p in model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(loss, params, create_graph=True)
        flat_grad = torch.cat([g.reshape(-1) for g in grads])
        per_sample_grads.append(flat_grad.detach())
        
        # Compute Hessian for sample i
        hessian = compute_explicit_hessian(model, x_i, y_i, loss_fn)
        per_sample_hessians.append(hessian)
    
    return per_sample_grads, per_sample_hessians


def test_batch_grad_and_hvp():
    """Test batch gradient and Hessian-vector product computation."""
    print("Testing batch gradient and Hessian-vector product computation...")
    
    # Setup
    torch.manual_seed(42)
    model = LinearNet(input_dim=2, output_dim=1)
    loss_fn = nn.MSELoss(reduction='mean')
    
    # Small dataset
    X = torch.randn(3, 2)
    Y = torch.randn(3, 1)
    
    # Random anchor point and delta
    anchor_params = flatten_params(model) + 0.1 * torch.randn_like(flatten_params(model))
    delta = 0.05 * torch.randn_like(anchor_params)
    
    # Compute using our implementation
    g_B, H_B_delta = compute_batch_grad_and_hvp(model, X, Y, loss_fn, anchor_params, delta)
    
    # Compute explicitly
    # Temporarily set model to anchor
    original_params = flatten_params(model).clone()
    set_model_params(model, anchor_params)
    
    try:
        # Explicit computation
        per_sample_grads, per_sample_hessians = compute_explicit_per_sample_gradients_and_hessians(
            model, X, Y, loss_fn
        )
        
        # Average gradients (batch gradient)
        explicit_g_B = torch.stack(per_sample_grads).mean(dim=0)
        
        # Average Hessian-vector products
        explicit_H_B_delta = torch.stack([H @ delta for H in per_sample_hessians]).mean(dim=0)
        
        # Compare results
        grad_diff = torch.norm(g_B - explicit_g_B).item()
        hvp_diff = torch.norm(H_B_delta - explicit_H_B_delta).item()
        
        print(f"Gradient difference: {grad_diff:.2e}")
        print(f"HVP difference: {hvp_diff:.2e}")
        
        assert grad_diff < 1e-5, f"Gradient computation error: {grad_diff}"
        assert hvp_diff < 1e-5, f"HVP computation error: {hvp_diff}"
        print("✓ Batch gradient and HVP computation test passed")
        
    finally:
        # Restore original parameters
        set_model_params(model, original_params)


def test_quadratic_loss_computation():
    """Test quadratic loss computation."""
    print("\nTesting quadratic loss computation...")
    
    # Setup
    torch.manual_seed(42)
    model = LinearNet(input_dim=2, output_dim=1)
    loss_fn = nn.MSELoss(reduction='mean')
    
    # Small dataset
    X = torch.randn(4, 2)
    Y = torch.randn(4, 1)
    
    # Set anchor point
    anchor_params = flatten_params(model).clone()
    delta = 0.1 * torch.randn_like(anchor_params)
    
    # Compute anchor loss
    with temporarily_set_params(model, anchor_params):
        output = model(X)
        anchor_loss = loss_fn(output, Y).item()
    
    # Compute using our implementation
    with temporarily_set_params(model, anchor_params):
        # Get gradient and Hessian at anchor
        output = model(X)
        loss = loss_fn(output, Y)
        
        params = [p for p in model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(loss, params, create_graph=True)
        full_gradient = torch.cat([g.reshape(-1) for g in grads])
        
        # HVP
        delta_list = unflatten_params(delta, model)
        dot_product = sum((g * d).sum() for g, d in zip(grads, delta_list))
        hvp_list = torch.autograd.grad(dot_product, params, retain_graph=False)
        full_hessian_delta = torch.cat([h.reshape(-1) for h in hvp_list])
    
    quad_loss = compute_quadratic_loss(full_gradient, full_hessian_delta, delta, anchor_loss)
    
    # Compute explicitly
    set_model_params(model, anchor_params)
    per_sample_grads, per_sample_hessians = compute_explicit_per_sample_gradients_and_hessians(
        model, X, Y, loss_fn
    )
    
    explicit_full_grad = torch.stack(per_sample_grads).mean(dim=0)
    explicit_full_hessian = torch.stack(per_sample_hessians).mean(dim=0)
    
    explicit_quad_loss = (anchor_loss + 
                        torch.dot(explicit_full_grad, delta).item() + 
                        0.5 * torch.dot(delta, explicit_full_hessian @ delta).item())
    
    loss_diff = abs(quad_loss - explicit_quad_loss)
    print(f"Quadratic loss difference: {loss_diff:.2e}")
    print(f"Our computation: {quad_loss:.6f}")
    print(f"Explicit computation: {explicit_quad_loss:.6f}")
    
    assert loss_diff < 1e-5, f"Quadratic loss computation error: {loss_diff}"
    print("✓ Quadratic loss computation test passed")


def test_quadratic_dynamics():
    """Test full quadratic dynamics over several steps."""
    print("\nTesting quadratic dynamics over multiple steps...")
    
    # Setup
    torch.manual_seed(42)
    model = LinearNet(input_dim=2, output_dim=1)
    loss_fn = nn.MSELoss(reduction='mean')
    device = torch.device('cpu')
    
    # Small dataset
    batch_size = 4
    X = torch.randn(8, 2)
    Y = torch.randn(8, 1)
    
    # Initialize quadratic approximation
    anchor_step = 2
    quad_approx = QuadraticApproximation(model, loss_fn, device, anchor_step)
    
    # Training parameters
    lr = 0.01
    n_steps = 6
    
    # Track results
    our_deltas = []
    explicit_deltas = []
    
    # Initialize explicit tracking
    explicit_delta = None
    anchor_params_explicit = None
    explicit_full_grad = None
    explicit_full_hessian = None
    
    for step in range(n_steps):
        # Sample a mini-batch
        indices = torch.randperm(X.size(0))[:batch_size]
        X_batch = X[indices]
        Y_batch = Y[indices]
        
        # Initialize anchor if needed
        if step == anchor_step:
            # Our implementation
            with temporarily_set_params(model, flatten_params(model)):
                output = model(X)
                anchor_loss = loss_fn(output, Y).item()
            quad_approx.initialize_anchor(step, anchor_loss)
            
            # Explicit computation setup
            anchor_params_explicit = flatten_params(model).clone()
            explicit_delta = torch.zeros_like(anchor_params_explicit)
            
            # Compute full gradient and Hessian at anchor for explicit method
            set_model_params(model, anchor_params_explicit)
            
            per_sample_grads, per_sample_hessians = compute_explicit_per_sample_gradients_and_hessians(
                model, X, Y, loss_fn
            )
            explicit_full_grad = torch.stack(per_sample_grads).mean(dim=0)
            explicit_full_hessian = torch.stack(per_sample_hessians).mean(dim=0)
        
        if quad_approx.is_active:
            # Our implementation
            quad_gradient = quad_approx.compute_quadratic_gradient(X_batch, Y_batch)
            quad_approx.update_delta(lr, quad_gradient)
            our_deltas.append(quad_approx.delta.clone())
            
            # Explicit computation
            # Compute batch gradient and Hessian at anchor
            set_model_params(model, anchor_params_explicit)
            
            per_sample_grads_batch, per_sample_hessians_batch = compute_explicit_per_sample_gradients_and_hessians(
                model, X_batch, Y_batch, loss_fn
            )
            
            explicit_batch_grad = torch.stack(per_sample_grads_batch).mean(dim=0)
            explicit_batch_hessian = torch.stack(per_sample_hessians_batch).mean(dim=0)
            
            # Update delta explicitly
            explicit_quad_grad = explicit_batch_grad + explicit_batch_hessian @ explicit_delta
            explicit_delta = explicit_delta - lr * explicit_quad_grad
            explicit_deltas.append(explicit_delta.clone())
    
    # Compare results
    if our_deltas and explicit_deltas:
        print(f"Comparing {len(our_deltas)} steps of quadratic dynamics...")
        
        for i, (our_delta, explicit_delta) in enumerate(zip(our_deltas, explicit_deltas)):
            delta_diff = torch.norm(our_delta - explicit_delta).item()
            print(f"Step {anchor_step + i + 1}: Delta difference = {delta_diff:.2e}")
            
            # Allow for small numerical differences
            assert delta_diff < 1e-4, f"Delta difference too large at step {i}: {delta_diff}"
        
        print("✓ Quadratic dynamics test passed")
    else:
        print("No quadratic steps to compare")


def test_negative_loss_investigation():
    """Test specifically for the negative loss issue with detailed mathematical analysis."""
    print("\nInvestigating negative loss issue...")
    
    # Setup that might trigger negative losses
    torch.manual_seed(123)
    model = TinyMLP(input_dim=2, hidden_dim=4, output_dim=1)
    loss_fn = nn.MSELoss(reduction='mean')
    device = torch.device('cpu')
    
    # Initialize with larger values to potentially trigger instability
    for p in model.parameters():
        p.data.normal_(0, 0.5)
    
    # Dataset
    X = torch.randn(6, 2)
    Y = torch.randn(6, 1)
    
    # Higher learning rate to potentially trigger issues
    lr = 0.1
    anchor_step = 1
    n_steps = 10
    
    quad_approx = QuadraticApproximation(model, loss_fn, device, anchor_step)
    
    print("Running dynamics that might cause negative losses...")
    
    # Store anchor computation details for analysis
    anchor_params = None
    anchor_loss_true = None
    full_gradient_at_anchor = None
    full_hessian_at_anchor = None
    
    for step in range(n_steps):
        # Mini-batch
        indices = torch.randperm(X.size(0))[:3]
        X_batch = X[indices]
        Y_batch = Y[indices]
        
        if step == anchor_step:
            # Compute true loss at anchor
            anchor_params = flatten_params(model).clone()
            with temporarily_set_params(model, anchor_params):
                output = model(X)
                anchor_loss_true = loss_fn(output, Y).item()
            
            quad_approx.initialize_anchor(step, anchor_loss_true)
            print(f"Anchor loss: {anchor_loss_true:.6f}")
            
            # Compute explicit full gradient and Hessian for analysis
            set_model_params(model, anchor_params)
            per_sample_grads, per_sample_hessians = compute_explicit_per_sample_gradients_and_hessians(
                model, X, Y, loss_fn
            )
            full_gradient_at_anchor = torch.stack(per_sample_grads).mean(dim=0)
            full_hessian_at_anchor = torch.stack(per_sample_hessians).mean(dim=0)
            
            # Analyze Hessian properties
            eigenvals = torch.linalg.eigvals(full_hessian_at_anchor).real
            min_eigval = eigenvals.min().item()
            max_eigval = eigenvals.max().item()
            print(f"Hessian eigenvalues range: [{min_eigval:.6f}, {max_eigval:.6f}]")
            
            if min_eigval < 0:
                print(f"⚠️  Hessian has negative eigenvalues! Quadratic approximation may be invalid.")
        
        if quad_approx.is_active:
            # Update using quadratic dynamics
            quad_gradient = quad_approx.compute_quadratic_gradient(X_batch, Y_batch)
            quad_approx.update_delta(lr, quad_gradient)
            
            # Compute quadratic loss
            quad_loss = quad_approx.compute_quadratic_loss_for_logging(X, Y)
            
            # Compute true loss at current parameters
            current_params = quad_approx.get_current_params()
            original_params = flatten_params(model).clone()
            
            set_model_params(model, current_params)
            output = model(X)
            true_loss = loss_fn(output, Y).item()
            set_model_params(model, original_params)
            
            # Compute explicit quadratic loss for verification
            if full_gradient_at_anchor is not None and full_hessian_at_anchor is not None:
                delta = quad_approx.delta
                explicit_quad_loss = (anchor_loss_true + 
                                    torch.dot(full_gradient_at_anchor, delta).item() + 
                                    0.5 * torch.dot(delta, full_hessian_at_anchor @ delta).item())
                
                quad_diff = abs(quad_loss - explicit_quad_loss) if quad_loss is not None else float('inf')
                
                print(f"Step {step}: Quad loss = {quad_loss:.6f}, True loss = {true_loss:.6f}, "
                      f"Explicit quad = {explicit_quad_loss:.6f}, Diff = {quad_diff:.2e}")
            else:
                print(f"Step {step}: Quad loss = {quad_loss:.6f}, True loss = {true_loss:.6f}")
            
            # Check for negative quadratic loss
            if quad_loss is not None and quad_loss < 0:
                print(f"🚨 Negative quadratic loss detected at step {step}: {quad_loss:.6f}")
                
                # Detailed analysis
                delta_norm = torch.norm(quad_approx.delta).item()
                linear_term = torch.dot(full_gradient_at_anchor, quad_approx.delta).item()
                quadratic_term = 0.5 * torch.dot(quad_approx.delta, full_hessian_at_anchor @ quad_approx.delta).item()
                
                print(f"  Delta norm: {delta_norm:.6f}")
                print(f"  Anchor loss: {anchor_loss_true:.6f}")
                print(f"  Linear term: {linear_term:.6f}")
                print(f"  Quadratic term: {quadratic_term:.6f}")
                print(f"  Sum: {anchor_loss_true + linear_term + quadratic_term:.6f}")
                
                # Check if quadratic term dominates and is negative
                if quadratic_term < 0 and abs(quadratic_term) > abs(linear_term) + anchor_loss_true:
                    print(f"  💡 Negative quadratic term dominates due to negative curvature!")
    
    print("✓ Negative loss investigation completed")


def test_small_network_comprehensive():
    """Comprehensive test with a very small network for complete mathematical verification."""
    print("\nRunning comprehensive test with tiny network...")
    
    # Extremely small network for complete traceability
    torch.manual_seed(42)
    model = nn.Linear(1, 1, bias=False)  # Single parameter
    loss_fn = nn.MSELoss(reduction='mean')
    
    # Single data point for simplicity
    X = torch.tensor([[2.0]])
    Y = torch.tensor([[1.0]])
    
    # Set initial parameter
    with torch.no_grad():
        model.weight.data = torch.tensor([[0.5]])
    
    print(f"Initial parameter: {model.weight.data.item():.6f}")
    
    # Compute everything analytically for this simple case
    # Loss = (w*x - y)^2 = (w*2 - 1)^2 = (2w - 1)^2
    # Gradient = 2*(2w - 1)*2 = 4*(2w - 1) = 8w - 4
    # Hessian = 8
    
    w_init = model.weight.data.item()
    analytical_loss = (2*w_init - 1)**2
    analytical_grad = 8*w_init - 4
    analytical_hessian = 8.0
    
    print(f"Analytical: loss={analytical_loss:.6f}, grad={analytical_grad:.6f}, hessian={analytical_hessian:.6f}")
    
    # Test our implementation
    anchor_params = flatten_params(model)
    delta = torch.tensor([0.1])  # Small perturbation
    
    g_B, H_B_delta = compute_batch_grad_and_hvp(model, X, Y, loss_fn, anchor_params, delta)
    
    print(f"Our implementation: grad={g_B.item():.6f}, hvp={H_B_delta.item():.6f}")
    print(f"Expected HVP: {analytical_hessian * delta.item():.6f}")
    
    # Check accuracy
    grad_error = abs(g_B.item() - analytical_grad)
    hvp_error = abs(H_B_delta.item() - analytical_hessian * delta.item())
    
    print(f"Gradient error: {grad_error:.2e}")
    print(f"HVP error: {hvp_error:.2e}")
    
    assert grad_error < 1e-6, f"Gradient error too large: {grad_error}"
    assert hvp_error < 1e-6, f"HVP error too large: {hvp_error}"
    
    print("✓ Small network comprehensive test passed")


def run_all_tests():
    """Run all quadratic approximation validation tests."""
    print("=" * 80)
    print("Comprehensive Quadratic Approximation Validation Tests")
    print("=" * 80)
    
    try:
        test_small_network_comprehensive()
        test_batch_grad_and_hvp()
        test_quadratic_loss_computation()  
        test_quadratic_dynamics()
        test_negative_loss_investigation()
        
        print("\n" + "=" * 80)
        print("🎉 All explicit validation tests passed!")
        print("The quadratic approximation implementation is mathematically correct.")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()