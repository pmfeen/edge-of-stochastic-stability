#!/usr/bin/env python3

import torch
import torch.nn as nn
from utils.nets import SquaredLoss, MLP
from utils.quadratic import QuadraticApproximation

def test_gauss_newton_quadratic_loss():
    """Test the new Gauss-Newton quadratic loss computation."""
    device = torch.device('cpu')
    
    # Create a small MLP for testing
    net = MLP(input_dim=4, hidden_dim=8, n_layers=2, output_dim=2)
    loss_fn = SquaredLoss()
    
    # Create dummy data
    X = torch.randn(16, 4)  # batch_size=16, input_dim=4
    Y = torch.randn(16, 2)  # batch_size=16, output_dim=2
    
    # Initialize quadratic approximation
    quad_approx = QuadraticApproximation(
        net=net, 
        loss_fn=loss_fn, 
        device=device,
        anchor_step=0, 
        use_gauss_newton=False  # We're using it for the anchor, but using GN for quadratic loss computation
    )
    
    # Initialize the anchor at step 0
    dummy_loss = 1.0
    quad_approx.initialize_anchor(current_step=0, anchor_loss=dummy_loss)
    
    # Make a small step to create non-zero delta
    quad_grad = quad_approx.compute_quadratic_gradient(X, Y)
    quad_approx.update_delta(learning_rate=0.01, quad_gradient=quad_grad)
    
    print("Testing Gauss-Newton quadratic loss computation...")
    
    try:
        # Test the original Hessian-based quadratic loss
        quad_loss_hessian = quad_approx.compute_quadratic_loss_for_logging(X, Y)
        print(f"Hessian-based quadratic loss: {quad_loss_hessian}")
        
        # Test the new Gauss-Newton-based quadratic loss
        quad_loss_gn = quad_approx.compute_quadratic_loss_gauss_newton_for_logging(X, Y)
        print(f"Gauss-Newton-based quadratic loss: {quad_loss_gn}")
        
        # Both should be real numbers (not None or NaN)
        assert quad_loss_hessian is not None, "Hessian quadratic loss should not be None"
        assert quad_loss_gn is not None, "Gauss-Newton quadratic loss should not be None"
        assert not torch.isnan(torch.tensor(quad_loss_hessian)), "Hessian quadratic loss should not be NaN"
        assert not torch.isnan(torch.tensor(quad_loss_gn)), "Gauss-Newton quadratic loss should not be NaN"
        
        print("✓ Test passed! Both Hessian and Gauss-Newton quadratic losses computed successfully.")
        
        # They should be different in general (since GN is an approximation to Hessian)
        diff = abs(quad_loss_hessian - quad_loss_gn)
        print(f"Difference between Hessian and Gauss-Newton quadratic losses: {diff}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gauss_newton_quadratic_loss()
    exit(0 if success else 1)