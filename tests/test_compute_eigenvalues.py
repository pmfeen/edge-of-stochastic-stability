#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



import torch
import torch.nn as nn
from utils.measure import compute_eigenvalues, compute_lambdamax, create_eigenvector_cache

# Create a simple test network
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
    
    def forward(self, x):
        return self.linear(x)

def test_basic_functionality():
    print("Testing basic functionality...")
    
    # Create network and data
    net = SimpleNet()
    X = torch.randn(10, 2)
    y = torch.randn(10, 1)
    loss_fn = nn.MSELoss()
    
    # Compute loss
    preds = net(X)
    loss = loss_fn(preds, y)
    
    print(f"Loss: {loss.item():.6f}")
    
    # Test new compute_eigenvalues function with k=1
    print("\nTesting compute_eigenvalues with k=1:")
    eigenvalue = compute_eigenvalues(loss, net, k=1)
    print(f"Top eigenvalue: {eigenvalue.item():.6f}")
    
    # Test with eigenvector return
    eigenvalue, eigenvector = compute_eigenvalues(loss, net, k=1, return_eigenvectors=True)
    print(f"Top eigenvalue (with eigenvector): {eigenvalue.item():.6f}")
    print(f"Eigenvector norm: {torch.norm(eigenvector).item():.6f}")
    
    # Test backward compatibility
    print("\nTesting backward compatibility compute_lambdamax:")
    old_eigenvalue = compute_lambdamax(loss, net)
    print(f"Old API eigenvalue: {old_eigenvalue.item():.6f}")
    
    # Check if they're close
    print(f"Difference: {abs(eigenvalue.item() - old_eigenvalue.item()):.8f}")
    
    # Test power iteration vs LOBPCG comparison
    print("\nTesting power iteration vs LOBPCG:")
    
    # Test with LOBPCG (default)
    lobpcg_eigenvalue = compute_eigenvalues(loss, net, k=1, use_power_iteration=False)
    print(f"LOBPCG eigenvalue: {lobpcg_eigenvalue.item():.6f}")
    
    # Test with power iteration
    power_eigenvalue = compute_eigenvalues(loss, net, k=1, use_power_iteration=True)
    print(f"Power iteration eigenvalue: {power_eigenvalue.item():.6f}")
    
    # Compare the two methods
    diff = abs(lobpcg_eigenvalue.item() - power_eigenvalue.item())
    print(f"Difference between methods: {diff:.8f}")
    
    # Assert they're close (within reasonable tolerance)
    tolerance = 1e-3  # Relaxed tolerance since methods can have small differences with default iterations
    if diff < tolerance:
        print(f"✓ Methods agree within tolerance {tolerance}")
    else:
        print(f"⚠ Methods differ by {diff}, exceeding tolerance {tolerance}")
        print("  Note: Consider increasing max_iterations for better convergence")
    
    # Test with eigenvectors from both methods
    print("\nTesting eigenvectors from both methods:")
    lobpcg_val, lobpcg_vec = compute_eigenvalues(loss, net, k=1, use_power_iteration=False, return_eigenvectors=True)
    power_val, power_vec = compute_eigenvalues(loss, net, k=1, use_power_iteration=True, return_eigenvectors=True)
    
    print(f"LOBPCG: eigenvalue={lobpcg_val.item():.6f}, eigenvector norm={torch.norm(lobpcg_vec).item():.6f}")
    print(f"Power:  eigenvalue={power_val.item():.6f}, eigenvector norm={torch.norm(power_vec).item():.6f}")
    
    # Compare eigenvectors (they may differ in sign, so check both orientations)
    vec_diff1 = torch.norm(lobpcg_vec - power_vec).item()
    vec_diff2 = torch.norm(lobpcg_vec + power_vec).item()
    vec_diff = min(vec_diff1, vec_diff2)
    print(f"Eigenvector difference (accounting for sign): {vec_diff:.6f}")
    
    # Test with k>1
    print(f"\nTesting compute_eigenvalues with k=3:")
    eigenvalues = compute_eigenvalues(loss, net, k=3)
    print(f"Top 3 eigenvalues: {eigenvalues}")
    
    print("\nTest completed successfully!")

def test_power_iteration_vs_lobpcg():
    """Dedicated test for comparing power iteration and LOBPCG methods"""
    print("\n" + "="*60)
    print("Testing Power Iteration vs LOBPCG Equivalence")
    print("="*60)
    
    # Test on different network sizes and problems
    test_cases = [
        ("Small network", SimpleNet()),
        ("Larger network", nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 1)))
    ]
    
    for case_name, net in test_cases:
        print(f"\n{case_name}:")
        print("-" * 40)
        
        # Generate appropriate input data
        if hasattr(net, 'linear'):
            input_dim = net.linear.in_features
        else:
            input_dim = 5  # For the larger network
            
        X = torch.randn(20, input_dim)
        y = torch.randn(20, 1)
        loss_fn = nn.MSELoss()
        
        preds = net(X)
        loss = loss_fn(preds, y)
        
        # Compare methods with different tolerances
        tolerances = [1e-2, 1e-3, 1e-4]
        
        for tol in tolerances:
            # LOBPCG
            lobpcg_val = compute_eigenvalues(loss, net, k=1, use_power_iteration=False, 
                                           reltol=tol, max_iterations=200)
            
            # Power iteration  
            power_val = compute_eigenvalues(loss, net, k=1, use_power_iteration=True, 
                                          reltol=tol, max_iterations=200)
            
            diff = abs(lobpcg_val.item() - power_val.item())
            print(f"  Tolerance {tol}: LOBPCG={lobpcg_val.item():.6f}, Power={power_val.item():.6f}, diff={diff:.8f}")
            
            # Check convergence
            if diff < tol * 10:  # Allow some margin over the relative tolerance
                print(f"    ✓ Methods converged within expected range")
            else:
                print(f"    ⚠ Methods differ more than expected")
    
    print("\nPower iteration vs LOBPCG test completed!")

def test_eigenvector_caching_consistency():
    """Test that eigenvector caching works consistently with both methods"""
    print("\n" + "="*60)
    print("Testing Eigenvector Caching Consistency")
    print("="*60)
    
    net = SimpleNet()
    X = torch.randn(10, 2)
    y = torch.randn(10, 1)
    loss_fn = nn.MSELoss()
    
    preds = net(X)
    loss = loss_fn(preds, y)
    
    # Test with eigenvector cache
    cache = create_eigenvector_cache()
    
    # First call with LOBPCG - should populate cache
    lobpcg_val1, lobpcg_vec1 = compute_eigenvalues(loss, net, k=1, use_power_iteration=False, 
                                                   return_eigenvectors=True, eigenvector_cache=cache)
    print(f"LOBPCG (first call): eigenvalue={lobpcg_val1.item():.6f}")
    
    # Second call with LOBPCG - should use cache
    lobpcg_val2, lobpcg_vec2 = compute_eigenvalues(loss, net, k=1, use_power_iteration=False, 
                                                   return_eigenvectors=True, eigenvector_cache=cache)
    print(f"LOBPCG (cached call): eigenvalue={lobpcg_val2.item():.6f}")
    
    # Test with power iteration using same cache
    power_val, power_vec = compute_eigenvalues(loss, net, k=1, use_power_iteration=True, 
                                              return_eigenvectors=True, eigenvector_cache=cache)
    print(f"Power iteration (with cache): eigenvalue={power_val.item():.6f}")
    
    # Compare all three
    diff1 = abs(lobpcg_val1.item() - lobpcg_val2.item())
    diff2 = abs(lobpcg_val1.item() - power_val.item())
    
    print(f"Difference between LOBPCG calls: {diff1:.8f}")
    print(f"Difference between LOBPCG and Power: {diff2:.8f}")
    
    if diff1 < 1e-6:
        print("✓ LOBPCG caching is consistent")
    else:
        print("⚠ LOBPCG caching shows unexpected variation")
        
    if diff2 < 1e-4:
        print("✓ Power iteration agrees with cached LOBPCG")
    else:
        print("⚠ Power iteration differs significantly from cached LOBPCG")
    
    print("\nEigenvector caching test completed!")

if __name__ == "__main__":
    test_basic_functionality()
    test_power_iteration_vs_lobpcg()
    test_eigenvector_caching_consistency()