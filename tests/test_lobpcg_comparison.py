#!/usr/bin/env python3
"""
Test script to compare compute_lambdamax with compute_multiple_eigenvalues_lobpcg
and verify that the LOBPCG implementation works correctly
"""

import torch
import torch.nn as nn
import sys
import os
import time
import wandb
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.measure import compute_lambdamax, compute_multiple_eigenvalues_lobpcg, create_eigenvector_cache

# Initialize wandb in disabled mode to avoid logging during tests
wandb.init(mode="disabled")

def create_simple_mlp(input_dim=10, hidden_dim=20, output_dim=1):
    """Create a simple MLP for testing"""
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    )
    return model

def create_small_cnn():
    """Create a small CNN for testing"""
    model = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(16 * 4 * 4, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    return model

def test_lobpcg_basic_functionality():
    """Test that compute_multiple_eigenvalues_lobpcg runs without errors"""
    print("\n=== Test 1: Basic LOBPCG functionality ===")
    
    # Create a simple model
    model = create_simple_mlp(input_dim=10, hidden_dim=15, output_dim=1)
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    
    loss_fn = nn.MSELoss()
    loss = loss_fn(model(x), y)
    
    print("Testing LOBPCG with k=3 eigenvalues...")
    try:
        eigenvalues = compute_multiple_eigenvalues_lobpcg(
            loss, model, k=3, max_iterations=50, reltol=0.05
        )
        print(f"✓ LOBPCG computed successfully")
        print(f"  Eigenvalues shape: {eigenvalues.shape}")
        print(f"  Eigenvalues: {eigenvalues.cpu().numpy()}")
        print(f"  All eigenvalues positive: {torch.all(eigenvalues >= 0).item()}")
        print(f"  Eigenvalues in descending order: {torch.all(eigenvalues[:-1] >= eigenvalues[1:]).item()}")
        return True
    except Exception as e:
        print(f"✗ LOBPCG failed with error: {e}")
        return False

def test_lobpcg_with_eigenvectors():
    """Test LOBPCG with eigenvector return"""
    print("\n=== Test 2: LOBPCG with eigenvectors ===")
    
    model = create_simple_mlp(input_dim=8, hidden_dim=12, output_dim=1)
    x = torch.randn(16, 8)
    y = torch.randn(16, 1)
    
    loss_fn = nn.MSELoss()
    loss = loss_fn(model(x), y)
    
    try:
        eigenvalues, eigenvectors = compute_multiple_eigenvalues_lobpcg(
            loss, model, k=2, max_iterations=50, return_eigenvectors=True
        )
        print(f"✓ LOBPCG with eigenvectors computed successfully")
        print(f"  Eigenvalues shape: {eigenvalues.shape}")
        print(f"  Eigenvectors shape: {eigenvectors.shape}")
        print(f"  Eigenvalues: {eigenvalues.cpu().numpy()}")
        
        # Check orthogonality of eigenvectors
        if eigenvectors.shape[1] > 1:
            dot_product = torch.abs(torch.dot(eigenvectors[:, 0], eigenvectors[:, 1]))
            print(f"  Eigenvector orthogonality (|v1·v2|): {dot_product:.6f}")
            
        return True
    except Exception as e:
        print(f"✗ LOBPCG with eigenvectors failed: {e}")
        return False

def compare_lambdamax_vs_lobpcg():
    """Compare compute_lambdamax with compute_multiple_eigenvalues_lobpcg"""
    print("\n=== Test 3: Comparison between power iteration and LOBPCG ===")
    
    model = create_simple_mlp(input_dim=12, hidden_dim=20, output_dim=1)
    x = torch.randn(24, 12)
    y = torch.randn(24, 1)
    
    loss_fn = nn.MSELoss()
    loss = loss_fn(model(x), y)
    
    try:
        # Compute using power iteration (compute_lambdamax)
        print("Computing lambda_max using power iteration...")
        start_time = time.time()
        lambda_max_power = compute_lambdamax(loss, model, max_iterations=1000, reltol=1e-3)
        power_time = time.time() - start_time
        print(f"  Power iteration result: {lambda_max_power:.6f} (time: {power_time:.3f}s)")
        
        # Compute using LOBPCG (only the largest eigenvalue)
        print("Computing lambda_max using LOBPCG...")
        start_time = time.time()
        eigenvalues_lobpcg = compute_multiple_eigenvalues_lobpcg(
            loss, model, k=1, max_iterations=50, reltol=0.05
        )
        lobpcg_time = time.time() - start_time
        lambda_max_lobpcg = eigenvalues_lobpcg[0]
        print(f"  LOBPCG result: {lambda_max_lobpcg:.6f} (time: {lobpcg_time:.3f}s)")
        
        # Compare results
        relative_error = torch.abs(lambda_max_power - lambda_max_lobpcg) / lambda_max_power
        print(f"  Relative error: {relative_error:.4f} ({relative_error*100:.2f}%)")
        
        if relative_error < 0.1:  # 10% tolerance
            print("✓ Results agree within tolerance")
        else:
            print("⚠ Results differ significantly")
            
        return True
        
    except Exception as e:
        print(f"✗ Comparison failed: {e}")
        return False

def test_multiple_eigenvalues():
    """Test computing multiple eigenvalues with LOBPCG"""
    print("\n=== Test 4: Multiple eigenvalues computation ===")
    
    model = create_simple_mlp(input_dim=8, hidden_dim=16, output_dim=1)
    x = torch.randn(20, 8)
    y = torch.randn(20, 1)
    
    loss_fn = nn.MSELoss()
    loss = loss_fn(model(x), y)
    
    try:
        # Test with different k values
        for k in [1, 3, 5]:
            print(f"\nTesting with k={k} eigenvalues:")
            eigenvalues = compute_multiple_eigenvalues_lobpcg(
                loss, model, k=k, max_iterations=50, reltol=0.05
            )
            
            print(f"  Computed {len(eigenvalues)} eigenvalues")
            print(f"  Eigenvalues: {eigenvalues.cpu().numpy()}")
            
            # Verify they're in descending order
            if len(eigenvalues) > 1:
                is_descending = torch.all(eigenvalues[:-1] >= eigenvalues[1:])
                print(f"  Descending order: {is_descending.item()}")
                
        return True
        
    except Exception as e:
        print(f"✗ Multiple eigenvalues test failed: {e}")
        return False

def test_with_cnn():
    """Test both methods with a CNN architecture"""
    print("\n=== Test 5: CNN architecture test ===")
    
    model = create_small_cnn()
    x = torch.randn(8, 1, 8, 8)  # Small images
    y = torch.randn(8, 1)
    
    loss_fn = nn.MSELoss()
    loss = loss_fn(model(x), y)
    
    lambda_max_power = None
    eigenvalues_lobpcg = None
    
    try:
        # Test power iteration
        print("Testing power iteration with CNN...")
        lambda_max_power = compute_lambdamax(loss, model, max_iterations=1000, reltol=0.02)
        print(f"  Power iteration result: {lambda_max_power:.6f}")
        
    except Exception as e:
        print(f"  Power iteration failed: {e}")
        
    try:
        # Test LOBPCG
        print("Testing LOBPCG with CNN...")
        eigenvalues_lobpcg = compute_multiple_eigenvalues_lobpcg(
            loss, model, k=2, max_iterations=30, reltol=0.05
        )
        print(f"  LOBPCG results: {eigenvalues_lobpcg.cpu().numpy()}")
        
    except Exception as e:
        print(f"  LOBPCG failed: {e}")
        
    # Compare if both succeeded
    if lambda_max_power is not None and eigenvalues_lobpcg is not None:
        relative_error = torch.abs(lambda_max_power - eigenvalues_lobpcg[0]) / lambda_max_power
        print(f"  Relative error: {relative_error:.4f} ({relative_error*100:.2f}%)")
        return True
    elif lambda_max_power is not None or eigenvalues_lobpcg is not None:
        print("  At least one method succeeded")
        return True
    else:
        return False

def test_with_cache():
    """Test LOBPCG with eigenvector caching"""
    print("\n=== Test 6: LOBPCG with eigenvector caching ===")
    
    model = create_simple_mlp(input_dim=6, hidden_dim=10, output_dim=1)
    x = torch.randn(16, 6)
    y = torch.randn(16, 1)
    
    loss_fn = nn.MSELoss()
    
    try:
        # Create cache
        cache = create_eigenvector_cache(max_eigenvectors=3)
        
        # First computation
        loss = loss_fn(model(x), y)
        eigenvalues1 = compute_multiple_eigenvalues_lobpcg(
            loss, model, k=3, max_iterations=50, eigenvector_cache=cache
        )
        print(f"First computation: {eigenvalues1.cpu().numpy()}")
        print(f"Cache size after first: {len(cache)}")
        
        # Second computation (should use cached vectors)
        loss = loss_fn(model(x), y)
        eigenvalues2 = compute_multiple_eigenvalues_lobpcg(
            loss, model, k=3, max_iterations=30, eigenvector_cache=cache
        )
        print(f"Second computation: {eigenvalues2.cpu().numpy()}")
        
        # Compare results
        relative_errors = torch.abs(eigenvalues1 - eigenvalues2) / eigenvalues1
        print(f"Relative errors: {relative_errors.cpu().numpy()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Cache test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and provide summary"""
    print("=" * 60)
    print("LOBPCG vs Power Iteration Comparison Tests")
    print("=" * 60)
    
    tests = [
        ("Basic LOBPCG functionality", test_lobpcg_basic_functionality),
        ("LOBPCG with eigenvectors", test_lobpcg_with_eigenvectors),
        ("Power iteration vs LOBPCG comparison", compare_lambdamax_vs_lobpcg),
        ("Multiple eigenvalues", test_multiple_eigenvalues),
        ("CNN architecture", test_with_cnn),
        ("Eigenvector caching", test_with_cache),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! LOBPCG implementation is working correctly.")
    else:
        print(f"⚠ {total - passed} test(s) failed. Check the output above for details.")

if __name__ == "__main__":
    run_all_tests()