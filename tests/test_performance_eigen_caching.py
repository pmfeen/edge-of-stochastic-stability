#!/usr/bin/env python3
"""
Performance test to show the benefit of eigenvector caching
"""

import torch
import torch.nn as nn
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.measure import compute_lambdamax, create_eigenvector_cache

def create_larger_model():
    """Create a larger model for better performance testing"""
    model = nn.Sequential(
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Linear(200, 100),
        nn.ReLU(),
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )
    return model

def performance_test():
    """Test performance benefits of eigenvector caching"""
    print("Testing performance benefits of eigenvector caching...")
    
    # Create a larger model for better performance testing
    model = create_larger_model()
    x = torch.randn(64, 100)
    y = torch.randn(64, 1)
    
    # Define loss function
    loss_fn = nn.MSELoss()
    
    # Test without cache
    print("\n=== Performance Test: Without Cache ===")
    times_no_cache = []
    eigenvals_no_cache = []
    
    for i in range(10):
        loss = loss_fn(model(x), y)
        start_time = time.time()
        eigenval = compute_lambdamax(loss, model, max_iterations=100)
        end_time = time.time()
        
        times_no_cache.append(end_time - start_time)
        eigenvals_no_cache.append(eigenval.item())
    
    avg_time_no_cache = sum(times_no_cache) / len(times_no_cache)
    print(f"Average time without cache: {avg_time_no_cache:.4f} seconds")
    print(f"Average eigenvalue: {sum(eigenvals_no_cache) / len(eigenvals_no_cache):.6f}")
    
    # Test with cache
    print("\n=== Performance Test: With Cache ===")
    cache = create_eigenvector_cache()
    times_with_cache = []
    eigenvals_with_cache = []
    
    for i in range(10):
        loss = loss_fn(model(x), y)
        start_time = time.time()
        eigenval = compute_lambdamax(loss, model, max_iterations=100, eigenvector_cache=cache)
        end_time = time.time()
        
        times_with_cache.append(end_time - start_time)
        eigenvals_with_cache.append(eigenval.item())
    
    avg_time_with_cache = sum(times_with_cache) / len(times_with_cache)
    print(f"Average time with cache: {avg_time_with_cache:.4f} seconds")
    print(f"Average eigenvalue: {sum(eigenvals_with_cache) / len(eigenvals_with_cache):.6f}")
    
    # Calculate speedup
    speedup = avg_time_no_cache / avg_time_with_cache
    print(f"\nSpeedup factor: {speedup:.2f}x")
    
    # Test convergence rates
    print("\n=== Convergence Analysis ===")
    
    # Without cache - count iterations to convergence
    loss = loss_fn(model(x), y)
    eigenval_no_cache = compute_lambdamax(loss, model, max_iterations=200, epsilon=1e-6)
    
    # With cache - count iterations to convergence
    cache_convergence = create_eigenvector_cache()
    loss = loss_fn(model(x), y)
    eigenval_with_cache = compute_lambdamax(loss, model, max_iterations=200, epsilon=1e-6, eigenvector_cache=cache_convergence)
    
    print(f"Final eigenvalue (no cache): {eigenval_no_cache:.6f}")
    print(f"Final eigenvalue (with cache): {eigenval_with_cache:.6f}")
    print(f"Difference: {abs(eigenval_no_cache - eigenval_with_cache):.8f}")
    
    print("\n=== Performance test completed! ===")

if __name__ == "__main__":
    performance_test()