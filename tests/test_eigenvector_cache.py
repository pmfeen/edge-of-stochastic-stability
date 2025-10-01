#!/usr/bin/env python3
"""
Test script to verify eigenvector caching functionality
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.measure import compute_lambdamax, create_eigenvector_cache, EigenvectorCache

def create_simple_model():
    """Create a simple model for testing"""
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )
    return model

def test_eigenvector_cache():
    """Test the eigenvector caching functionality"""
    print("Testing eigenvector caching...")
    
    # Create a simple model and some dummy data
    model = create_simple_model()
    x = torch.randn(16, 10)
    y = torch.randn(16, 1)
    
    # Define loss function
    loss_fn = nn.MSELoss()
    
    # Test 1: No cache (baseline)
    print("\n=== Test 1: No cache (baseline) ===")
    loss = loss_fn(model(x), y)
    eigenval_no_cache = compute_lambdamax(loss, model, max_iterations=50)
    print(f"Lambda max (no cache): {eigenval_no_cache:.6f}")
    
    # Test 2: Using EigenvectorCache
    print("\n=== Test 2: Using EigenvectorCache ===")
    cache = create_eigenvector_cache()
    
    # First computation (should use gradient as init)
    loss = loss_fn(model(x), y)
    eigenval_first = compute_lambdamax(loss, model, max_iterations=50, eigenvector_cache=cache)
    print(f"Lambda max (first with cache): {eigenval_first:.6f}")
    print(f"Cache has eigenvector: {len(cache) > 0}")
    
    # Second computation (should use cached eigenvector)
    loss = loss_fn(model(x), y)
    eigenval_second = compute_lambdamax(loss, model, max_iterations=50, eigenvector_cache=cache)
    print(f"Lambda max (second with cache): {eigenval_second:.6f}")
    
    # Test 3: Using dict-style cache (backward compatibility)
    print("\n=== Test 3: Using dict-style cache (backward compatibility) ===")
    dict_cache = {}
    
    # First computation
    loss = loss_fn(model(x), y)
    eigenval_dict_first = compute_lambdamax(loss, model, max_iterations=50, eigenvector_cache=dict_cache)
    print(f"Lambda max (dict cache first): {eigenval_dict_first:.6f}")
    print(f"Dict cache has eigenvector: {'eigenvector' in dict_cache}")
    
    # Second computation
    loss = loss_fn(model(x), y)
    eigenval_dict_second = compute_lambdamax(loss, model, max_iterations=50, eigenvector_cache=dict_cache)
    print(f"Lambda max (dict cache second): {eigenval_dict_second:.6f}")
    
    # Test 4: Return eigenvector functionality
    print("\n=== Test 4: Return eigenvector functionality ===")
    loss = loss_fn(model(x), y)
    eigenval, eigenvector = compute_lambdamax(loss, model, max_iterations=50, return_eigenvector=True)
    print(f"Lambda max: {eigenval:.6f}")
    print(f"Eigenvector shape: {eigenvector.shape}")
    print(f"Eigenvector norm: {torch.norm(eigenvector):.6f}")
    
    # Test 5: Test with gHg computation
    print("\n=== Test 5: Test with gHg computation ===")
    cache_ghg = create_eigenvector_cache()
    loss = loss_fn(model(x), y)
    eigenval, gHg = compute_lambdamax(loss, model, max_iterations=50, 
                                     eigenvector_cache=cache_ghg, 
                                     compute_gHg=True)
    print(f"Lambda max: {eigenval:.6f}")
    print(f"gHg: {gHg:.6f}")
    print(f"Cache has eigenvector: {len(cache_ghg) > 0}")
    
    print("\n=== All tests completed successfully! ===")

if __name__ == "__main__":
    test_eigenvector_cache()