#!/usr/bin/env python3
"""
Example usage of the eigenvector caching functionality
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Add the parent directory to the path to access utils.measure
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.measure import compute_lambdamax, calculate_averaged_lambdamax, create_eigenvector_cache

def simulate_training_with_cache():
    """
    Simulate a training loop where we compute lambda_max at each step
    and reuse the eigenvector from the previous step for faster convergence.
    """
    print("Simulating training with eigenvector caching...")
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(20, 50),
        nn.ReLU(),
        nn.Linear(50, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )
    
    # Generate some dummy data
    x = torch.randn(32, 20)
    y = torch.randn(32, 1)
    loss_fn = nn.MSELoss()
    
    # Create eigenvector cache
    cache = create_eigenvector_cache()
    
    print("\n=== Training Simulation ===")
    print("Step | Lambda Max | Cached Eigenvector | Computation Time")
    print("-" * 55)
    
    for step in range(10):
        # Slightly modify the model to simulate training steps
        if step > 0:
            with torch.no_grad():
                for param in model.parameters():
                    param += torch.randn_like(param) * 0.01
        
        # Compute loss
        loss = loss_fn(model(x), y)
        
        # Compute lambda_max with caching
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if start_time:
            start_time.record()
            
        eigenval = compute_lambdamax(loss, model, max_iterations=50, eigenvector_cache=cache)
        
        if start_time:
            end_time = torch.cuda.Event(enable_timing=True)
            end_time.record()
            torch.cuda.synchronize()
            elapsed = start_time.elapsed_time(end_time)
        else:
            elapsed = 0.0
        
        print(f"{step:4d} | {eigenval:10.6f} | {len(cache) > 0:17} | {elapsed:12.2f} ms")
    
    print("\n=== Example with averaged lambda_max ===")
    
    # Example using calculate_averaged_lambdamax with cache
    cache_averaged = create_eigenvector_cache()
    
    sharpnesses = calculate_averaged_lambdamax(
        model, x, y, loss_fn, 
        batch_size=8, 
        n_estimates=20,
        eigenvector_cache=cache_averaged
    )
    
    print(f"Averaged lambda_max: {torch.mean(torch.tensor(sharpnesses)):.6f}")
    print(f"Standard deviation: {torch.std(torch.tensor(sharpnesses)):.6f}")
    print(f"Number of estimates: {len(sharpnesses)}")
    
    print("\n=== Example with multiple eigenvectors (future LOBPCG) ===")
    
    # Demonstrate storing multiple eigenvectors
    multi_cache = create_eigenvector_cache(max_eigenvectors=3)
    
    # Generate some eigenvectors
    loss = loss_fn(model(x), y)
    eigenval1, eigenvec1 = compute_lambdamax(loss, model, max_iterations=30, return_eigenvector=True)
    eigenval2, eigenvec2 = compute_lambdamax(loss, model, max_iterations=30, return_eigenvector=True)
    eigenval3, eigenvec3 = compute_lambdamax(loss, model, max_iterations=30, return_eigenvector=True)
    
    # Store multiple eigenvectors
    multi_cache.store_eigenvectors([eigenvec1, eigenvec2, eigenvec3], [eigenval1, eigenval2, eigenval3])
    
    print(f"Stored {len(multi_cache)} eigenvectors")
    print(f"Eigenvalues: {[f'{ev:.6f}' for ev in multi_cache.eigenvalues]}")
    
    # Demonstrate warm start with multiple vectors
    warm_start_vectors = multi_cache.get_warm_start_vectors()
    if warm_start_vectors:
        print(f"Retrieved {len(warm_start_vectors)} warm start vectors")
        for i, vec in enumerate(warm_start_vectors):
            print(f"  Vector {i}: shape {vec.shape}, norm {torch.norm(vec):.6f}")

if __name__ == "__main__":
    simulate_training_with_cache()