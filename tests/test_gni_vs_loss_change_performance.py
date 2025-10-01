#!/usr/bin/env python3
"""
Performance comparison test between GNI calculation and expected one-step full loss change.
Tests both functions with the same number of estimates using a real network and dataset.
"""

import time
import torch as T
import numpy as np
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.measure import calculate_averaged_gni, calculate_expected_one_step_full_loss_change
from utils.nets import MLP, initialize_net
import torch.nn as nn


def setup_test_environment(dataset_size=1000, batch_size=32, device='cpu'):
    """Set up network, dataset, and optimizer for testing."""
    print(f"Setting up test environment with {dataset_size} samples...")
    
    # Create a small MLP network
    net = MLP(input_dim=3072, hidden_dim=128, n_layers=2, output_dim=10)
    initialize_net(net, scale=0.2)
    net = net.to(device)
    
    # Create synthetic data (CIFAR-10 like)
    X = T.randn(dataset_size, 3072).to(device)  # Flattened 32x32x3 images
    Y = T.randint(0, 10, (dataset_size,)).to(device)  # Random labels
    
    # Loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = T.optim.SGD(net.parameters(), lr=0.01)
    
    print(f"Network parameters: {sum(p.numel() for p in net.parameters())}")
    print(f"Dataset size: {len(X)}")
    print(f"Batch size: {batch_size}")
    
    return net, X, Y, loss_fn, optimizer


def run_performance_test(net, X, Y, loss_fn, optimizer, batch_size, n_estimates_list, device='cpu'):
    """Run performance comparison for different numbers of estimates."""
    print(f"\nRunning performance tests on {device}...")
    print("=" * 60)
    
    results = {}
    
    for n_estimates in n_estimates_list:
        print(f"\nTesting with {n_estimates} estimates:")
        print("-" * 40)
        
        # Test GNI calculation
        print("Testing GNI calculation...")
        net.train()
        
        start_time = time.perf_counter()
        try:
            gni_result = calculate_averaged_gni(
                net=net,
                X=X,
                Y=Y, 
                loss_fn=loss_fn,
                batch_size=batch_size,
                n_estimates=n_estimates,
                min_estimates=min(10, n_estimates),
                tolerance=0.01
            )
            gni_time = time.perf_counter() - start_time
            gni_success = True
            print(f"  GNI result: {gni_result:.6f}")
        except Exception as e:
            gni_time = time.perf_counter() - start_time
            gni_success = False
            gni_result = None
            print(f"  GNI failed: {e}")
        
        print(f"  GNI time: {gni_time:.3f} seconds")
        
        # Test expected loss change calculation
        print("Testing expected loss change calculation...")
        net.train()
        
        start_time = time.perf_counter()
        try:
            loss_change_result = calculate_expected_one_step_full_loss_change(
                net=net,
                X=X,
                Y=Y,
                loss_fn=loss_fn,
                optimizer=optimizer,
                batch_size=batch_size,
                n_estimates=n_estimates,
                min_estimates=min(10, n_estimates),
                eps=0.01,
                eval_batch_size=256  # Use batched evaluation for memory efficiency
            )
            loss_change_time = time.perf_counter() - start_time
            loss_change_success = True
            print(f"  Loss change result: {loss_change_result:.6f}")
        except Exception as e:
            loss_change_time = time.perf_counter() - start_time
            loss_change_success = False
            loss_change_result = None
            print(f"  Loss change failed: {e}")
            
        print(f"  Loss change time: {loss_change_time:.3f} seconds")
        
        # Calculate speedup
        if gni_success and loss_change_success:
            if gni_time < loss_change_time:
                speedup = loss_change_time / gni_time
                faster = "GNI"
            else:
                speedup = gni_time / loss_change_time
                faster = "Loss Change"
            print(f"  {faster} is {speedup:.2f}x faster")
        
        # Store results
        results[n_estimates] = {
            'gni_time': gni_time,
            'gni_result': gni_result,
            'gni_success': gni_success,
            'loss_change_time': loss_change_time,
            'loss_change_result': loss_change_result,
            'loss_change_success': loss_change_success
        }
    
    return results


def print_summary(results):
    """Print summary of all test results."""
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"{'Estimates':<10} {'GNI Time':<12} {'Loss Change Time':<17} {'Faster Method':<15} {'Speedup':<10}")
    print("-" * 70)
    
    total_gni_time = 0
    total_loss_change_time = 0
    
    for n_estimates, result in results.items():
        gni_time = result['gni_time']
        loss_change_time = result['loss_change_time']
        
        total_gni_time += gni_time
        total_loss_change_time += loss_change_time
        
        if result['gni_success'] and result['loss_change_success']:
            if gni_time < loss_change_time:
                speedup = loss_change_time / gni_time
                faster = "GNI"
            else:
                speedup = gni_time / loss_change_time  
                faster = "Loss Change"
            speedup_str = f"{speedup:.2f}x"
        else:
            faster = "N/A"
            speedup_str = "N/A"
            
        print(f"{n_estimates:<10} {gni_time:<12.3f} {loss_change_time:<17.3f} {faster:<15} {speedup_str:<10}")
    
    print("-" * 70)
    print(f"{'TOTAL':<10} {total_gni_time:<12.3f} {total_loss_change_time:<17.3f}", end="")
    
    if total_gni_time < total_loss_change_time:
        overall_speedup = total_loss_change_time / total_gni_time
        overall_faster = "GNI"
    else:
        overall_speedup = total_gni_time / total_loss_change_time
        overall_faster = "Loss Change"
        
    print(f" {overall_faster:<15} {overall_speedup:.2f}x")


def main():
    """Main test function."""
    print("GNI vs Expected Loss Change Performance Test")
    print("=" * 60)
    
    # Test parameters
    dataset_size = 1000
    batch_size = 32
    n_estimates_list = [50, 100, 200, 500]
    
    # Detect device
    device = 'cuda' if T.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    T.manual_seed(42)
    np.random.seed(42)
    
    try:
        # Setup test environment
        net, X, Y, loss_fn, optimizer = setup_test_environment(
            dataset_size=dataset_size, 
            batch_size=batch_size,
            device=device
        )
        
        # Run performance tests
        results = run_performance_test(
            net, X, Y, loss_fn, optimizer, batch_size, n_estimates_list, device
        )
        
        # Print summary
        print_summary(results)
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())