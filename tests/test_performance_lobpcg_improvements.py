"""
Detailed performance analysis of lobpcg_faster.py vs lobpcg.py

This investigates when and why the optimizations provide speedups.
"""

import torch
import numpy as np
import time
import sys
import os

# Add utils to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from lobpcg import torch_lobpcg as lobpcg_fast
from lobpcg_old import torch_lobpcg as lobpcg_orig


def create_test_matrix(n, condition_number=1e3, device='cpu', dtype=torch.float32):
    """Create a well-conditioned symmetric positive definite test matrix."""
    eigenvals = torch.logspace(0, np.log10(condition_number), n, device=device, dtype=dtype)
    Q, _ = torch.linalg.qr(torch.randn(n, n, device=device, dtype=dtype))
    A_matrix = Q @ torch.diag(eigenvals) @ Q.T
    
    def matvec(X):
        return A_matrix @ X
    
    return matvec, A_matrix, eigenvals.flip(0)


def benchmark_implementations(n, k, num_runs=3, device='cpu'):
    """Benchmark both implementations with multiple runs."""
    
    print(f"\nBenchmarking n={n}, k={k}, device={device}")
    print("-" * 50)
    
    # Create test problem
    A_func, A_matrix, true_eigenvals = create_test_matrix(n, device=device)
    
    times_fast = []
    times_orig = []
    iters_fast = []
    iters_orig = []
    
    for run in range(num_runs):
        torch.manual_seed(42 + run)  # Different seed each run
        X0 = torch.randn(n, k, device=device, dtype=torch.float32)
        
        # Benchmark fast implementation
        if torch.cuda.is_available() and device == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()
        eigenvals_fast, eigenvecs_fast, iter_fast = lobpcg_fast(A_func, X0.clone(), max_iter=100)
        if torch.cuda.is_available() and device == 'cuda':
            torch.cuda.synchronize()
        time_fast = time.time() - start_time
        
        # Benchmark original implementation
        if torch.cuda.is_available() and device == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()
        eigenvals_orig, eigenvecs_orig, iter_orig = lobpcg_orig(A_func, X0.clone(), max_iter=100)
        if torch.cuda.is_available() and device == 'cuda':
            torch.cuda.synchronize()
        time_orig = time.time() - start_time
        
        times_fast.append(time_fast)
        times_orig.append(time_orig)
        iters_fast.append(iter_fast)
        iters_orig.append(iter_orig)
        
        # Check results are similar
        if not torch.allclose(eigenvals_fast, eigenvals_orig, rtol=1e-3):
            print(f"WARNING: Results differ in run {run}")
            print(f"  Fast: {eigenvals_fast}")
            print(f"  Orig: {eigenvals_orig}")
    
    avg_time_fast = np.mean(times_fast)
    avg_time_orig = np.mean(times_orig)
    avg_iters_fast = np.mean(iters_fast)
    avg_iters_orig = np.mean(iters_orig)
    
    speedup = avg_time_orig / avg_time_fast
    
    print(f"Fast LOBPCG: {avg_time_fast:.4f}s ± {np.std(times_fast):.4f}s ({avg_iters_fast:.1f} iters)")
    print(f"Orig LOBPCG: {avg_time_orig:.4f}s ± {np.std(times_orig):.4f}s ({avg_iters_orig:.1f} iters)")
    print(f"Speedup: {speedup:.2f}x")
    
    return speedup, avg_time_fast, avg_time_orig


def test_warm_restart_performance():
    """Test performance benefit of warm restarts."""
    print("\n" + "="*60)
    print("WARM RESTART PERFORMANCE TEST")
    print("="*60)
    
    n, k = 200, 3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    A_func, A_matrix, true_eigenvals = create_test_matrix(n, device=device)
    
    # Cold start
    X0_cold = torch.randn(n, k, device=device, dtype=torch.float32)
    
    start_time = time.time()
    eigenvals_cold, eigenvecs_cold, iters_cold = lobpcg_fast(A_func, X0_cold, max_iter=100)
    time_cold = time.time() - start_time
    
    # Warm restart with converged eigenvectors
    X0_warm = eigenvecs_cold.clone()
    
    start_time = time.time()
    eigenvals_warm, eigenvecs_warm, iters_warm = lobpcg_fast(
        A_func, X0_warm, max_iter=10, skip_p_on_first_iter=True
    )
    time_warm = time.time() - start_time
    
    print(f"Cold start: {time_cold:.4f}s ({iters_cold} iterations)")
    print(f"Warm restart: {time_warm:.4f}s ({iters_warm} iterations)")
    print(f"Warm restart speedup: {time_cold/time_warm:.1f}x")


def test_mixed_precision_performance():
    """Test mixed precision performance (CUDA only)."""
    if not torch.cuda.is_available():
        print("\nSkipping mixed precision test (CUDA not available)")
        return
    
    print("\n" + "="*60)
    print("MIXED PRECISION PERFORMANCE TEST")
    print("="*60)
    
    n, k = 500, 5
    device = 'cuda'
    
    A_func, A_matrix, true_eigenvals = create_test_matrix(n, device=device)
    X0 = torch.randn(n, k, device=device, dtype=torch.float32)
    
    # Full precision
    torch.cuda.synchronize()
    start_time = time.time()
    eigenvals_fp32, eigenvecs_fp32, iters_fp32 = lobpcg_fast(A_func, X0.clone(), max_iter=50)
    torch.cuda.synchronize()
    time_fp32 = time.time() - start_time
    
    # Mixed precision
    torch.cuda.synchronize()
    start_time = time.time()
    eigenvals_mixed, eigenvecs_mixed, iters_mixed = lobpcg_fast(
        A_func, X0.clone(), max_iter=50, hvp_autocast_dtype=torch.bfloat16
    )
    torch.cuda.synchronize()
    time_mixed = time.time() - start_time
    
    print(f"Full precision (FP32): {time_fp32:.4f}s ({iters_fp32} iterations)")
    print(f"Mixed precision (BF16): {time_mixed:.4f}s ({iters_mixed} iterations)")
    print(f"Mixed precision speedup: {time_fp32/time_mixed:.2f}x")
    
    # Check accuracy
    rel_error = torch.abs(eigenvals_fp32 - eigenvals_mixed) / eigenvals_fp32
    print(f"Relative error in eigenvalues: {rel_error.max().item():.2e}")


def main():
    """Run comprehensive performance analysis."""
    print("LOBPCG PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device}")
    
    # Test different problem sizes
    problem_sizes = [
        (100, 3),
        (200, 5),
        (500, 5),
        (1000, 10),
    ]
    
    speedups = []
    
    for n, k in problem_sizes:
        speedup, _, _ = benchmark_implementations(n, k, device=device)
        speedups.append(speedup)
    
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("Problem Size | Speedup")
    print("-" * 25)
    for (n, k), speedup in zip(problem_sizes, speedups):
        print(f"{n:4d} x {k:2d}     | {speedup:6.2f}x")
    
    # Test warm restarts
    test_warm_restart_performance()
    
    # Test mixed precision
    test_mixed_precision_performance()
    
    print(f"\n" + "="*60)
    print("CONCLUSIONS")
    print("="*60)
    
    avg_speedup = np.mean(speedups)
    if avg_speedup > 1.2:
        print("✓ lobpcg_faster.py provides significant speedups!")
    elif avg_speedup > 0.9:
        print("~ lobpcg_faster.py has comparable performance")
    else:
        print("⚠ lobpcg_faster.py is slower on average")
        print("  This might be due to overhead from optimizations on small problems")
    
    print(f"Average speedup: {avg_speedup:.2f}x")
    print("\nKey benefits of lobpcg_faster.py:")
    print("• Warm restart capability for iterative training")
    print("• Mixed precision support for large-scale problems")
    print("• More robust numerics with QR-based orthogonalization")
    print("• Single-pass block projection reduces overhead")


if __name__ == "__main__":
    main()