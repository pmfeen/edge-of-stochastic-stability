"""
Comprehensive tests for lobpcg_faster.py implementation.

This test suite validates the optimized LOBPCG implementation against:
1. Original lobpcg.py implementation
2. SciPy's eigsh (for reference matrices)
3. Power iteration method (for largest eigenvalue)
4. PyTorch's full eigendecomposition (for small matrices)

Tests cover:
- Correctness of eigenvalues and eigenvectors
- Performance benchmarks
- Edge cases and numerical stability
- Warm restart functionality
"""

import torch
import numpy as np
import pytest
import time
import sys
import os

# Add utils to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from lobpcg_faster import torch_lobpcg as lobpcg_fast
from lobpcg import torch_lobpcg as lobpcg_orig

# For scipy comparison (when available)
try:
    from scipy.sparse.linalg import eigsh
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, skipping scipy comparison tests")


def power_iteration(A, x0, max_iter=1000, tol=1e-6):
    """Simple power iteration to find largest eigenvalue and eigenvector."""
    x = x0 / torch.linalg.norm(x0)
    
    for i in range(max_iter):
        x_new = A(x.unsqueeze(1)).squeeze(1)
        x_new = x_new / torch.linalg.norm(x_new)
        
        # Check convergence
        if torch.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    
    # Compute eigenvalue
    Ax = A(x.unsqueeze(1)).squeeze(1)
    eigenval = torch.dot(x, Ax).item()
    
    return eigenval, x, i + 1


def create_test_matrix(n, condition_number=1e3, device='cpu', dtype=torch.float32):
    """Create a well-conditioned symmetric positive definite test matrix."""
    # Generate eigenvalues with specified condition number
    eigenvals = torch.logspace(0, np.log10(condition_number), n, device=device, dtype=dtype)
    
    # Generate random orthogonal matrix
    Q, _ = torch.linalg.qr(torch.randn(n, n, device=device, dtype=dtype))
    
    # Construct matrix A = Q * diag(eigenvals) * Q^T
    A_matrix = Q @ torch.diag(eigenvals) @ Q.T
    
    # Create matrix-vector product function
    def matvec(X):
        return A_matrix @ X
    
    return matvec, A_matrix, eigenvals.flip(0)  # Return eigenvals in descending order


def create_neural_network_hessian(input_size=100, hidden_size=50, batch_size=32, device='cpu', dtype=torch.float32):
    """Create a realistic Hessian matrix from a neural network for testing."""
    torch.manual_seed(42)  # For reproducibility
    
    # Simple MLP
    net = torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, 1)
    ).to(device=device, dtype=dtype)
    
    # Generate random data
    X = torch.randn(batch_size, input_size, device=device, dtype=dtype)
    y = torch.randn(batch_size, 1, device=device, dtype=dtype)
    
    # Compute loss
    criterion = torch.nn.MSELoss()
    
    def hvp_func(v):
        """Hessian-vector product using autograd."""
        # Flatten parameters
        params = []
        for p in net.parameters():
            params.append(p.view(-1))
        params = torch.cat(params)
        
        # Compute gradients
        output = net(X)
        loss = criterion(output, y)
        grads = torch.autograd.grad(loss, net.parameters(), create_graph=True)
        
        # Flatten gradients
        flat_grads = []
        for g in grads:
            flat_grads.append(g.view(-1))
        flat_grads = torch.cat(flat_grads)
        
        # Compute HVP
        if v.dim() == 1:
            v = v.unsqueeze(1)
        
        hvps = []
        for i in range(v.shape[1]):
            hvp = torch.autograd.grad(flat_grads, net.parameters(), 
                                    grad_outputs=v[:, i], retain_graph=True)
            flat_hvp = []
            for h in hvp:
                flat_hvp.append(h.view(-1))
            hvps.append(torch.cat(flat_hvp))
        
        return torch.stack(hvps, dim=1)
    
    # Get parameter count
    param_count = sum(p.numel() for p in net.parameters())
    
    return hvp_func, param_count


class TestLOBPCGFaster:
    
    def __init__(self):
        """Setup for tests."""
        torch.manual_seed(12345)
        np.random.seed(12345)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float32
    
    def test_basic_functionality(self):
        """Test basic functionality on a small known matrix."""
        n, k = 50, 3
        
        # Create test matrix
        A_func, A_matrix, true_eigenvals = create_test_matrix(n, device=self.device, dtype=self.dtype)
        
        # Random initial guess
        X0 = torch.randn(n, k, device=self.device, dtype=self.dtype)
        
        # Test both implementations
        eigenvals_fast, eigenvecs_fast, iters_fast = lobpcg_fast(A_func, X0, max_iter=50)
        eigenvals_orig, eigenvecs_orig, iters_orig = lobpcg_orig(A_func, X0.clone(), max_iter=50)
        
        # Check eigenvalues match (both should find top-k)
        assert torch.allclose(eigenvals_fast, true_eigenvals[:k], rtol=1e-4, atol=1e-6), \
            f"Fast LOBPCG eigenvalues don't match: {eigenvals_fast} vs {true_eigenvals[:k]}"
        
        assert torch.allclose(eigenvals_orig, true_eigenvals[:k], rtol=1e-4, atol=1e-6), \
            f"Original LOBPCG eigenvalues don't match: {eigenvals_orig} vs {true_eigenvals[:k]}"
        
        # Check that both implementations find similar eigenvalues
        assert torch.allclose(eigenvals_fast, eigenvals_orig, rtol=1e-3), \
            f"Fast and original LOBPCG disagree: {eigenvals_fast} vs {eigenvals_orig}"
        
        print(f"✓ Basic functionality test passed")
        print(f"  Fast LOBPCG: {iters_fast} iterations")
        print(f"  Original LOBPCG: {iters_orig} iterations")
    
    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_against_scipy_eigsh(self):
        """Compare against scipy's eigsh for reference."""
        n, k = 100, 5
        
        # Create test matrix
        A_func, A_matrix, true_eigenvals = create_test_matrix(n, condition_number=1e4, 
                                                             device='cpu', dtype=torch.float32)
        
        # Convert to numpy for scipy
        A_numpy = A_matrix.detach().cpu().numpy()
        
        # Scipy solution
        scipy_eigenvals, scipy_eigenvecs = eigsh(A_numpy, k=k, which='LA')
        scipy_eigenvals = scipy_eigenvals[::-1].copy()  # Sort descending and copy to fix stride
        
        # LOBPCG solution
        X0 = torch.randn(n, k, dtype=torch.float32)
        eigenvals_fast, eigenvecs_fast, iters_fast = lobpcg_fast(A_func, X0, max_iter=100)
        
        # Compare eigenvalues
        scipy_eigenvals_torch = torch.tensor(scipy_eigenvals, dtype=torch.float32)
        assert torch.allclose(eigenvals_fast, scipy_eigenvals_torch, rtol=1e-4), \
            f"LOBPCG vs scipy eigenvalues: {eigenvals_fast} vs {scipy_eigenvals_torch}"
        
        print(f"✓ scipy comparison test passed")
        print(f"  LOBPCG: {eigenvals_fast}")
        print(f"  SciPy:  {scipy_eigenvals_torch}")
    
    def test_power_iteration_comparison(self):
        """Compare largest eigenvalue with power iteration."""
        n = 200
        
        # Create test matrix
        A_func, A_matrix, true_eigenvals = create_test_matrix(n, device=self.device, dtype=self.dtype)
        
        # Power iteration for largest eigenvalue
        x0 = torch.randn(n, device=self.device, dtype=self.dtype)
        pi_eigenval, pi_eigenvec, pi_iters = power_iteration(A_func, x0)
        
        # LOBPCG for largest eigenvalue
        X0 = torch.randn(n, 1, device=self.device, dtype=self.dtype)
        lobpcg_eigenvals, lobpcg_eigenvecs, lobpcg_iters = lobpcg_fast(A_func, X0, max_iter=50)
        
        # Compare largest eigenvalue (use relative tolerance for large eigenvalues)
        rel_error = abs(lobpcg_eigenvals[0].item() - pi_eigenval) / abs(pi_eigenval)
        assert rel_error < 1e-4, \
            f"LOBPCG vs power iteration: {lobpcg_eigenvals[0].item()} vs {pi_eigenval} (rel_error: {rel_error})"
        
        print(f"✓ Power iteration comparison passed")
        print(f"  LOBPCG: {lobpcg_eigenvals[0].item()} ({lobpcg_iters} iters)")
        print(f"  Power:  {pi_eigenval} ({pi_iters} iters)")
    
    def test_neural_network_hessian(self):
        """Test on realistic neural network Hessian."""
        hvp_func, param_count = create_neural_network_hessian(
            input_size=20, hidden_size=10, device=self.device, dtype=self.dtype
        )
        
        k = min(5, param_count // 10)  # Reasonable number of eigenvalues
        X0 = torch.randn(param_count, k, device=self.device, dtype=self.dtype)
        
        # Test both implementations
        eigenvals_fast, eigenvecs_fast, iters_fast = lobpcg_fast(hvp_func, X0, max_iter=30)
        eigenvals_orig, eigenvecs_orig, iters_orig = lobpcg_orig(hvp_func, X0.clone(), max_iter=30)
        
        # Check that eigenvalues are positive (Hessian should be PSD)
        assert torch.all(eigenvals_fast >= -1e-6), f"Negative eigenvalues found: {eigenvals_fast}"
        assert torch.all(eigenvals_orig >= -1e-6), f"Negative eigenvalues found: {eigenvals_orig}"
        
        # Check that both implementations agree
        assert torch.allclose(eigenvals_fast, eigenvals_orig, rtol=1e-2), \
            f"Fast vs original on NN Hessian: {eigenvals_fast} vs {eigenvals_orig}"
        
        print(f"✓ Neural network Hessian test passed")
        print(f"  Fast LOBPCG: {eigenvals_fast} ({iters_fast} iters)")
        print(f"  Orig LOBPCG: {eigenvals_orig} ({iters_orig} iters)")
    
    def test_warm_restart(self):
        """Test warm restart functionality."""
        n, k = 100, 3
        
        # Create test matrix
        A_func, A_matrix, true_eigenvals = create_test_matrix(n, device=self.device, dtype=self.dtype)
        
        # Cold start
        X0_cold = torch.randn(n, k, device=self.device, dtype=self.dtype)
        eigenvals_cold, eigenvecs_cold, iters_cold = lobpcg_fast(A_func, X0_cold, max_iter=50)
        
        # Warm start (use converged eigenvectors as initial guess)
        X0_warm = eigenvecs_cold.clone()
        eigenvals_warm, eigenvecs_warm, iters_warm = lobpcg_fast(
            A_func, X0_warm, max_iter=10, skip_p_on_first_iter=True
        )
        
        # Warm start should converge much faster
        assert iters_warm <= 3, f"Warm restart should converge quickly, got {iters_warm} iterations"
        
        # Should find same eigenvalues
        assert torch.allclose(eigenvals_cold, eigenvals_warm, rtol=1e-5), \
            f"Cold vs warm restart eigenvalues: {eigenvals_cold} vs {eigenvals_warm}"
        
        print(f"✓ Warm restart test passed")
        print(f"  Cold start: {iters_cold} iterations")
        print(f"  Warm start: {iters_warm} iterations")
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        n, k = 20, 2
        A_func, A_matrix, true_eigenvals = create_test_matrix(n, device=self.device, dtype=self.dtype)
        
        # Test with very small matrix (should fall back to full eigendecomposition)
        n_small = 10
        A_func_small, _, _ = create_test_matrix(n_small, device=self.device, dtype=self.dtype)
        X0_small = torch.randn(n_small, 3, device=self.device, dtype=self.dtype)
        
        eigenvals_small, eigenvecs_small, iters_small = lobpcg_fast(A_func_small, X0_small)
        assert iters_small == 1, "Small matrix should use direct eigendecomposition"
        
        # Test with k=1 (single eigenvector)
        X0_single = torch.randn(n, 1, device=self.device, dtype=self.dtype)
        eigenvals_single, eigenvecs_single, iters_single = lobpcg_fast(A_func, X0_single)
        assert eigenvals_single.shape == (1,), f"Expected shape (1,), got {eigenvals_single.shape}"
        
        print(f"✓ Edge cases test passed")
    
    def test_mixed_precision(self):
        """Test mixed precision functionality (if CUDA available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision test")
        
        n, k = 100, 3
        device = 'cuda'
        
        # Create test matrix
        A_func, A_matrix, true_eigenvals = create_test_matrix(n, device=device, dtype=torch.float32)
        
        # Test with mixed precision
        X0 = torch.randn(n, k, device=device, dtype=torch.float32)
        eigenvals_mixed, eigenvecs_mixed, iters_mixed = lobpcg_fast(
            A_func, X0, max_iter=50, hvp_autocast_dtype=torch.bfloat16
        )
        
        # Compare with full precision
        eigenvals_full, eigenvecs_full, iters_full = lobpcg_fast(A_func, X0.clone(), max_iter=50)
        
        # Should get similar results
        assert torch.allclose(eigenvals_mixed, eigenvals_full, rtol=1e-3), \
            f"Mixed vs full precision: {eigenvals_mixed} vs {eigenvals_full}"
        
        print(f"✓ Mixed precision test passed")
        print(f"  Mixed precision: {eigenvals_mixed}")
        print(f"  Full precision:  {eigenvals_full}")
    
    def test_performance_benchmark(self):
        """Performance benchmark comparison."""
        n, k = 500, 5
        
        # Create test matrix
        A_func, A_matrix, true_eigenvals = create_test_matrix(n, device=self.device, dtype=self.dtype)
        
        # Initial guess
        X0 = torch.randn(n, k, device=self.device, dtype=self.dtype)
        
        # Benchmark fast implementation
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        eigenvals_fast, eigenvecs_fast, iters_fast = lobpcg_fast(A_func, X0.clone(), max_iter=50)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        time_fast = time.time() - start_time
        
        # Benchmark original implementation
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        eigenvals_orig, eigenvecs_orig, iters_orig = lobpcg_orig(A_func, X0.clone(), max_iter=50)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        time_orig = time.time() - start_time
        
        speedup = time_orig / time_fast if time_fast > 0 else float('inf')
        
        print(f"✓ Performance benchmark completed")
        print(f"  Fast LOBPCG: {time_fast:.4f}s ({iters_fast} iters)")
        print(f"  Orig LOBPCG: {time_orig:.4f}s ({iters_orig} iters)")
        print(f"  Speedup: {speedup:.2f}x")
        
        # Results should be similar
        assert torch.allclose(eigenvals_fast, eigenvals_orig, rtol=1e-3), \
            f"Performance test eigenvalues disagree: {eigenvals_fast} vs {eigenvals_orig}"


def run_comprehensive_test():
    """Run all tests and summarize results."""
    print("=" * 60)
    print("LOBPCG_FASTER.PY COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    tester = TestLOBPCGFaster()
    
    tests = [
        ("Basic Functionality", tester.test_basic_functionality),
        ("Power Iteration Comparison", tester.test_power_iteration_comparison),
        ("Neural Network Hessian", tester.test_neural_network_hessian),
        ("Warm Restart", tester.test_warm_restart),
        ("Edge Cases", tester.test_edge_cases),
        ("Performance Benchmark", tester.test_performance_benchmark),
    ]
    
    # Add scipy test if available
    if HAS_SCIPY:
        tests.insert(1, ("SciPy eigsh Comparison", tester.test_against_scipy_eigsh))
    
    # Add mixed precision test if CUDA available
    if torch.cuda.is_available():
        tests.append(("Mixed Precision", tester.test_mixed_precision))
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {e}")
            failed += 1
    
    print(f"\n" + "=" * 60)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)