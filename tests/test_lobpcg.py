"""Test module for LOBPCG eigenvalue computation."""

import torch
import numpy as np
import sys
import os

# Add the parent directory to the path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.lobpcg_old import torch_lobpcg

# Optional: import pytest if available, otherwise use simple assertions
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False


class TestLOBPCG:
    """Test suite for the LOBPCG algorithm implementation."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        torch.manual_seed(42)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float64  # Use float64 for better numerical precision in tests
    
    def test_small_symmetric_matrix(self):
        """Test LOBPCG on a small known symmetric matrix."""
        # Create a simple 4x4 symmetric matrix with known eigenvalues
        A_mat = torch.tensor([
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 3.0, 1.0, 0.0], 
            [0.0, 1.0, 2.0, 1.0],
            [0.0, 0.0, 1.0, 1.0]
        ], dtype=self.dtype, device=self.device)
        
        # Define matrix-vector multiplication function
        def A(x):
            return torch.matmul(A_mat, x)
        
        # Initialize random search directions
        n, k = 4, 2
        X = torch.randn(n, k, dtype=self.dtype, device=self.device)
        
        # Compute eigenvalues using LOBPCG
        theta_lobpcg, U_lobpcg, iterations = torch_lobpcg(A, X, max_iter=50, tol=1e-10)
        
        # Compute reference eigenvalues using torch.linalg.eigh
        theta_ref, U_ref = torch.linalg.eigh(A_mat)
        theta_ref = theta_ref[:k]  # Take the k largest eigenvalues
        
        # Sort both results for comparison (LOBPCG should return largest eigenvalues)
        theta_lobpcg_sorted, _ = torch.sort(theta_lobpcg, descending=True)
        theta_ref_sorted, _ = torch.sort(theta_ref, descending=True)
        print(f"LOBPCG Eigenvalues: {theta_lobpcg_sorted}")
        print(f"Reference Eigenvalues: {theta_ref_sorted}")
        
        # Check that eigenvalues match within tolerance
        torch.testing.assert_close(theta_lobpcg_sorted, theta_ref_sorted, atol=1e-3, rtol=1e-3)

        
        # Verify that the computed eigenvectors satisfy A*v = lambda*v
        for i in range(k):
            Av = A(U_lobpcg[:, i:i+1])
            lambda_v = theta_lobpcg[i] * U_lobpcg[:, i:i+1]
            torch.testing.assert_close(Av, lambda_v, atol=1e-6, rtol=1e-6)
    
    def test_diagonal_matrix(self):
        """Test LOBPCG on a diagonal matrix where eigenvalues are known exactly."""
        n, k = 10, 3
        diag_values = torch.arange(1, n+1, dtype=self.dtype, device=self.device)
        A_mat = torch.diag(diag_values)
        
        def A(x):
            return torch.matmul(A_mat, x)
        
        X = torch.randn(n, k, dtype=self.dtype, device=self.device)
        
        theta_lobpcg, U_lobpcg, iterations = torch_lobpcg(A, X, max_iter=50, tol=1e-12)
        
        # For a diagonal matrix, the largest k eigenvalues should be the largest k diagonal elements
        expected_eigenvalues = diag_values[-k:].flip(0)  # largest k in descending order
        
        # Sort LOBPCG results for comparison
        theta_sorted, _ = torch.sort(theta_lobpcg, descending=True)
        
        torch.testing.assert_close(theta_sorted, expected_eigenvalues, atol=1e-8, rtol=1e-8)
    
    def test_identity_matrix(self):
        """Test LOBPCG on identity matrix (all eigenvalues = 1)."""
        n, k = 8, 3
        
        def A(x):
            return x  # Identity matrix multiplication
        
        X = torch.randn(n, k, dtype=self.dtype, device=self.device)
        
        theta_lobpcg, U_lobpcg, iterations = torch_lobpcg(A, X, max_iter=10, tol=1e-12)
        
        # All eigenvalues should be 1.0
        expected = torch.ones(k, dtype=self.dtype, device=self.device)
        torch.testing.assert_close(theta_lobpcg, expected, atol=1e-10, rtol=1e-10)
    
    def test_convergence_behavior(self):
        """Test that LOBPCG converges and returns reasonable iteration count."""
        n, k = 20, 4
        
        # Create a well-conditioned symmetric matrix
        A_mat = torch.randn(n, n, dtype=self.dtype, device=self.device)
        A_mat = A_mat + A_mat.T  # Make symmetric
        A_mat += 5 * torch.eye(n, dtype=self.dtype, device=self.device)  # Make positive definite
        
        def A(x):
            return torch.matmul(A_mat, x)
        
        X = torch.randn(n, k, dtype=self.dtype, device=self.device)
        
        theta_lobpcg, U_lobpcg, iterations = torch_lobpcg(A, X, max_iter=50, tol=1e-8)
        
        # Check that algorithm converged (didn't hit max iterations)
        assert iterations < 50, f"LOBPCG didn't converge in 50 iterations (took {iterations})"
        
        # Check that we got the right number of eigenvalues
        assert theta_lobpcg.shape[0] == k
        assert U_lobpcg.shape == (n, k)
        
        # Verify eigenvector orthogonality
        U_T_U = torch.matmul(U_lobpcg.T, U_lobpcg)
        I = torch.eye(k, dtype=self.dtype, device=self.device)
        torch.testing.assert_close(U_T_U, I, atol=1e-8, rtol=1e-8)
    
    def test_small_matrix_fallback(self):
        """Test the fallback to direct eigenvalue computation for small matrices."""
        n, k = 6, 2  # n < 4*k, should trigger fallback
        
        A_mat = torch.randn(n, n, dtype=self.dtype, device=self.device)
        A_mat = A_mat + A_mat.T  # Make symmetric
        
        def A(x):
            return torch.matmul(A_mat, x)
        
        X = torch.randn(n, k, dtype=self.dtype, device=self.device)
        
        theta_lobpcg, U_lobpcg, iterations = torch_lobpcg(A, X, max_iter=50, tol=1e-10)
        
        # Should have taken only 1 iteration (direct computation)
        assert iterations == 1
        
        # Compare with direct eigenvalue computation
        theta_ref, _ = torch.linalg.eigh(A_mat)
        theta_ref = theta_ref[-k:]  # Take largest k eigenvalues
        
        theta_sorted, _ = torch.sort(theta_lobpcg, descending=True)
        theta_ref_sorted, _ = torch.sort(theta_ref, descending=True)
        
        torch.testing.assert_close(theta_sorted, theta_ref_sorted, atol=1e-10, rtol=1e-10)
    
    def test_different_dtypes(self):
        """Test LOBPCG with different floating point types."""
        n, k = 12, 3
        
        for dtype in [torch.float32, torch.float64]:
            A_mat = torch.randn(n, n, dtype=dtype, device=self.device)
            A_mat = A_mat + A_mat.T
            A_mat += torch.eye(n, dtype=dtype, device=self.device)
            
            def A(x):
                return torch.matmul(A_mat, x)
            
            X = torch.randn(n, k, dtype=dtype, device=self.device)
            
            theta_lobpcg, U_lobpcg, iterations = torch_lobpcg(A, X, max_iter=50)
            
            # Check output types
            assert theta_lobpcg.dtype == dtype
            assert U_lobpcg.dtype == dtype
            
            # Basic sanity checks
            assert theta_lobpcg.shape[0] == k
            assert U_lobpcg.shape == (n, k)


if __name__ == "__main__":
    # Run tests if this file is executed directly
    test_suite = TestLOBPCG()
    
    print("Running LOBPCG tests...")
    
    test_methods = [
        test_suite.test_small_symmetric_matrix,
        test_suite.test_diagonal_matrix, 
        test_suite.test_identity_matrix,
        test_suite.test_convergence_behavior,
        test_suite.test_small_matrix_fallback,
        test_suite.test_different_dtypes
    ]
    
    for i, test_method in enumerate(test_methods):
        try:
            test_suite.setup_method()
            test_method()
            print(f"✓ {test_method.__name__} passed")
        except Exception as e:
            print(f"✗ {test_method.__name__} failed: {e}")
    
    print("Test run complete!")
