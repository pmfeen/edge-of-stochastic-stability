#!/usr/bin/env python3
"""Simple test runner for LOBPCG without external dependencies."""

import torch
import numpy as np
import sys
import os

# Add the parent directory to the path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.lobpcg_old import torch_lobpcg


def run_lobpcg_tests():
    """Run comprehensive tests for the LOBPCG implementation."""
    
    print("=" * 60)
    print("LOBPCG Test Suite")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running tests on device: {device}")
    
    passed = 0
    failed = 0
    
    def assert_close(actual, expected, atol=1e-6, rtol=1e-6, test_name=""):
        """Custom assertion for tensor equality."""
        try:
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
            return True
        except AssertionError as e:
            print(f"FAILED {test_name}: {e}")
            return False
    
    # Test 1: Small symmetric matrix
    print("\n1. Testing small symmetric matrix...")
    try:
        torch.manual_seed(42)
        A_mat = torch.tensor([
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 3.0, 1.0, 0.0], 
            [0.0, 1.0, 2.0, 1.0],
            [0.0, 0.0, 1.0, 1.0]
        ], dtype=torch.float64, device=device)
        
        def A(x):
            return torch.matmul(A_mat, x)
        
        n, k = 4, 2
        X = torch.randn(n, k, dtype=torch.float64, device=device)
        
        theta_lobpcg, U_lobpcg, iterations = torch_lobpcg(A, X, max_iter=50, tol=1e-10)
        
        # Reference solution
        theta_ref, U_ref = torch.linalg.eigh(A_mat)
        theta_ref = theta_ref[-k:]  # Take largest k eigenvalues
        
        theta_lobpcg_sorted, _ = torch.sort(theta_lobpcg, descending=True)
        theta_ref_sorted, _ = torch.sort(theta_ref, descending=True)
        
        if assert_close(theta_lobpcg_sorted, theta_ref_sorted, atol=1e-6, test_name="eigenvalues"):
            print("   ✓ Eigenvalues match reference solution")
            passed += 1
        else:
            failed += 1
            
    except Exception as e:
        print(f"   ✗ Test failed with exception: {e}")
        failed += 1
    
    # Test 2: Identity matrix
    print("\n2. Testing identity matrix...")
    try:
        torch.manual_seed(42)
        n, k = 8, 3
        
        def A(x):
            return x  # Identity matrix
        
        X = torch.randn(n, k, dtype=torch.float64, device=device)
        
        theta_lobpcg, U_lobpcg, iterations = torch_lobpcg(A, X, max_iter=10, tol=1e-12)
        
        expected = torch.ones(k, dtype=torch.float64, device=device)
        
        if assert_close(theta_lobpcg, expected, atol=1e-10, test_name="identity eigenvalues"):
            print("   ✓ Identity matrix eigenvalues correct")
            passed += 1
        else:
            failed += 1
            
    except Exception as e:
        print(f"   ✗ Test failed with exception: {e}")
        failed += 1
    
    # Test 3: Diagonal matrix
    print("\n3. Testing diagonal matrix...")
    try:
        torch.manual_seed(42)
        n, k = 10, 3
        diag_values = torch.arange(1, n+1, dtype=torch.float64, device=device)
        A_mat = torch.diag(diag_values)
        
        def A(x):
            return torch.matmul(A_mat, x)
        
        X = torch.randn(n, k, dtype=torch.float64, device=device)
        
        theta_lobpcg, U_lobpcg, iterations = torch_lobpcg(A, X, max_iter=50, tol=1e-12)
        
        expected_eigenvalues = diag_values[-k:].flip(0)
        theta_sorted, _ = torch.sort(theta_lobpcg, descending=True)
        
        if assert_close(theta_sorted, expected_eigenvalues, atol=1e-8, test_name="diagonal eigenvalues"):
            print("   ✓ Diagonal matrix eigenvalues correct")
            passed += 1
        else:
            failed += 1
            
    except Exception as e:
        print(f"   ✗ Test failed with exception: {e}")
        failed += 1
    
    # Test 4: Small matrix fallback
    print("\n4. Testing small matrix fallback...")
    try:
        torch.manual_seed(42)
        n, k = 6, 2  # n < 4*k
        
        A_mat = torch.randn(n, n, dtype=torch.float64, device=device)
        A_mat = A_mat + A_mat.T
        
        def A(x):
            return torch.matmul(A_mat, x)
        
        X = torch.randn(n, k, dtype=torch.float64, device=device)
        
        theta_lobpcg, U_lobpcg, iterations = torch_lobpcg(A, X, max_iter=50, tol=1e-10)
        
        if iterations == 1:
            print("   ✓ Small matrix fallback triggered correctly")
            passed += 1
        else:
            print(f"   ✗ Expected 1 iteration (fallback), got {iterations}")
            failed += 1
            
    except Exception as e:
        print(f"   ✗ Test failed with exception: {e}")
        failed += 1
    
    # Test 5: Orthogonality check
    print("\n5. Testing eigenvector orthogonality...")
    try:
        torch.manual_seed(42)
        n, k = 15, 4
        
        A_mat = torch.randn(n, n, dtype=torch.float64, device=device)
        A_mat = A_mat + A_mat.T
        A_mat += 5 * torch.eye(n, dtype=torch.float64, device=device)
        
        def A(x):
            return torch.matmul(A_mat, x)
        
        X = torch.randn(n, k, dtype=torch.float64, device=device)
        
        theta_lobpcg, U_lobpcg, iterations = torch_lobpcg(A, X, max_iter=50, tol=1e-8)
        
        # Check orthogonality
        U_T_U = torch.matmul(U_lobpcg.T, U_lobpcg)
        I = torch.eye(k, dtype=torch.float64, device=device)
        
        if assert_close(U_T_U, I, atol=1e-8, test_name="orthogonality"):
            print("   ✓ Eigenvectors are orthonormal")
            passed += 1
        else:
            failed += 1
            
    except Exception as e:
        print(f"   ✗ Test failed with exception: {e}")
        failed += 1
    
    # Test 6: Eigenvalue equation check
    print("\n6. Testing eigenvalue equation A*v = λ*v...")
    try:
        torch.manual_seed(42)
        n, k = 12, 3
        
        A_mat = torch.randn(n, n, dtype=torch.float64, device=device)
        A_mat = A_mat + A_mat.T
        A_mat += torch.eye(n, dtype=torch.float64, device=device)
        
        def A(x):
            return torch.matmul(A_mat, x)
        
        X = torch.randn(n, k, dtype=torch.float64, device=device)
        
        theta_lobpcg, U_lobpcg, iterations = torch_lobpcg(A, X, max_iter=10000, tol=1e-8)
        
        # Check A*v = λ*v for each eigenpair
        all_correct = True
        for i in range(k):
            Av = A(U_lobpcg[:, i:i+1])
            lambda_v = theta_lobpcg[i] * U_lobpcg[:, i:i+1]
            
            if not assert_close(Av, lambda_v, atol=1e-4, rtol=1e-3, test_name=f"eigenvalue equation {i}"):
                all_correct = False
        
        if all_correct:
            print("   ✓ All eigenvalue equations satisfied")
            passed += 1
        else:
            failed += 1
            
    except Exception as e:
        print(f"   ✗ Test failed with exception: {e}")
        failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    run_lobpcg_tests()
