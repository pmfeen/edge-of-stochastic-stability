"""Regression tests comparing LOBPCG against SciPy's Lanczos (eigsh).

The goal is to validate that our GPU-friendly LOBPCG implementation
matches a classical Lanczos solver when applied to the Hessian of a
small network. We check both the randomly initialised model and the
same model after a short synthetic training run.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

import sys
from pathlib import Path
# Add parent directory to Python path to import from utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.measure import create_hessian_vector_product
from utils.lobpcg import torch_lobpcg

try:
    from scipy.sparse.linalg import LinearOperator, eigsh
except ImportError as exc:  # pragma: no cover - pytest will surface the failure
    raise RuntimeError("SciPy is required for lobpcg vs Lanczos tests") from exc


@pytest.fixture(name="toy_data")
def fixture_toy_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a deterministic synthetic regression dataset."""
    torch.manual_seed(0)
    dtype = torch.float32
    X = torch.randn(64, 5, dtype=dtype)
    true_weights = torch.randn(5, 1, dtype=dtype)
    y = X @ true_weights + 0.05 * torch.randn(64, 1, dtype=dtype)
    return X, y


def _build_model() -> nn.Module:
    """Create the tiny network used across test scenarios."""
    torch.manual_seed(1)
    model = nn.Sequential(
        nn.Linear(5, 12),
        nn.Tanh(),
        nn.Linear(12, 1),
    )
    return model.double()


def _train_briefly(model: nn.Module, data: tuple[torch.Tensor, torch.Tensor], steps: int = 200) -> None:
    """Run a short SGD training loop to move away from the random initialisation."""
    X, y = data
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()


def _top_eigenvalues_lobpcg(model: nn.Module, X: torch.Tensor, y: torch.Tensor, k: int = 2) -> torch.Tensor:
    """Compute top Hessian eigenvalues via the project's LOBPCG implementation."""
    loss_fn = nn.MSELoss()
    loss = loss_fn(model(X), y)
    hvp = create_hessian_vector_product(loss, model)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    n_params = sum(p.numel() for p in model.parameters())
    init = torch.randn(n_params, k, device=device, dtype=dtype)
    eigenvalues, _, _ = torch_lobpcg(hvp, init, max_iter=1000, tol=1e-16)
    return eigenvalues.detach().cpu()


def _top_eigenvalues_lanczos(model: nn.Module, X: torch.Tensor, y: torch.Tensor, k: int = 2) -> torch.Tensor:
    """Compute top Hessian eigenvalues via SciPy's Lanczos implementation."""
    loss_fn = nn.MSELoss()
    loss = loss_fn(model(X), y)
    hvp = create_hessian_vector_product(loss, model)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    np_dtype = np.float64 if dtype == torch.float64 else np.float32
    n_params = sum(p.numel() for p in model.parameters())

    def matvec(vec: np.ndarray) -> np.ndarray:
        torch_vec = torch.from_numpy(vec).to(device=device, dtype=dtype)
        result = hvp(torch_vec)
        return result.detach().cpu().numpy().astype(np_dtype, copy=False)

    linear_op = LinearOperator(
        shape=(n_params, n_params),
        matvec=matvec,
        dtype=np_dtype,
    )

    # eigsh returns eigenvalues in ascending order; reverse to descending for comparison
    eigvals, _ = eigsh(linear_op, k=k, which="LM", tol=1e-10, maxiter=200)
    eigvals = np.sort(eigvals)[::-1]
    return torch.from_numpy(eigvals.copy()).to(dtype)


@pytest.mark.parametrize("trained", [False, True])
def test_lobpcg_matches_lanczos(toy_data: tuple[torch.Tensor, torch.Tensor], trained: bool) -> None:
    """Ensure LOBPCG agrees with Lanczos at init and after a brief training run."""
    X, y = toy_data
    model = _build_model().float()

    if trained:
        _train_briefly(model, toy_data)

    lobpcg_vals = _top_eigenvalues_lobpcg(model, X, y)
    lanczos_vals = _top_eigenvalues_lanczos(model, X, y)

    print("LOBPCG eigenvalues:", lobpcg_vals)
    print("Lanczos eigenvalues:", lanczos_vals)

    torch.testing.assert_close(lobpcg_vals, lanczos_vals, rtol=1e-4, atol=1e-6)


def _lobpcg_vs_lanczos_on_matrix(A: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Helper that compares eigenvalues for a dense symmetric matrix."""

    assert A.shape[0] == A.shape[1], "Matrix must be square"
    n = A.shape[0]
    dtype = A.dtype
    device = A.device

    def matmul(vec: torch.Tensor) -> torch.Tensor:
        if vec.ndim == 1:
            return A @ vec
        return A @ vec

    init = torch.randn(n, k, device=device, dtype=dtype)
    eigvals_lobpcg, _, _ = torch_lobpcg(matmul, init, max_iter=200, tol=1e-10)

    np_dtype = np.float64 if dtype == torch.float64 else np.float32

    def scipy_matvec(vec: np.ndarray) -> np.ndarray:
        torch_vec = torch.from_numpy(vec).to(device=device, dtype=dtype)
        result = (A @ torch_vec).detach().cpu().numpy().astype(np_dtype, copy=False)
        return result

    linear_op = LinearOperator(
        shape=(n, n),
        matvec=scipy_matvec,
        dtype=np_dtype,
    )
    eigvals_lanczos, _ = eigsh(linear_op, k=k, which="LM", tol=1e-12, maxiter=500)
    eigvals_lanczos = np.sort(eigvals_lanczos)[::-1]

    return eigvals_lobpcg.detach().cpu(), torch.from_numpy(eigvals_lanczos.copy()).to(dtype)


def test_lobpcg_matches_lanczos_on_random_matrix() -> None:
    """Sanity check LOBPCG against Lanczos on a small random SPD matrix."""
    torch.manual_seed(1234)
    n = 16
    k = 3
    base = torch.randn(n, n, dtype=torch.float32)
    A = base @ base.T + 0.5 * torch.eye(n, dtype=torch.float32)

    lobpcg_vals, lanczos_vals = _lobpcg_vs_lanczos_on_matrix(A, k)

    print("LOBPCG eigenvalues:", lobpcg_vals)
    print("Lanczos eigenvalues:", lanczos_vals)

    torch.testing.assert_close(lobpcg_vals, lanczos_vals, rtol=1e-6, atol=1e-8)
    


if __name__ == "__main__":
    test_lobpcg_matches_lanczos_on_random_matrix()
