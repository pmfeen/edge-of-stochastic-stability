import torch
import math

from utils.measure import compute_gradient_projection_ratios


def test_grad_projection_basic_orthonormal():
    """
    For an orthonormal basis V = [e1, e2], and gradient g = 3 e1 + 4 e2 + 12 e3,
    we expect cumulative ratios:
      - i=1: 3/13
      - i=2: 5/13
      - residual: 12/13
    """
    n = 5
    e1 = torch.zeros(n); e1[0] = 1.0
    e2 = torch.zeros(n); e2[1] = 1.0
    e3 = torch.zeros(n); e3[2] = 1.0

    g = 3.0 * e1 + 4.0 * e2 + 12.0 * e3
    V = torch.stack([e1, e2], dim=1)  # shape [n, 2]

    out = compute_gradient_projection_ratios(g, V)

    denom = torch.linalg.vector_norm(g).item()
    assert math.isclose(out["grad_projection_01"], 3.0 / denom, rel_tol=1e-6)
    assert math.isclose(out["grad_projection_02"], 5.0 / denom, rel_tol=1e-6)
    assert math.isclose(out["grad_projection_residual"], 12.0 / denom, rel_tol=1e-6)


def test_grad_projection_cap_and_keys():
    """
    When more than 20 eigenvectors are provided, the metric should log only up to 20.
    With g = e1 and V containing e1..e25, all cumulative projections should be 1.0 and residual 0.0.
    """
    n = 30
    # Identity columns
    V = torch.eye(n)[:, :25]
    g = torch.zeros(n); g[0] = 1.0

    out = compute_gradient_projection_ratios(g, V)

    # Only keys 01..20 should exist (plus residual)
    for i in range(1, 21):
        key = f"grad_projection_{i:02d}"
        assert key in out
        assert math.isclose(out[key], 1.0, rel_tol=1e-7)
    assert "grad_projection_21" not in out
    assert math.isclose(out["grad_projection_residual"], 0.0, rel_tol=1e-7)


def test_grad_projection_zero_grad():
    """
    Zero gradient should return zeros for projections and residual.
    """
    n = 4
    V = torch.eye(n)[:, :2]
    g = torch.zeros(n)

    out = compute_gradient_projection_ratios(g, V)

    assert out["grad_projection_01"] == 0.0
    assert out["grad_projection_02"] == 0.0
    assert out["grad_projection_residual"] == 0.0


def test_grad_projection_respects_eigenvalue_ordering():
    """
    If eigenvalues are provided, the function should sort eigenvectors by descending value.
    With V = [e1, e2], eigenvalues = [1.0, 10.0], and g = e2, we expect
    grad_projection_01 == 1.0 (since e2 is the top eigenvector).
    """
    n = 3
    e1 = torch.tensor([1.0, 0.0, 0.0])
    e2 = torch.tensor([0.0, 1.0, 0.0])
    V = torch.stack([e1, e2], dim=1)  # columns
    g = e2.clone()

    out = compute_gradient_projection_ratios(g, V, eigenvalues=[1.0, 10.0])

    assert math.isclose(out["grad_projection_01"], 1.0, rel_tol=1e-7)
    assert math.isclose(out["grad_projection_02"], 1.0, rel_tol=1e-7)
    assert math.isclose(out["grad_projection_residual"], 0.0, rel_tol=1e-7)

