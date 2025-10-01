import torch
import pytest

import sys
from pathlib import Path
# Add parent directory to Python path to import from utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.measure import estimate_hessian_trace
from utils.nets import SquaredLoss


def test_estimate_hessian_trace_matches_identity_design():
    net = torch.nn.Linear(2, 1, bias=False)
    # Deterministic weights help ensure the Hessian remains the identity matrix
    with torch.no_grad():
        net.weight.copy_(torch.tensor([[1.0, -1.0]]))

    X = torch.eye(2)
    Y = torch.zeros(2)
    loss_fn = SquaredLoss()

    # Hutchinson estimator with deterministic probes
    rng = torch.Generator()
    rng.manual_seed(2024)

    trace_estimate = estimate_hessian_trace(
        net,
        X,
        Y,
        loss_fn,
        max_estimates=2048,
        min_estimates=2048,
        eps=1e-4,
        generator=rng,
    )

    # For this dataset, the Hessian equals the identity, so the trace must be 2.
    assert trace_estimate == pytest.approx(2.0, rel=0.02, abs=1e-2)
