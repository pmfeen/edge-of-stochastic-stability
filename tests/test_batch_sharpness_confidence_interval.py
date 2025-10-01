import torch
import pytest

from utils.measure import calculate_averaged_grad_H_grad


class _DeterministicLinear(torch.nn.Module):
    def __init__(self, weight: float = 1.5):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.linear.weight.fill_(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _constant_dataset(num_samples: int = 8):
    X = torch.ones(num_samples, 1)
    Y = torch.ones(num_samples)
    return X, Y


def test_batch_sharpness_confidence_interval_matches_scalar_estimate():
    net = _DeterministicLinear()
    loss_fn = torch.nn.MSELoss(reduction="mean")
    X, Y = _constant_dataset()

    stats = calculate_averaged_grad_H_grad(
        net=net,
        X=X,
        Y=Y,
        loss_fn=loss_fn,
        batch_size=4,
        n_estimates=5,
        min_estimates=2,
        eps=0.0,
        return_confidence_interval=True,
    )

    assert isinstance(stats, dict)
    assert pytest.approx(stats["mean"], rel=1e-6) == stats["ci"][0]
    assert pytest.approx(stats["mean"], rel=1e-6) == stats["ci"][1]
    assert stats["stderr"] == 0.0
    assert stats["confidence_level"] == 0.95
    assert stats["num_samples"] >= 2

    baseline = calculate_averaged_grad_H_grad(
        net=net,
        X=X,
        Y=Y,
        loss_fn=loss_fn,
        batch_size=4,
        n_estimates=5,
        min_estimates=2,
        eps=0.0,
    )

    assert isinstance(baseline, float)
    assert pytest.approx(baseline, rel=1e-6) == stats["mean"]


def test_batch_sharpness_confidence_interval_expectation_inside():
    net = _DeterministicLinear()
    loss_fn = torch.nn.MSELoss(reduction="mean")
    X, Y = _constant_dataset()

    stats = calculate_averaged_grad_H_grad(
        net=net,
        X=X,
        Y=Y,
        loss_fn=loss_fn,
        batch_size=4,
        n_estimates=5,
        min_estimates=2,
        eps=0.0,
        expectation_inside=True,
        return_confidence_interval=True,
        confidence_level=0.99,
    )

    assert stats["confidence_level"] == 0.99
    assert pytest.approx(stats["mean"], rel=1e-6) == stats["ci"][0]
    assert pytest.approx(stats["mean"], rel=1e-6) == stats["ci"][1]


def test_batch_sharpness_confidence_interval_invalid_level():
    net = _DeterministicLinear()
    loss_fn = torch.nn.MSELoss(reduction="mean")
    X, Y = _constant_dataset()

    with pytest.raises(ValueError):
        calculate_averaged_grad_H_grad(
            net=net,
            X=X,
            Y=Y,
            loss_fn=loss_fn,
            batch_size=4,
            n_estimates=4,
            min_estimates=2,
            eps=0.0,
            return_confidence_interval=True,
            confidence_level=1.01,
        )
