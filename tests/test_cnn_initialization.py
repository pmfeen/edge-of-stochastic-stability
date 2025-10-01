import torch
import torch.nn as nn

import sys
from pathlib import Path
# Add parent directory to Python path to import from utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.measure import compute_eigenvalues
from utils.nets import CNN, initialize_net


def _collect_weights(net):
    return torch.cat(
        [param.detach().flatten() for name, param in net.named_parameters() if "weight" in name],
        dim=0,
    )


def test_cnn_init_scale_affects_lambda_max():
    batch_size = 4
    input_shape = (3, 32, 32)
    fc_hidden = 64
    output_dim = 10
    init_seed = 1337

    torch.manual_seed(2024)
    inputs = torch.randn(batch_size, *input_shape)
    targets = torch.randn(batch_size, output_dim)
    loss_fn = nn.MSELoss()

    scales = [0.2, 0.3, 0.4, 0.6, 1.0]
    lambdas = {}
    unscaled_weights = None

    for scale in scales:
        net = CNN(fc_hidden, output_dim)
        initialize_net(net, scale=scale, seed=init_seed)

        # All biases should start at zero to keep activations unbiased
        for name, param in net.named_parameters():
            if "bias" in name:
                assert torch.allclose(param.detach(), torch.zeros_like(param))

        # Verify that different scales are just scalar multiples of the same base draws
        weights = _collect_weights(net)
        base_weights = weights / scale
        if unscaled_weights is None:
            unscaled_weights = base_weights
        else:
            assert torch.allclose(unscaled_weights, base_weights, atol=1e-6)

        predictions = net(inputs)
        loss = loss_fn(predictions, targets)
        eigenvalues = compute_eigenvalues(loss, net, max_iterations=40, reltol=1e-3)
        lambdas[scale] = eigenvalues.item()

    print("CNN eigenvalues by init_scale:")
    for scale in scales:
        print(f"  init_scale={scale}: eigenvalues={lambdas[scale]:.6f}")

    assert lambdas[scales[0]] < lambdas[scales[1]]


if __name__ == "__main__":
    test_cnn_init_scale_affects_lambda_max()