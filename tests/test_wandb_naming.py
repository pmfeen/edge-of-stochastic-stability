import unittest

from utils.naming import sanitize_run_name_part as _sanitize_run_name_part
from utils.naming import compose_run_name as _compose_run_name


class DummyArgs:
    def __init__(self, dataset, model, batch, lr, wandb_name=None):
        self.dataset = dataset
        self.model = model
        self.batch = batch
        self.lr = lr
        self.wandb_name = wandb_name


class TestWandbNaming(unittest.TestCase):
    def test_sanitize_basic(self):
        self.assertEqual(_sanitize_run_name_part("abcDEF-123_45.6"), "abcDEF-123_45.6")

    def test_sanitize_spaces_and_symbols(self):
        self.assertEqual(_sanitize_run_name_part("  My Exp! v1  "), "My-Exp-v1")

    def test_sanitize_collapse_dashes(self):
        self.assertEqual(_sanitize_run_name_part("a---b***c"), "a-b-c")

    def test_sanitize_empty(self):
        self.assertEqual(_sanitize_run_name_part("   "), "run")

    def test_compose_without_suffix(self):
        args = DummyArgs("cifar10", "mlp", 8, 0.01)
        name = _compose_run_name(args)
        self.assertTrue(name.startswith("cifar10_mlp_b8_lr0.01"))

    def test_compose_with_suffix(self):
        args = DummyArgs("cifar10", "mlp", 8, 0.01, "my exp v2!")
        name = _compose_run_name(args)
        self.assertTrue(name.startswith("cifar10_mlp_b8_lr0.01"))
        self.assertTrue(name.endswith("my-exp-v2"))


if __name__ == '__main__':
    unittest.main()
