"""Architecture defaults and compatibility tests without allocating the 20M model."""

from dataclasses import asdict
from contextlib import redirect_stderr
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class ModelConfigTests(unittest.TestCase):
    def test_default_model_has_approximately_20m_parameters(self):
        import torch
        from neural.model import ModelConfig, TransformerValue
        config = ModelConfig()
        self.assertEqual((config.width, config.heads, config.layers, config.feedforward),
                         (512, 8, 6, 2048))
        # Meta tensors verify every actual parameter shape while avoiding a
        # roughly 80 MB allocation merely to count the default architecture.
        with torch.device("meta"):
            model = TransformerValue(config)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 19_142_657)
        with torch.device("meta"):
            policy_model = TransformerValue(ModelConfig(policy_head=True))
        self.assertEqual(sum(parameter.numel() for parameter in policy_model.parameters()), 20_000_257)

    def test_invalid_architecture_values_are_rejected(self):
        from neural.model import ModelConfig
        invalid = (
            {"width": 31}, {"width": 513}, {"width": True}, {"width": 512.0},
            {"heads": 0}, {"heads": 3}, {"heads": 12}, {"heads": True},
            {"width": 33, "heads": 8}, {"layers": 0}, {"layers": 9},
            {"layers": 6.0}, {"feedforward": 511}, {"feedforward": 2049},
            {"feedforward": True}, {"policy_head": 1},
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                ModelConfig(**arguments)

    def test_supported_architecture_boundaries(self):
        from neural.model import ModelConfig
        ModelConfig(width=32, heads=1, layers=1, feedforward=32)
        ModelConfig(width=512, heads=8, layers=8, feedforward=2048)
        for heads in (1, 2, 4, 8):
            with self.subTest(heads=heads):
                self.assertEqual(ModelConfig(heads=heads).heads, heads)

    def test_training_cli_uses_model_defaults_and_accepts_small_overrides(self):
        from neural.model import ModelConfig
        from neural.train import main
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data.jsonl"
            data.write_text(json.dumps({"position": {"board": [[[[12]]]], "action": 0}, "value": 0}) + "\n",
                            encoding="utf-8")
            base = ["train.py", "--data", str(data), "--device", "cpu"]
            small = ["--width", "32", "--heads", "4", "--layers", "1", "--feedforward", "64"]
            for arguments, expected in (([], ModelConfig()),
                                        (small, ModelConfig(width=32, heads=4, layers=1, feedforward=64))):
                with self.subTest(arguments=arguments):
                    # Stop immediately after real argument parsing/configuration;
                    # a CLI regression should not need 20M weights or an update.
                    with patch.object(sys, "argv", base + arguments), \
                            patch("neural.model.TransformerValue", side_effect=RuntimeError("configuration captured")) as constructor, \
                            redirect_stderr(io.StringIO()) as errors:
                        self.assertEqual(main(), 1)
                    self.assertEqual(constructor.call_args.args[0], expected)
                    self.assertIn("configuration captured", errors.getvalue())

    def test_previous_default_checkpoint_keeps_its_saved_architecture(self):
        import torch
        from neural.encoding import ENCODING_VERSION
        from neural.model import ARCHITECTURE, ModelConfig, TransformerValue, load_checkpoint, metadata
        torch.set_num_threads(2)
        config = ModelConfig(width=128, heads=4, layers=4, feedforward=384, max_tokens=512)
        original = TransformerValue(config)
        saved_config = asdict(config)
        saved_config.pop("policy_head")  # Earlier value-only checkpoints omit this field.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save({"architecture": ARCHITECTURE, "encodingVersion": ENCODING_VERSION,
                        "config": saved_config, "state_dict": original.state_dict(), "trainedSteps": 1}, path)
            loaded, checkpoint = load_checkpoint(path, torch.device("cpu"))
            self.assertEqual((loaded.config.width, loaded.config.heads, loaded.config.layers,
                              loaded.config.feedforward), (128, 4, 4, 384))
            self.assertEqual(metadata(loaded, checkpoint, path)["parameters"], 694_017)
            self.assertFalse(loaded.config.policy_head)
            for name, value in original.state_dict().items():
                self.assertTrue(torch.equal(value, loaded.state_dict()[name]), name)


if __name__ == "__main__":
    unittest.main()
