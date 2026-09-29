"""Representation-independent inference, replay, and LC0 checkpoint routing."""

from contextlib import redirect_stderr
from dataclasses import dataclass
import importlib.util
import io
import json
import math
from pathlib import Path
import random
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from neural.test_policy import moves, position


@dataclass
class CustomEncoding:
    value: float
    context: dict


def custom_model():
    import torch
    from neural.model import ModelConfig

    class RepresentationModel(torch.nn.Module):
        """A different input shape exposes accidental legacy encoder calls."""
        def __init__(self):
            super().__init__()
            self.config = ModelConfig(width=32, heads=1, layers=1, feedforward=32,
                                      max_tokens=64, policy_head=True)
            self.scale = torch.nn.Parameter(torch.tensor(0.25))
            self.policy_trained_steps = 1
            self.baseline_metadata = {"sha256": "test-baseline"}

        def encode_position(self, source):
            return CustomEncoding(float(source["action"]),
                                  {"truncated": False, "frontierTruncated": False, "custom": True})

        def collate(self, encoded, device):
            return (torch.tensor([item.value for item in encoded], device=device),)

        def encode(self, values):
            return self.scale * values

        def forward(self, values):
            return self.encode(values).tanh()

        def score_moves(self, context, features, padding_mask):
            logits = context.unsqueeze(1) * features[:, :, 33]
            return logits.masked_fill(padding_mask, -torch.inf)

    return RepresentationModel()


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class IntegrationTests(unittest.TestCase):
    def test_inference_policy_and_evaluation_use_model_representation(self):
        import torch
        from neural.evaluate import evaluate_records
        from neural.model import encode_for_model, metadata, predict, predict_policy
        from neural.policy import encode_moves, policy_loss

        model, device = custom_model(), torch.device("cpu")
        with patch("neural.model.encode_position", side_effect=AssertionError("legacy input used")), \
                patch("neural.model.collate", side_effect=AssertionError("legacy batch used")):
            values, contexts = predict(model, [position(), position()], device)
            self.assertEqual(values, [500.0, 500.0])
            self.assertTrue(all(context["custom"] for context in contexts))
            self.assertEqual(predict_policy(model, position(), moves(), device), [0.0, -1.0])
            encoded = encode_for_model(model, position())
            model.train()
            metrics = evaluate_records(model, [(encoded, math.tanh(0.5))], device)
            self.assertEqual(metrics["samples"], 1)
            self.assertAlmostEqual(metrics["normalizedMse"], 0, places=12)
            self.assertTrue(model.training)
            loss = policy_loss(model, [(encoded, encode_moves(position(), moves()), 1)], device)
            loss.backward()
            self.assertTrue(torch.isfinite(model.scale.grad).item())
            self.assertNotEqual(model.scale.grad.item(), 0)
        info = metadata(model, {"architecture": "5d-lc0-transfer-v1", "trainedSteps": 3}, "model.pt")
        self.assertEqual(info["architecture"], "5d-lc0-transfer-v1")
        self.assertEqual(info["baseline"], {"sha256": "test-baseline"})

    def test_replay_callback_encodes_value_and_component_prefixes(self):
        from neural.train import records, stream_training

        model = custom_model()
        prefix = position()
        prefix["action"] = 4
        row = {"position": position(), "value": 500, "weight": 2,
               "policy": [{"position": prefix, "moves": moves(), "target": 1}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            encoded, target, weight, policy = next(records(path, 64, include_weight=True,
                                                         include_policy=True, encoder=model.encode_position))
            self.assertEqual((encoded.value, weight, policy[0][0].value), (2, 2, 4))
            self.assertAlmostEqual(target, math.tanh(0.5))
            stream = stream_training(path, 64, 1, random.Random(3), encoder=model.encode_position)
            self.assertEqual([next(stream)[0].value for _ in range(3)], [2, 2, 2])

    def test_transfer_move_encoder_accepts_rules_supplied_submit(self):
        import torch
        from neural.lc0_encoding import encode_transfer_moves
        from neural.model import predict_policy

        model = custom_model()
        candidates = [*moves(), None]
        with self.assertRaises(ValueError):
            predict_policy(model, position(), candidates, torch.device("cpu"))
        model.encode_moves = encode_transfer_moves
        scores = predict_policy(model, position(), candidates, torch.device("cpu"))
        self.assertEqual(scores, [0.0, -1.0, 0.0])

    def test_checkpoint_routes_to_transfer_validator_and_preserves_failures(self):
        import torch
        from neural.model import load_checkpoint

        transfer = ModuleType("neural.lc0_model")
        transfer.load_transfer_checkpoint = Mock(return_value=custom_model())
        payload = {"architecture": "5d-lc0-transfer-v1", "trainedSteps": 3}
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"neural.lc0_model": transfer}):
            path = Path(directory) / "transfer.pt"
            torch.save(payload, path)
            loaded, checkpoint = load_checkpoint(path, torch.device("cpu"), max_tokens=128)
            self.assertIs(loaded, transfer.load_transfer_checkpoint.return_value)
            self.assertEqual(checkpoint, payload)
            transfer.load_transfer_checkpoint.assert_called_once_with(payload, torch.device("cpu"), max_tokens=128)
            transfer.load_transfer_checkpoint.side_effect = ValueError("invalid transfer weights")
            with self.assertRaisesRegex(ValueError, "invalid transfer weights"):
                load_checkpoint(path, torch.device("cpu"))

    def test_legacy_train_entrypoint_resumes_transfer_and_reports_errors(self):
        import torch
        from neural.train import main

        trainer = ModuleType("neural.lc0_train")
        trainer.train = Mock(return_value=0)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"neural.lc0_train": trainer}):
            path = Path(directory) / "transfer.pt"
            torch.save({"architecture": "5d-lc0-transfer-v1"}, path)
            with patch.object(sys, "argv", ["train.py", "--resume", str(path), "--steps", "7"]):
                self.assertEqual(main(), 0)
                self.assertEqual(trainer.train.call_args.args[0].steps, 7)
                trainer.train.side_effect = ValueError("invalid replay")
                errors = io.StringIO()
                with redirect_stderr(errors):
                    self.assertEqual(main(), 1)
                self.assertEqual(json.loads(errors.getvalue())["error"], "invalid replay")


if __name__ == "__main__":
    unittest.main()
