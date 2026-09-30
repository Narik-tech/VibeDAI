import copy
from dataclasses import asdict
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from neural.encoding import ENCODING_VERSION, encode_position
from neural.policy import MOVE_FEATURES, POLICY_VERSION, encode_moves, encode_policy_records

ROOT = Path(__file__).resolve().parents[1]


def position():
    squares = [[12, 0, 0], [0, 4, 0], [0, 0, 11]]
    return {"board": [[copy.deepcopy(squares) for _ in range(3)]], "action": 2}


def moves():
    return [[[0, 2, 1, 1], [0, 2, 2, 1]], [[0, 2, 1, 1], [0, 0, 2, 1]]]


class DescriptorTests(unittest.TestCase):
    def test_soft_targets_validate_and_preserve_legacy_records(self):
        label = {"moves": moves(), "target": 1, "targetVersion": 1, "targetWeights": [3., 1.]}
        example = encode_policy_records(position(), [label])[0]
        self.assertEqual(example[3], [.75, .25])
        self.assertEqual(len(encode_policy_records(position(), [{"moves": moves(), "target": 1}])[0]), 3)
        for changes in ({"targetWeights": [0., 0.]}, {"targetWeights": [-1, 1]},
                        {"targetWeights": [1]}, {"targetWeights": [math.nan, 1]},
                        {"targetWeights": [True, 1]}, {"targetWeights": None},
                        {"targetVersion": 2}, {"targetVersion": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                encode_policy_records(position(), [{**label, **changes}])
        del label["targetVersion"]
        with self.assertRaises(ValueError):
            encode_policy_records(position(), [label])

    def test_temporal_coordinates_and_special_endpoints_do_not_alias(self):
        features = encode_moves(position(), moves())
        self.assertTrue(all(len(row) == MOVE_FEATURES for row in features))
        self.assertNotEqual(features[0], features[1])
        self.assertEqual(features[0][-2], 0)
        self.assertEqual(features[1][-2], 1)
        self.assertEqual(features[1][-5], -2)
        promoted = copy.deepcopy(moves()[0])
        promoted[1].append(10)
        castle = copy.deepcopy(moves()[0]) + [[0, 2, 0, 0], [0, 2, 0, 1]]
        self.assertNotEqual(encode_moves(position(), [promoted])[0], features[0])
        self.assertNotEqual(encode_moves(position(), [castle])[0], features[0])

    def test_rejects_bad_descriptors_and_target_indices(self):
        for malformed in ([], [None], [[[0, 2, 1, 1]]], [[[0, 2, 1, 1], [0, 99, 0, 0]]],
                          [[[0, 2, 1, 1], [0, 0, True, 0]]]):
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                encode_moves(position(), malformed)
        for target in (-1, 2, True, "0"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                encode_policy_records(position(), [{"moves": moves(), "target": target}])


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class PolicyModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)

    def test_policy_loss_learns_mask_and_encodes_position_once(self):
        import torch
        from neural.model import ModelConfig, TransformerValue, collate, predict_policy
        from neural.policy import collate_moves, policy_loss
        torch.manual_seed(5)
        model = TransformerValue(ModelConfig(width=32, heads=4, layers=1, feedforward=64,
                                             max_tokens=64, dropout=0, policy_head=True))
        device = torch.device("cpu")
        encoded = encode_position(position(), 64)
        features = encode_moves(position(), moves())
        examples = [(encoded, features, 1), (encoded, features[:1], 0)]
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        before = policy_loss(model, examples, device).item()
        for _ in range(10):
            optimizer.zero_grad()
            loss = policy_loss(model, examples, device)
            loss.backward()
            optimizer.step()
        self.assertLess(policy_loss(model, examples, device).item(), before)
        inputs, mask = collate_moves([features, features[:1]], device)
        logits = model.score_moves(model.encode(*collate([encoded, encoded], device)), inputs, mask)
        self.assertEqual(logits[1, 1].item(), -math.inf)
        self.assertIsNone(predict_policy(model, position(), moves(), device))
        model.policy_trained_steps = 10
        with patch.object(model, "encode", wraps=model.encode) as encode:
            scores = predict_policy(model, position(), moves(), device)
            self.assertEqual(encode.call_count, 1)
        self.assertGreater(scores[1], scores[0])
        reversed_scores = predict_policy(model, position(), list(reversed(moves())), device)
        self.assertEqual(scores, list(reversed(reversed_scores)))

    def test_soft_loss_never_penalizes_unobserved_candidates(self):
        import torch
        from neural.policy import policy_example_losses, policy_choice_count
        logits = torch.tensor([[2., 1., 80.], [1., 80., -80.]], requires_grad=True)
        examples = [(None, [None] * 3, 0, [.7, .3, 0.]),
                    (None, [None] * 3, 0, [1., 0., 0.])]
        mask = torch.zeros((2, 3), dtype=torch.bool)
        loss = policy_example_losses(logits, mask, examples)
        expected = -(torch.tensor([.7, .3]) * torch.tensor([2., 1.]).log_softmax(-1)).sum()
        self.assertAlmostEqual(loss[0].item(), expected.item())
        self.assertEqual(loss[1].item(), 0.)
        loss.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad[0, 2].item(), 0.)
        self.assertEqual(logits.grad[1].abs().sum().item(), 0.)
        self.assertEqual(policy_choice_count(examples[0]), 2)
        self.assertEqual(policy_choice_count(examples[1]), 1)
        self.assertEqual(policy_choice_count(examples[0][:3]), 3)

    def test_policy_evaluation_measures_full_candidate_retention_excluding_forced_prefixes(self):
        import torch
        from types import SimpleNamespace
        from neural.evaluate import evaluate_policy_examples

        class RankedModel(torch.nn.Module):
            config = SimpleNamespace(policy_head=True)
            policy_trained_steps = 1

            def encode(self, *batch):
                return batch[0]

            def score_moves(self, context, features, mask):
                return torch.arange(features.shape[1], dtype=torch.float32).expand(features.shape[0], -1)

        encoded = encode_position(position(), 64)
        features = encode_moves(position(), moves() * 10)
        model = RankedModel()
        examples = [(encoded, features, 0, [.25] + [0.] * 18 + [.75]),
                    (encoded, features, 12), (encoded, features, 19), (encoded, features[:1], 0)]
        metrics = evaluate_policy_examples(model, iter(examples), torch.device("cpu"), batch_size=2)
        self.assertTrue(model.training)
        self.assertEqual(metrics["prefixes"], 4)
        self.assertEqual(metrics["decisionPrefixes"], 3)
        self.assertEqual(metrics["forcedPrefixes"], 1)
        self.assertAlmostEqual(metrics["top1Accuracy"], 1 / 3)
        self.assertAlmostEqual(metrics["top8Retention"], 2 / 3)
        self.assertAlmostEqual(metrics["top16Retention"], 2 / 3)
        self.assertTrue(math.isfinite(metrics["crossEntropy"]))
        self.assertTrue(math.isfinite(metrics["supervisedLoss"]))
        model.policy_trained_steps = 0
        self.assertFalse(evaluate_policy_examples(model, examples, torch.device("cpu"))["available"])

    def test_training_records_keep_soft_targets(self):
        from neural.train import records
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "soft.jsonl"
            row = {"position": position(), "value": 0, "policy": [{"moves": moves(), "target": 0,
                   "targetVersion": 1, "targetWeights": [.8, .2]}]}
            data.write_text(json.dumps(row) + "\n")
            example = next(records(data, 64, include_policy=True))[2][0]
            self.assertEqual(example[3], [.8, .2])

    def test_legacy_and_initialized_heads_never_report_policy_support(self):
        import torch
        from neural.model import ARCHITECTURE, ModelConfig, TransformerValue, load_checkpoint, metadata, predict_policy
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            for policy_head in (False, True):
                model = TransformerValue(ModelConfig(width=32, heads=4, layers=1, feedforward=64, policy_head=policy_head))
                config = asdict(model.config)
                if not policy_head:
                    del config["policy_head"]  # Exact legacy checkpoint format.
                payload = {"architecture": ARCHITECTURE, "encodingVersion": ENCODING_VERSION,
                           "config": config, "state_dict": model.state_dict(), "trainedSteps": 2}
                torch.save(payload, path)
                loaded, checkpoint = load_checkpoint(path, torch.device("cpu"))
                self.assertFalse(metadata(loaded, checkpoint, path)["policyAvailable"])
                self.assertIsNone(predict_policy(loaded, position(), moves(), torch.device("cpu")))
                payload["policyTrainedSteps"] = 1
                payload["policyVersion"] = 999
                torch.save(payload, path)
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    load_checkpoint(path, torch.device("cpu"))

    def test_cli_upgrades_legacy_checkpoint_and_service_validates_policy(self):
        import torch
        from neural.model import load_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            data, output = directory / "data.jsonl", directory / "model.pt"
            row = {"position": position(), "value": 10}
            data.write_text(json.dumps(row) + "\n", encoding="utf-8")
            command = [sys.executable, "neural/train.py", "--data", str(data), "--output", str(output),
                       "--device", "cpu", "--steps", "2", "--batch-size", "1", "--width", "32",
                       "--heads", "4", "--layers", "1", "--feedforward", "64", "--max-tokens", "64"]
            first = subprocess.run(command, capture_output=True, text=True, cwd=ROOT, timeout=60)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(torch.load(output, weights_only=True)["policyTrainedSteps"], 0)
            row.update(policyVersion=POLICY_VERSION, policy=[{"moves": moves(), "target": 1}])
            data.write_text(json.dumps(row) + "\n", encoding="utf-8")
            for expected in (2, 4):
                if expected == 4:
                    row["policy"][0].update(targetVersion=1, targetWeights=[.1, .9])
                    data.write_text(json.dumps(row) + "\n", encoding="utf-8")
                trained = subprocess.run(command + ["--resume", str(output)], capture_output=True, text=True, cwd=ROOT, timeout=60)
                self.assertEqual(trained.returncode, 0, trained.stderr)
                self.assertEqual(json.loads(trained.stdout.splitlines()[-1])["policyTrainedSteps"], expected)
                loaded, _ = load_checkpoint(output, torch.device("cpu"))
                self.assertEqual(loaded.policy_trained_steps, expected)
            evaluated = subprocess.run([sys.executable, "neural/evaluate.py", "--data", str(data),
                                        "--checkpoints", str(output), "--device", "cpu"],
                                       capture_output=True, text=True, cwd=ROOT, timeout=60)
            self.assertEqual(evaluated.returncode, 0, evaluated.stderr)
            metrics = json.loads(evaluated.stdout)["checkpoints"][0]["policy"]
            self.assertEqual(metrics["softPrefixes"], 1)
            self.assertEqual(metrics["top8Retention"], 1.)
            self.assertEqual(metrics["top16Retention"], 1.)
            requests = [dict(id=1, type="policy", position=position(), moves=moves()),
                        dict(id=2, type="policy", position=position(), moves=[[]]),
                        dict(id=3, type="unknown", positions=[position()])]
            service = subprocess.run([sys.executable, "neural/service.py", "--checkpoint", str(output), "--device", "cpu"],
                                     input="".join(json.dumps(request) + "\n" for request in requests),
                                     capture_output=True, text=True, cwd=ROOT, timeout=60)
            self.assertEqual(service.returncode, 0, service.stderr)
            messages = [json.loads(line) for line in service.stdout.splitlines()]
            self.assertTrue(messages[0]["model"]["policyAvailable"])
            self.assertEqual(len(messages[1]["scores"]), 2)
            self.assertTrue(all(math.isfinite(score) for score in messages[1]["scores"]))
            self.assertIn("error", messages[2])
            self.assertIn("error", messages[3])


if __name__ == "__main__":
    unittest.main()
