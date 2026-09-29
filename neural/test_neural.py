"""Run with python -m unittest neural.test_neural (model tests require PyTorch)."""

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

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from neural.encoding import ENCODING_VERSION, MAX_TOKENS, encode_position


def position():
    return {"board": [[[[12, 0], [0, -11]], [[12, 2], [0, -11]]]], "action": 1, "promotions": [9, 10]}


class EncodingTests(unittest.TestCase):
    def test_history_geometry_unmoved_and_side_survive(self):
        encoded = encode_position(position())
        self.assertEqual(len(encoded.global_features), 32)
        self.assertFalse(encoded.context["truncated"])
        self.assertTrue(any(c[1] == 11 and c[2] == 1 for c in encoded.categories))
        self.assertEqual({c[1] for c in encoded.coordinates[1:]}, {0, 1})
        self.assertEqual({c[4] for c in encoded.coordinates[1:]}, {0, 1})
        changed = position()
        changed["board"][0][0][0][1] = 4
        self.assertNotEqual(encode_position(changed).categories, encoded.categories)
        changed = position()
        changed["action"] = 2
        self.assertNotEqual(encode_position(changed).global_features, encoded.global_features)

    def test_nearest_history_boards_preserved_without_partial_boards(self):
        source = position()
        source["board"][0] = [copy.deepcopy(source["board"][0][1]) for _ in range(30)]
        encoded = encode_position(source, 16)
        self.assertEqual(encoded.context["tokens"], 13)
        self.assertTrue(encoded.context["truncated"])
        self.assertFalse(encoded.context["frontierTruncated"])
        self.assertEqual(sum(c[4] == 1 for c in encoded.categories), 4)
        self.assertEqual(encoded, encode_position(source, 16))
        historic_times = {coordinate[1] for category, coordinate in zip(encoded.categories[1:], encoded.coordinates[1:]) if not category[4]}
        self.assertEqual(historic_times, {27, 28})
        for time in (27, 28, 29):
            self.assertEqual(sum(coordinate[1] == time for coordinate in encoded.coordinates[1:]), 4)

    def test_frontier_overflow_drops_whole_board_and_is_disclosed(self):
        squares = [[2] * 8 for _ in range(8)]
        squares[0][0], squares[7][7] = 12, 11
        encoded = encode_position({"board": [[squares]], "action": 0}, 16)
        self.assertEqual(encoded.context["tokens"], 1)
        self.assertEqual(encoded.context["totalTokens"], 66)
        self.assertTrue(encoded.context["truncated"])
        self.assertTrue(encoded.context["frontierTruncated"])
        self.assertEqual(encoded.global_features[7], 1)

    def test_default_context_retains_exactly_4096_tokens_then_drops_farthest_board(self):
        # Each complete board costs three tokens: its marker and two kings.
        source = {"board": [[[[12, 11]] for _ in range(1365)]], "action": 0}
        original = copy.deepcopy(source)
        encoded = encode_position(source)
        self.assertEqual(encoded.context["tokens"], MAX_TOKENS)
        self.assertFalse(encoded.context["truncated"])
        self.assertEqual(source, original)

        source["board"][0].append([[12, 11]])
        source["action"] = 1
        encoded = encode_position(source)
        self.assertEqual(encoded.context["tokens"], MAX_TOKENS)
        self.assertEqual(encoded.context["totalTokens"], MAX_TOKENS + 3)
        self.assertTrue(encoded.context["truncated"])
        self.assertFalse(encoded.context["frontierTruncated"])
        self.assertEqual({coordinate[1] for coordinate in encoded.coordinates[1:]}, set(range(1, 1366)))
        self.assertEqual(encoded.global_features[5], MAX_TOKENS / (MAX_TOKENS + 3))

    def test_token_budget_bounds(self):
        self.assertEqual(encode_position(position(), MAX_TOKENS), encode_position(position()))
        for invalid in (15, 4097, True, 4096.0):
            with self.subTest(max_tokens=invalid), self.assertRaises(ValueError):
                encode_position(position(), invalid)

    def test_even_timelines_use_upstream_coordinates(self):
        source = position()
        source["board"] = [None, source["board"][0], source["board"][0]]
        encoded = encode_position(source)
        self.assertEqual({coordinate[0] for coordinate in encoded.coordinates[1:]}, {-1, 0})
        self.assertEqual(encoded.global_features[6], 1)

    def test_invalid_input_rejected(self):
        for invalid in ({}, {"board": [], "action": 0}, {"board": [[[[25]]]], "action": 0},
                        {"board": [[[[True]]]], "action": 0}, {"board": [[[[0], [0, 0]]]], "action": 0}):
            with self.assertRaises(ValueError):
                encode_position(invalid)


class TrainingDataTests(unittest.TestCase):
    def test_legacy_records_and_optional_weights_preserve_evaluation_pairs(self):
        from neural.train import dataset_weight_mean, records
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "train.jsonl"
            rows = [{"position": position(), "value": 400},
                    {"position": position(), "value": -400, "weight": 3}]
            data.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            pairs = list(records(data, 64))
            weighted = list(records(data, 64, include_weight=True))
            self.assertEqual([len(row) for row in pairs], [2, 2])
            self.assertEqual([row[:2] for row in weighted], pairs)
            self.assertEqual([row[2] for row in weighted], [1, 3])
            self.assertAlmostEqual(pairs[0][1], math.tanh(0.4))
            with patch("neural.encoding.encode_position", side_effect=AssertionError("must not encode during weight scan")):
                self.assertEqual(dataset_weight_mean(data), 2)

    def test_invalid_weights_fail_with_file_and_line(self):
        from neural.train import dataset_weight_mean, records
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "train.jsonl"
            legacy = json.dumps({"position": position(), "value": 400})
            for weight in (0, -1, True, None, "1", [], {}, float("nan"), float("inf"), -float("inf")):
                with self.subTest(weight=weight):
                    invalid = json.dumps({"position": position(), "value": 400, "weight": weight})
                    data.write_text(legacy + "\n" + invalid + "\n", encoding="utf-8")
                    for read in (lambda: list(records(data, 64, include_weight=True)), lambda: dataset_weight_mean(data)):
                        with self.assertRaisesRegex(ValueError, "train.jsonl:2: weight must be positive and finite"):
                            read()
            data.write_text("\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "training file has no examples"):
                dataset_weight_mean(data)


class ServiceBatchTests(unittest.TestCase):
    def test_default_batch_size_tracks_checkpoint_size(self):
        from neural.service import inference_batch_size
        self.assertEqual(inference_batch_size(694_017), 16)
        self.assertEqual(inference_batch_size(9_999_999), 16)
        self.assertEqual(inference_batch_size(10_000_000), 1)
        self.assertEqual(inference_batch_size(20_000_257), 1)

    def test_explicit_batch_sizes_override_both_defaults(self):
        from neural.service import inference_batch_size
        for parameters in (694_017, 20_000_257):
            for requested in (1, 16, 128):
                self.assertEqual(inference_batch_size(parameters, requested), requested)

    def test_invalid_batch_sizes_are_rejected(self):
        from neural.service import inference_batch_size
        for requested in (0, -1, 129, 1.5, True, "16"):
            with self.subTest(requested=requested), self.assertRaisesRegex(ValueError, "batch-size"):
                inference_batch_size(20_000_257, requested)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)

    def test_training_updates_weights_and_checkpoint_roundtrips(self):
        import torch
        from neural.model import ARCHITECTURE, ModelConfig, TransformerValue, collate, load_checkpoint, predict
        torch.manual_seed(23)
        model = TransformerValue(ModelConfig(width=32, heads=4, layers=1, feedforward=64, max_tokens=64, dropout=0))
        batch = collate([encode_position(position(), 64)], torch.device("cpu"))
        target = torch.tensor([0.5])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
        before = torch.nn.functional.mse_loss(model(*batch), target).item()
        for _ in range(12):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(*batch), target)
            loss.backward()
            optimizer.step()
        after = torch.nn.functional.mse_loss(model(*batch), target).item()
        self.assertLess(after, before)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            torch.save({"architecture": ARCHITECTURE, "encodingVersion": ENCODING_VERSION,
                        "config": asdict(model.config), "state_dict": model.state_dict(), "trainedSteps": 12}, checkpoint)
            loaded, _ = load_checkpoint(checkpoint, torch.device("cpu"))
            self.assertEqual(predict(model, [position()], torch.device("cpu")), predict(loaded, [position()], torch.device("cpu")))
            response = subprocess.run([sys.executable, "neural/service.py", "--checkpoint", str(checkpoint), "--device", "cpu"],
                                      input=json.dumps({"id": "test", "positions": [position()]}) + "\n" + json.dumps({"id": "bad", "positions": [{}]}) + "\n",
                                      capture_output=True, text=True, cwd=ROOT, timeout=60)
            self.assertEqual(response.returncode, 0, response.stderr)
            messages = [json.loads(line) for line in response.stdout.splitlines()]
            self.assertTrue(messages[0]["ready"])
            self.assertEqual(messages[0]["batchSize"], 16)
            self.assertEqual(messages[1]["id"], "test")
            self.assertTrue(math.isfinite(messages[1]["values"][0]))
            self.assertIn("error", messages[2])
            for requested, valid in (("2", True), ("0", False)):
                with self.subTest(batch_size=requested):
                    response = subprocess.run([sys.executable, "neural/service.py", "--checkpoint", str(checkpoint),
                                               "--device", "cpu", "--batch-size", requested],
                                              input="", capture_output=True, text=True, cwd=ROOT, timeout=60)
                    ready = json.loads(response.stdout.splitlines()[0])
                    self.assertEqual(ready["ready"], valid)
                    self.assertEqual(response.returncode, 0 if valid else 1, response.stderr)
                    if valid:
                        self.assertEqual(ready["batchSize"], 2)
                    else:
                        self.assertIn("batch-size", ready["error"])

    def test_padding_does_not_change_single_position_value(self):
        import torch
        from neural.model import ModelConfig, TransformerValue, predict
        torch.manual_seed(11)
        model = TransformerValue(ModelConfig(width=32, heads=4, layers=1, feedforward=64, max_tokens=64, dropout=0))
        larger = position()
        larger["board"][0].extend(copy.deepcopy(larger["board"][0]))
        alone, _ = predict(model, [position()], torch.device("cpu"))
        batched, _ = predict(model, [position(), larger], torch.device("cpu"))
        self.assertAlmostEqual(alone[0], batched[0], delta=0.01)

    def test_missing_and_untrained_checkpoints_fail_closed(self):
        import torch
        from neural.model import ARCHITECTURE, load_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "absent.pt"
            with self.assertRaises(FileNotFoundError):
                load_checkpoint(path, torch.device("cpu"))
            torch.save({"architecture": ARCHITECTURE, "encodingVersion": ENCODING_VERSION, "trainedSteps": 0}, path)
            with self.assertRaisesRegex(ValueError, "no completed training steps"):
                load_checkpoint(path, torch.device("cpu"))

    def test_training_cli_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "train.jsonl"
            output = Path(directory) / "model.pt"
            data.write_text(json.dumps({"position": position(), "value": 400}) + "\n", encoding="utf-8")
            command = [sys.executable, "neural/train.py", "--data", str(data), "--output", str(output), "--device", "cpu",
                       "--steps", "2", "--batch-size", "2", "--width", "32", "--heads", "4", "--layers", "1", "--feedforward", "64", "--max-tokens", "64"]
            first = subprocess.run(command, capture_output=True, text=True, cwd=ROOT, timeout=60)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(json.loads(first.stdout.splitlines()[0])["sampleWeightMean"], 1)
            self.assertEqual(json.loads(first.stdout.splitlines()[-1])["trainedSteps"], 2)
            second = subprocess.run(command + ["--resume", str(output)], capture_output=True, text=True, cwd=ROOT, timeout=60)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(json.loads(second.stdout.splitlines()[-1])["trainedSteps"], 4)

    def test_unequal_length_games_have_equal_loss_and_gradient_contributions(self):
        import torch
        from neural.train import dataset_weight_mean, records, weighted_mse
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "train.jsonl"
            # One short game's position and three long-game positions, each
            # game contributing a total weight of two after normalization.
            raw = [{"position": position(), "value": math.atanh(0.5) * 1000,
                    "gameId": game, "weight": weight}
                   for game, weight in [("short", 2), ("long", 2 / 3), ("long", 2 / 3), ("long", 2 / 3)]]
            data.write_text("\n".join(json.dumps(row) for row in raw) + "\n", encoding="utf-8")
            rows = list(records(data, 64, include_weight=True))
            mean = dataset_weight_mean(data)
            weights = torch.tensor([row[2] / mean for row in rows])
            targets = torch.tensor([row[1] for row in rows])
            predictions_by_game = torch.zeros(2, requires_grad=True)
            loss = weighted_mse(predictions_by_game[torch.tensor([0, 1, 1, 1])], targets, weights)
            self.assertAlmostEqual(loss.item(), 0.25)
            loss.backward()
            torch.testing.assert_close(predictions_by_game.grad, torch.tensor([-0.5, -0.5]))

            # The same per-game contribution survives batches of one; a
            # batch-local division by sum(weights) would fail this assertion.
            batch_one_gradients = []
            batch_one_losses = []
            for target, weight in zip(targets, weights):
                prediction = torch.zeros(1, requires_grad=True)
                item_loss = weighted_mse(prediction, target.reshape(1), weight.reshape(1))
                item_loss.backward()
                batch_one_losses.append(item_loss.item())
                batch_one_gradients.append(prediction.grad.item())
            self.assertAlmostEqual(batch_one_losses[0], sum(batch_one_losses[1:]))
            self.assertAlmostEqual(batch_one_gradients[0], sum(batch_one_gradients[1:]), places=6)
            self.assertAlmostEqual(sum(batch_one_losses) / len(rows), loss.item())

    def test_weighted_training_cli_and_resume_use_dataset_normalization(self):
        import torch
        from neural.model import ModelConfig, TransformerValue, collate
        with tempfile.TemporaryDirectory() as directory:
            data, output = Path(directory) / "train.jsonl", Path(directory) / "model.pt"
            rows = [{"position": position(), "value": 400, "weight": 1},
                    {"position": position(), "value": -400, "weight": 3}]
            data.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            torch.manual_seed(42)
            model = TransformerValue(ModelConfig(width=32, heads=4, layers=1, feedforward=64, max_tokens=64, dropout=0))
            with torch.no_grad():
                prediction = model(*collate([encode_position(position(), 64)], torch.device("cpu"))).item()
            expected_first_loss = 0.5 * (prediction - math.tanh(0.4)) ** 2
            command = [sys.executable, "neural/train.py", "--data", str(data), "--output", str(output), "--device", "cpu",
                       "--steps", "2", "--batch-size", "1", "--shuffle-buffer", "1", "--dropout", "0",
                       "--width", "32", "--heads", "4", "--layers", "1", "--feedforward", "64", "--max-tokens", "64"]
            first = subprocess.run(command, capture_output=True, text=True, cwd=ROOT, timeout=60)
            self.assertEqual(first.returncode, 0, first.stderr)
            events = [json.loads(line) for line in first.stdout.splitlines()]
            self.assertEqual(events[0]["sampleWeightMean"], 2)
            self.assertAlmostEqual(next(event["loss"] for event in events if event["event"] == "train"), expected_first_loss, places=6)
            checkpoint = torch.load(output, weights_only=True)
            self.assertEqual(checkpoint["trainedSteps"], 2)
            self.assertEqual(checkpoint["training"]["sampleWeightMean"], 2)
            self.assertEqual(checkpoint["training"]["lossWeighting"], "dataset-normalized sample weights")
            # Resuming scans the current replay, not the previous dataset's mean.
            rows[1]["weight"] = 7
            data.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            second = subprocess.run(command + ["--resume", str(output)], capture_output=True, text=True, cwd=ROOT, timeout=60)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(json.loads(second.stdout.splitlines()[0])["sampleWeightMean"], 4)
            checkpoint = torch.load(output, weights_only=True)
            self.assertEqual(checkpoint["trainedSteps"], 4)
            self.assertEqual(checkpoint["training"]["sampleWeightMean"], 4)

    def test_best_output_preserves_better_resumed_baseline_and_evaluation(self):
        import torch
        from neural.model import ARCHITECTURE, ModelConfig, TransformerValue
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            data, validation = directory / "train.jsonl", directory / "validation.jsonl"
            baseline, latest, best = (directory / name for name in ("baseline.pt", "latest.pt", "best.pt"))
            data.write_text(json.dumps({"position": position(), "value": 1000}) + "\n", encoding="utf-8")
            validation.write_text((json.dumps({"position": position(), "value": 0}) + "\n") * 3, encoding="utf-8")
            model = TransformerValue(ModelConfig(width=32, heads=4, layers=1, feedforward=64, max_tokens=64, dropout=0))
            with torch.no_grad():
                model.head[3].weight.zero_()
                model.head[3].bias.zero_()
            torch.save({"architecture": ARCHITECTURE, "encodingVersion": ENCODING_VERSION,
                        "config": asdict(model.config), "state_dict": model.state_dict(),
                        "trainedSteps": 4, "examplesSeen": 8, "label": "fixture",
                        "training": {"note": "baseline provenance"}}, baseline)
            command = [sys.executable, "neural/train.py", "--data", str(data), "--validation-data", str(validation),
                       "--resume", str(baseline), "--output", str(latest), "--best-output", str(best),
                       "--device", "cpu", "--steps", "2", "--batch-size", "2", "--save-every", "1"]
            trained = subprocess.run(command, capture_output=True, text=True, cwd=ROOT, timeout=60)
            self.assertEqual(trained.returncode, 0, trained.stderr)
            result = json.loads(trained.stdout.splitlines()[-1])
            self.assertEqual(result["trainedSteps"], 6)
            self.assertEqual(result["baselineValidation"]["samples"], 3)
            self.assertLessEqual(result["bestValidation"]["mse"], result["baselineValidation"]["mse"])
            self.assertGreater(result["validation"]["mse"], result["bestValidation"]["mse"])
            best_payload = torch.load(best, weights_only=True)
            self.assertEqual(best_payload["trainedSteps"], 4)
            self.assertEqual(best_payload["training"]["note"], "baseline provenance")
            self.assertEqual(torch.load(latest, weights_only=True)["trainedSteps"], 6)
            evaluated = subprocess.run([sys.executable, "neural/evaluate.py", "--data", str(validation),
                                        "--checkpoints", str(best), str(latest), "--device", "cpu", "--batch-size", "2"],
                                       capture_output=True, text=True, cwd=ROOT, timeout=60)
            self.assertEqual(evaluated.returncode, 0, evaluated.stderr)
            report = json.loads(evaluated.stdout)
            self.assertEqual(len(report["checkpoints"]), 2)
            self.assertEqual(report["checkpoints"][0]["samples"], 3)
            self.assertEqual(report["checkpoints"][0]["normalizedMse"], 0)
            self.assertEqual(report["checkpoints"][0]["maeClippedCp"], 0)
            self.assertGreater(report["checkpoints"][1]["normalizedMse"], 0)


if __name__ == "__main__":
    unittest.main()
