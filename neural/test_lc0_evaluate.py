import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from .lc0_evaluate import dataset_audit, evaluate_checkpoint, evaluate_rows, file_sha256, holdout_audit
from .lc0_model import ARCHITECTURE
from .test_lc0_transfer import position, small_model


class FixedModel(torch.nn.Module):
    """Exact metrics fixture: zero value; SUBMIT scores two, other moves zero."""
    def encode_position(self, _):
        return SimpleNamespace(context={"truncated": False, "frontierTruncated": False})

    def collate(self, rows, device):
        return (torch.zeros(len(rows), device=device),)

    def forward(self, value):
        return value

    def encode(self, value):
        return value

    def score_moves(self, context, features, mask):
        return (features[:, :, -1] < 0).float() * 2


def row(stage="spatial", group="validation", value=1000):
    p = position()
    move = [[0, 0, 1, 1], [0, 0, 2, 1]]
    return {"position": p, "value": value, "group": group, "gameId": group + ":game", "stage": stage,
            "lc0PolicyVersion": 1, "lc0Policy": [{"moves": [move, None], "target": 1},
                                                  {"moves": [move, None], "target": 0},
                                                  {"moves": [None], "target": 0}]}


class EvaluateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_every_nontrivial_prefix_contributes_singletons_do_not_inflate_accuracy(self):
        model = FixedModel().train()
        metrics = evaluate_rows(model, iter([row()]), torch.device("cpu"), batch_size=1)
        self.assertTrue(model.training)
        self.assertEqual(metrics["value"]["samples"], 1)
        self.assertAlmostEqual(metrics["value"]["normalizedMse"], math.tanh(1) ** 2, places=6)
        self.assertAlmostEqual(metrics["value"]["maeClippedCp"], 1000, places=3)
        policy = metrics["policy"]
        self.assertEqual(policy["prefixes"], 3)
        self.assertEqual(policy["decisionPrefixes"], 2)
        self.assertEqual(policy["forcedPrefixes"], 1)
        self.assertEqual(policy["submitTargets"], 2)
        self.assertEqual(policy["decisionSubmitTargets"], 1)
        self.assertEqual(policy["top1Accuracy"], .5)
        self.assertAlmostEqual(policy["crossEntropy"], math.log1p(math.exp(-2)) + 1, places=6)
        self.assertEqual(policy["uniformTop1Accuracy"], .5)
        self.assertAlmostEqual(policy["uniformCrossEntropy"], math.log(2))
        self.assertEqual(metrics["stages"]["spatial"]["policy"], policy)

    def test_stage_temporal_counts_and_padding_are_batch_invariant(self):
        first, second = row(), row(stage="temporal", value=-500)
        second["position"]["board"].append(copy.deepcopy(second["position"]["board"][0]))
        jump = [[0, 0, 1, 1], [1, 0, 2, 1]]
        second["lc0Policy"] = [{"moves": [jump, None, [[0, 0, 1, 1], [0, 0, 2, 1]]], "target": 0}]
        one = evaluate_rows(FixedModel(), [first, second], torch.device("cpu"), batch_size=1)
        many = evaluate_rows(FixedModel(), [first, second], torch.device("cpu"), batch_size=4)
        self.assertEqual(one, many)
        self.assertEqual(one["policy"]["temporalTargets"], 1)
        self.assertEqual(one["policy"]["decisionTemporalTargets"], 1)
        self.assertEqual(one["stages"]["temporal"]["value"]["samples"], 1)
        self.assertEqual(one["stages"]["spatial"]["value"]["samples"], 1)

    def test_malformed_later_prefix_is_not_hidden_by_sampling(self):
        malformed = row()
        malformed["lc0Policy"][-1]["target"] = 99
        model = FixedModel().train()
        with self.assertRaisesRegex(ValueError, "target"):
            evaluate_rows(model, [malformed], torch.device("cpu"))
        self.assertTrue(model.training)

    def test_group_game_and_position_audit_and_checkpoint_roundtrip_do_not_mutate_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            train_path, data_path, checkpoint_path = root / "train.jsonl", root / "val.jsonl", root / "model.pt"
            train_row, val_row = row(group="train"), row()
            train_row["position"]["board"][0][0][1][1] = 0
            train_path.write_text(json.dumps(train_row) + "\n", encoding="utf-8")
            data_path.write_text(json.dumps(val_row) + "\n", encoding="utf-8")
            training, validation = dataset_audit(train_path), dataset_audit(data_path)
            audit = holdout_audit(validation, training, file_sha256(train_path))
            self.assertTrue(audit["verifiedAgainstCheckpointTraining"])
            overlap = holdout_audit(validation, validation)
            self.assertFalse(overlap["disjoint"])
            self.assertEqual(overlap["overlappingGroups"], ["validation"])
            self.assertEqual(overlap["overlappingPositions"], 1)
            self.assertEqual(overlap["overlappingGameIds"], ["validation:game"])
            parity_path = root / "parity.jsonl"
            parity = row(group="different")
            parity["position"]["action"] += 2
            parity_path.write_text(json.dumps(parity) + "\n", encoding="utf-8")
            self.assertEqual(holdout_audit(validation, dataset_audit(parity_path))["overlappingPositions"], 1)
            prefix_path = root / "prefix.jsonl"
            prefix_row = copy.deepcopy(train_row)
            prefix_row["lc0Policy"][0]["position"] = val_row["position"]
            prefix_path.write_text(json.dumps(prefix_row) + "\n", encoding="utf-8")
            self.assertEqual(holdout_audit(validation, dataset_audit(prefix_path))["overlappingPositions"], 1)
            model = small_model()
            payload = {"architecture": ARCHITECTURE, "encodingVersion": 1, "config": asdict(model.config),
                       "backboneConfig": model.backbone.config, "state_dict": model.state_dict(),
                       "trainedSteps": 1, "policyVersion": 1, "policyTrainedSteps": 1, "baseline": model.baseline_metadata,
                       "training": {"data": str(train_path), "dataSha256": file_sha256(train_path),
                                    "validationSha256": file_sha256(data_path)}}
            torch.save(payload, checkpoint_path)
            before = {path: file_sha256(path) for path in (train_path, data_path, checkpoint_path)}
            report = evaluate_checkpoint(checkpoint_path, data_path, torch.device("cpu"), batch_size=2)
            self.assertTrue(report["matchesRecordedValidationHash"])
            self.assertTrue(report["holdoutAudit"]["verifiedAgainstCheckpointTraining"])
            self.assertEqual(report["model"]["baseline"], model.baseline_metadata)
            self.assertEqual(report["policy"]["decisionPrefixes"], 2)
            self.assertTrue(math.isfinite(report["policy"]["crossEntropy"]))
            self.assertEqual(before, {path: file_sha256(path) for path in before})
            with self.assertRaisesRegex(ValueError, "audit limit"):
                dataset_audit(data_path, max_rows=0)


if __name__ == "__main__":
    unittest.main()
