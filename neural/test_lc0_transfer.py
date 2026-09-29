"""Transfer contracts: spatial orientation, temporal identity, masks, learning."""
import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch
from .lc0_backbone import LC0Backbone, BACKBONE_VERSION
from .lc0_encoding import encode_transfer_position, encode_transfer_moves
from .lc0_model import LC0Transfer, TransferConfig, ARCHITECTURE
from .lc0_train import audit_data, check_holdout, train
from .model import load_checkpoint, predict, predict_policy


def small_model():
    config = {"version": BACKBONE_VERSION, "embedding_dim": 32, "positional_channels": 2,
              "embedding_ffn_dim": 48, "heads": 4, "default_activation": 1,
              "ffn_activation": 1, "smolgen_activation": 7,
              "encoders": [{"attention_dim": 32, "ffn_dim": 48, "smolgen": None}]}
    torch.manual_seed(7)
    return LC0Transfer(LC0Backbone(config), TransferConfig(width=32, heads=4, layers=1,
                       feedforward=64, max_boards=4, dropout=0.), {"source_sha256": "a" * 64})


def position():
    return {"board": [[[[12, 0, 0, 0], [0, -2, 0, 0], [0, 0, 1, 0], [0, 0, 0, 11]]]],
            "action": 0, "promotions": [10, 8, 6, 4]}


class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_plane_color_piece_order_and_history(self):
        p = position()
        white = encode_transfer_position(p)
        self.assertEqual(white.planes[0, 0, 1, 1], 1)  # White pawn, not bishop.
        self.assertEqual(white.planes[0, 6, 2, 2], 1)
        self.assertEqual(white.planes[0, 5, 0, 0], 1)
        self.assertEqual(white.unmoved[0, 9], 1)
        p["board"][0].append(p["board"][0][0])
        p["action"] = 1
        black = encode_transfer_position(p)
        self.assertEqual(black.planes[1, 0, 5, 2], 1)
        self.assertEqual(black.planes[1, 6, 6, 1], 1)
        self.assertEqual(black.planes[1, 108, 0, 0], 1)
        self.assertEqual(black.planes[1, 13, 5, 2], 1)
        self.assertEqual(black.planes[:, 109].sum(), 0)

    def test_history_and_frontier_budget(self):
        p = position()
        p["board"].extend([p["board"][0], p["board"][0]])
        encoded = encode_transfer_position(p, max_boards=2)
        self.assertTrue(encoded.context["frontierTruncated"])
        self.assertEqual(encoded.context["frontierBoardsOmitted"], 1)
        self.assertEqual(encoded.context["totalBoards"], 3)
        p["board"][0] = [[[0] * 9]]
        with self.assertRaisesRegex(ValueError, "8x8"):
            encode_transfer_position(p)

    def test_padding_policy_temporal_endpoints_and_frozen_gradients(self):
        model = small_model().eval()
        p = position()
        q = position()
        q["board"].append(q["board"][0])
        one, _ = predict(model, [p], torch.device("cpu"))
        two, _ = predict(model, [p, q], torch.device("cpu"))
        self.assertEqual(one[0], two[0])
        moves = [[[0, 0, 1, 1], [0, 0, 2, 1]], [[0, 0, 1, 1], [1, 0, 2, 1]], None]
        features = torch.tensor([encode_transfer_moves(q, moves)])
        context = model.encode(*model.collate([model.encode_position(q)], "cpu"))
        logits = model.score_moves(context, features, torch.tensor([[False, False, True]]))
        self.assertTrue(torch.isneginf(logits[0, 2]))
        self.assertNotEqual(logits[0, 0].item(), logits[0, 1].item())
        torch.nn.functional.cross_entropy(logits, torch.tensor([1])).backward()
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.endpoint_projection.parameters()))

    def test_train_resume_roundtrip_and_holdout_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = small_model()
            source, output, data = root / "source.pt", root / "out.pt", root / "data.jsonl"
            # Fixture is marked trained only to exercise the resume protocol.
            payload = {"architecture": ARCHITECTURE, "encodingVersion": 1, "config": asdict(model.config),
                       "backboneConfig": model.backbone.config, "state_dict": model.state_dict(),
                       "trainedSteps": 1, "policyTrainedSteps": 0, "baseline": model.baseline_metadata}
            torch.save(payload, source)
            p = position()
            moves = [[[0, 0, 1, 1], [0, 0, 2, 1]], None]
            data.write_text(json.dumps({"position": p, "value": 300, "group": "training",
                           "lc0PolicyVersion": 1, "lc0Policy": [{"moves": moves, "target": 0}]}) + "\n")
            args = argparse.Namespace(resume=str(source), data=str(data), output=str(output),
                                      device="cpu", steps=2, batch_size=1, learning_rate=.001, log_every=1)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(train(args), 0)
            loaded, checkpoint = load_checkpoint(output, torch.device("cpu"))
            self.assertEqual(checkpoint["trainedSteps"], 3)
            self.assertEqual(checkpoint["policyTrainedSteps"], 2)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(model.backbone.parameters(), loaded.backbone.parameters())))
            self.assertFalse(torch.equal(model.head[1].weight, loaded.head[1].weight))
            self.assertEqual(len(predict_policy(loaded, p, moves, torch.device("cpu"))), 2)
            info = audit_data(data)
            with self.assertRaisesRegex(ValueError, "overlap"):
                check_holdout(info, info)
            changed = position()
            changed["board"][0][0][1][1] = 0
            prefix_data = root / "prefix.jsonl"
            prefix_data.write_text(json.dumps({"position": changed, "value": 0, "group": "another",
                                   "lc0PolicyVersion": 1, "lc0Policy": [{"position": p, "moves": moves, "target": 0}]}) + "\n")
            with self.assertRaisesRegex(ValueError, "positions overlap"):
                check_holdout(info, audit_data(prefix_data))
            # Temporary checkpoint files must not alias data even with a safe destination.
            args.output = str(root / "data")
            protected = root / "data.tmp"
            protected.write_text(data.read_text())
            args.data = str(protected)
            with self.assertRaisesRegex(ValueError, "overwrite"):
                train(args)
            self.assertEqual(protected.read_text(), data.read_text())
            payload["trainedSteps"] = 0
            torch.save(payload, source)
            with self.assertRaisesRegex(ValueError, "completed training"):
                load_checkpoint(source, torch.device("cpu"))

    def test_refreezing_keeps_original_anchor_and_restore_checks_shapes(self):
        from .lc0_model import load_transfer_checkpoint
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = small_model()
            model.set_trainable_backbone(1)
            name, parameter = next((n, p) for n, p in model.backbone.named_parameters() if p.requires_grad)
            original_anchor = parameter.detach().clone() - .125
            source, output, data = root / "source.pt", root / "frozen.pt", root / "data.jsonl"
            payload = {"architecture": ARCHITECTURE, "encodingVersion": 1, "config": asdict(model.config),
                       "backboneConfig": model.backbone.config, "state_dict": model.state_dict(),
                       "trainedSteps": 1, "policyTrainedSteps": 0, "baseline": model.baseline_metadata,
                       "backboneAnchors": {name: original_anchor}}
            torch.save(payload, source)
            data.write_text(json.dumps({"position": position(), "value": 300}) + "\n")
            args = argparse.Namespace(resume=str(source), data=str(data), output=str(output),
                                      device="cpu", steps=1, batch_size=1, learning_rate=.001, unfreeze_last=0)
            with redirect_stdout(io.StringIO()):
                train(args)
            frozen = torch.load(output, weights_only=True)
            self.assertTrue(torch.equal(frozen["backboneAnchors"][name], original_anchor))
            self.assertTrue(torch.equal(frozen["state_dict"]["backbone." + name], parameter))
            # This configuration formerly tried a large allocation before rejecting keys.
            payload["backboneConfig"] = {**payload["backboneConfig"], "embedding_dim": 8192}
            with self.assertRaisesRegex(ValueError, "shape/dtype mismatch"):
                load_transfer_checkpoint(payload, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
