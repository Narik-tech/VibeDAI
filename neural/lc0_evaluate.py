"""Evaluate LC0-transfer agreement with 5D teacher values and component policies.

Every supplied policy prefix with more than one candidate contributes once.
These supervised metrics do not measure playing strength or prove full-rules
legality of the alternatives supplied by the dataset.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch

try:
    from .lc0_encoding import encode_transfer_moves
    from .lc0_model import ARCHITECTURE
    from .lc0_train import read_rows
    from .model import choose_device, load_checkpoint
    from .policy import collate_moves
except ImportError:
    from lc0_encoding import encode_transfer_moves
    from lc0_model import ARCHITECTURE
    from lc0_train import read_rows
    from model import choose_device, load_checkpoint
    from policy import collate_moves


def file_sha256(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def dataset_audit(path, max_rows=1000000):
    """Bound audit metadata; inference retains only a small encoded batch."""
    groups, games, positions, stages = Counter(), set(), set(), Counter()
    root_positions = set()
    count = missing_groups = prefix_count = 0

    def position_digest(state):
        text = json.dumps([state["action"] % 2, state.get("promotions"), state["board"]], separators=(",", ":"))
        return hashlib.sha256(text.encode()).hexdigest()

    for row in read_rows(path):
        count += 1
        if count > max_rows:
            raise ValueError(f"dataset exceeds the {max_rows} row audit limit")
        group = row.get("group", row.get("gameId"))
        if group is None:
            missing_groups += 1
        elif not isinstance(group, str) or not group:
            raise ValueError("group/gameId must be a nonempty string")
        else:
            groups[group] += 1
        if row.get("gameId") is not None:
            if not isinstance(row["gameId"], str) or not row["gameId"]:
                raise ValueError("gameId must be a nonempty string")
            games.add(row["gameId"])
        root_hash = position_digest(row["position"])
        root_positions.add(root_hash)
        positions.add(root_hash)
        labels = row.get("lc0Policy", row.get("policy"))
        if labels is not None:
            if not isinstance(labels, list) or not 1 <= len(labels) <= 256:
                raise ValueError("policy must contain 1–256 prefixes")
            for label in labels:
                if not isinstance(label, dict):
                    raise ValueError("policy prefix must be an object")
                prefix_count += 1
                positions.add(position_digest(label.get("position", row["position"])))
                if len(positions) > 1000000:
                    raise ValueError("dataset exceeds the 1000000 unique-position audit limit")
        stages[str(row.get("stage", "replay"))] += 1
    if not count:
        raise ValueError("evaluation dataset has no examples")
    summary = {"path": str(Path(path).resolve()), "sha256": file_sha256(path), "rows": count,
               "groups": dict(sorted(groups.items())), "groupCount": len(groups), "gameCount": len(games),
               "rowsWithoutGroups": missing_groups, "uniquePositions": len(positions),
               "uniqueRootPositions": len(root_positions), "policyPrefixPositions": prefix_count,
               "duplicateRootPositionRows": count - len(root_positions), "stages": dict(sorted(stages.items())),
               "positionIdentity": "side to move, promotion rules, complete board history"}
    return {"summary": summary, "groups": set(groups), "games": games, "positions": positions}


def holdout_audit(evaluation, training, expected_training_hash=None):
    groups = sorted(evaluation["groups"] & training["groups"])
    games = sorted(evaluation["games"] & training["games"])
    position_count = len(evaluation["positions"] & training["positions"])
    hash_match = None if not expected_training_hash else training["summary"]["sha256"] == expected_training_hash
    complete_groups = not evaluation["summary"]["rowsWithoutGroups"] and not training["summary"]["rowsWithoutGroups"]
    return {"training": training["summary"], "matchesCheckpointTrainingHash": hash_match,
            "overlappingGroups": groups, "overlappingGameIds": games, "overlappingPositions": position_count,
            "groupsComplete": complete_groups,
            "disjoint": bool(complete_groups and not groups and not games and not position_count),
            "verifiedAgainstCheckpointTraining": bool(hash_match and complete_groups and not groups and not games and not position_count)}


def _bucket():
    return {"values": 0, "squaredError": 0., "absoluteCpError": 0., "contextTruncated": 0,
            "frontierTruncated": 0, "prefixes": 0, "decisionPrefixes": 0, "forcedPrefixes": 0,
            "submitTargets": 0, "temporalTargets": 0, "decisionSubmitTargets": 0, "decisionTemporalTargets": 0,
            "correct": 0, "crossEntropySum": 0., "uniformAccuracySum": 0., "uniformCrossEntropySum": 0.,
            "candidateSum": 0, "policyContextTruncated": 0, "policyFrontierTruncated": 0}


def _finish(bucket):
    n, p = bucket["values"], bucket["decisionPrefixes"]
    return {"value": {"samples": n, "normalizedMse": bucket["squaredError"] / n if n else None,
                      "maeClippedCp": bucket["absoluteCpError"] / n if n else None,
                      "contextTruncated": bucket["contextTruncated"], "frontierTruncated": bucket["frontierTruncated"]},
            "policy": {**{key: bucket[key] for key in ("prefixes", "decisionPrefixes", "forcedPrefixes", "submitTargets",
                           "temporalTargets", "decisionSubmitTargets", "decisionTemporalTargets", "correct",
                           "policyContextTruncated", "policyFrontierTruncated")},
                       "top1Accuracy": bucket["correct"] / p if p else None,
                       "crossEntropy": bucket["crossEntropySum"] / p if p else None,
                       "uniformTop1Accuracy": bucket["uniformAccuracySum"] / p if p else None,
                       "uniformCrossEntropy": bucket["uniformCrossEntropySum"] / p if p else None,
                       "meanCandidates": bucket["candidateSum"] / p if p else None}}


def evaluate_rows(model, rows, device, batch_size=4, candidate_budget=16384):
    """Stream all values and all nontrivial prefixes, with bounded padding."""
    if type(batch_size) is not int or not 1 <= batch_size <= 128:
        raise ValueError("batch-size must be 1–128")
    if type(candidate_budget) is not int or not 16384 <= candidate_budget <= 1048576:
        raise ValueError("candidate-budget must be 16384–1048576")
    total, stages = _bucket(), {}
    value_batch, policy_batch = [], []
    was_training = model.training
    model.eval()

    def buckets(stage):
        if stage not in stages:
            stages[stage] = _bucket()
        return total, stages[stage]

    def values():
        if not value_batch:
            return
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            prediction = model(*model.collate([item[0] for item in value_batch], device)).float()
            if prediction.shape != (len(value_batch),) or not torch.isfinite(prediction).all():
                raise ValueError("model returned invalid value predictions")
            target = torch.tensor([item[1] for item in value_batch], device=device)
            square = (prediction - target).square().cpu().tolist()
            cp_error = (1000 * (torch.atanh(prediction.clamp(-.999, .999)) - torch.atanh(target.clamp(-.999, .999)))).abs().cpu().tolist()
        for (encoded, _, stage), mse, mae in zip(value_batch, square, cp_error):
            for bucket in buckets(stage):
                bucket["values"] += 1
                bucket["squaredError"] += mse
                bucket["absoluteCpError"] += mae
                bucket["contextTruncated"] += bool(encoded.context["truncated"])
                bucket["frontierTruncated"] += bool(encoded.context["frontierTruncated"])
        value_batch.clear()

    def policies():
        if not policy_batch:
            return
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            context = model.encode(*model.collate([item[0] for item in policy_batch], device))
            features, mask = collate_moves([item[1] for item in policy_batch], device)
            logits = model.score_moves(context, features, mask).float()
            if logits.shape != mask.shape or not torch.isfinite(logits[~mask]).all():
                raise ValueError("model returned invalid policy logits")
            # Enforce the supplied mask even if a model hook omits it.
            logits = logits.masked_fill(mask, -torch.inf)
            targets = torch.tensor([item[2] for item in policy_batch], device=device)
            losses = torch.nn.functional.cross_entropy(logits, targets, reduction="none").cpu().tolist()
            correct = (logits.argmax(-1) == targets).cpu().tolist()
        for (encoded, features, _, stage), loss, right in zip(policy_batch, losses, correct):
            for bucket in buckets(stage):
                bucket["decisionPrefixes"] += 1
                bucket["correct"] += right
                bucket["crossEntropySum"] += loss
                bucket["uniformAccuracySum"] += 1 / len(features)
                bucket["uniformCrossEntropySum"] += math.log(len(features))
                bucket["candidateSum"] += len(features)
                bucket["policyContextTruncated"] += bool(encoded.context["truncated"])
                bucket["policyFrontierTruncated"] += bool(encoded.context["frontierTruncated"])
        policy_batch.clear()

    try:
        for row in rows:
            stage = str(row.get("stage", "replay"))
            value_batch.append((model.encode_position(row["position"]), math.tanh(row["value"] / 1000), stage))
            if len(value_batch) >= batch_size:
                values()
            labels = row.get("lc0Policy", row.get("policy"))
            if labels is None:
                continue
            if not isinstance(labels, list) or not 1 <= len(labels) <= 256:
                raise ValueError("policy must contain 1–256 prefixes")
            for label in labels:
                if not isinstance(label, dict):
                    raise ValueError("policy prefix must be an object")
                prefix = label.get("position", row["position"])
                moves = label.get("moves")
                features = encode_transfer_moves(prefix, moves)
                target = label.get("target")
                if type(target) is not int or not 0 <= target < len(features):
                    raise ValueError("policy target must index supplied candidates")
                selected = moves[target]
                submit = selected is None
                temporal = not submit and selected[0][:2] != selected[1][:2]
                encoded = model.encode_position(prefix)
                for bucket in buckets(stage):
                    bucket["prefixes"] += 1
                    bucket["forcedPrefixes"] += len(features) == 1
                    bucket["submitTargets"] += submit
                    bucket["temporalTargets"] += temporal
                    bucket["decisionSubmitTargets"] += submit and len(features) > 1
                    bucket["decisionTemporalTargets"] += temporal and len(features) > 1
                if len(features) == 1:
                    continue
                width = max([len(features)] + [len(item[1]) for item in policy_batch])
                if policy_batch and (len(policy_batch) >= batch_size or width * (len(policy_batch) + 1) > candidate_budget):
                    policies()
                policy_batch.append((encoded, features, target, stage))
        values()
        policies()
        if not total["values"]:
            raise ValueError("evaluation dataset has no examples")
        return {**_finish(total), "stages": {stage: _finish(bucket) for stage, bucket in sorted(stages.items())}}
    finally:
        model.train(was_training)


def evaluate_checkpoint(checkpoint_path, data_path, device, batch_size=4, candidate_budget=16384, training_data=None):
    started = time.perf_counter()
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("checkpoint must use the LC0 transfer architecture")
    info = dataset_audit(data_path)
    recorded = checkpoint.get("training", {})
    training_path = training_data or recorded.get("data")
    if training_path and Path(training_path).is_file():
        holdout = holdout_audit(info, dataset_audit(training_path), recorded.get("dataSha256"))
    elif training_data:
        raise ValueError(f"training audit file does not exist: {training_data}")
    else:
        holdout = {"verifiedAgainstCheckpointTraining": False, "reason": "Recorded training data unavailable; group separation not verified."}
    metrics = evaluate_rows(model, read_rows(data_path), device, batch_size, candidate_budget)
    return {"version": 1, "meaning": "Agreement with supplied White-relative 5D teacher evaluations and masked component choices; playing strength unmeasured.",
            "valueTransform": "tanh(White centipawns / 1000)", "centipawnClip": 1000 * math.atanh(.999),
            "policyDenominator": "Every supplied prefix with more than one candidate, unweighted; singleton choices excluded from accuracy and cross-entropy.",
            "model": {"path": str(Path(checkpoint_path).resolve()), "sha256": file_sha256(checkpoint_path),
                      "architecture": checkpoint["architecture"], "trainedSteps": checkpoint.get("trainedSteps"),
                      "policyTrainedSteps": checkpoint.get("policyTrainedSteps"), "config": checkpoint.get("config"),
                      "baseline": checkpoint.get("baseline"), "selection": checkpoint.get("selection")},
            "data": info["summary"], "holdoutAudit": holdout,
            "matchesRecordedValidationHash": info["summary"]["sha256"] == recorded["validationSha256"] if recorded.get("validationSha256") else None,
            "device": str(device), "batchSize": batch_size, "paddedCandidateBudget": candidate_budget,
            **metrics, "seconds": round(time.perf_counter() - started, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--training-data", help="override the recorded training path for group/hash audit")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--candidate-budget", type=int, default=16384)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", help="write the complete JSON report")
    args = parser.parse_args()
    try:
        if not 1 <= args.threads <= 32:
            raise ValueError("threads must be 1–32")
        if args.output and Path(args.output).resolve() in {Path(path).resolve() for path in (args.checkpoint, args.data, args.training_data) if path}:
            raise ValueError("evaluation output cannot overwrite a checkpoint or dataset")
        torch.set_num_threads(args.threads)
        report = evaluate_checkpoint(args.checkpoint, args.data, choose_device(args.device), args.batch_size,
                                     args.candidate_budget, args.training_data)
        if args.output and report["holdoutAudit"].get("training", {}).get("path") == str(Path(args.output).resolve()):
            raise ValueError("evaluation output cannot overwrite recorded training data")
        output = json.dumps(report, indent=2, allow_nan=False)
        if args.output:
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(output + "\n", encoding="utf-8")
        print(output, flush=True)
        return 0
    except Exception as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
