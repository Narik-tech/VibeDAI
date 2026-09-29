"""Train LCZero spatial transfer + 5D adapters, or resume it in self-play."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import torch

try:
    from .lc0_model import ARCHITECTURE, TransferConfig, create_transfer_model
    from .lc0_encoding import TRANSFER_ENCODING_VERSION, encode_transfer_moves
    from .model import choose_device, load_checkpoint
    from .policy import collate_moves, POLICY_VERSION
    from .train import MAX_LINE_BYTES, write_checkpoint, weighted_mse
    from .evaluate import evaluate_records
except ImportError:
    from lc0_model import ARCHITECTURE, TransferConfig, create_transfer_model
    from lc0_encoding import TRANSFER_ENCODING_VERSION, encode_transfer_moves
    from model import choose_device, load_checkpoint
    from policy import collate_moves, POLICY_VERSION
    from train import MAX_LINE_BYTES, write_checkpoint, weighted_mse
    from evaluate import evaluate_records


def read_rows(path):
    with open(path, "rb") as stream:
        index = 0
        while True:
            line = stream.readline(MAX_LINE_BYTES + 1)
            if not line:
                return
            index += 1
            if len(line) > MAX_LINE_BYTES:
                raise ValueError(f"{path}:{index}: row exceeds 32 MiB")
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or "position" not in row:
                raise ValueError(f"{path}:{index}: missing position")
            value, weight = row.get("value"), row.get("weight", 1.)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{path}:{index}: value must be finite White-relative cp")
            if type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0:
                raise ValueError(f"{path}:{index}: weight must be positive and finite")
            if row.get("lc0Policy") is not None and row.get("lc0PolicyVersion") != 1:
                raise ValueError("unsupported lc0PolicyVersion")
            if row.get("policy") is not None and row.get("policyVersion", 1) != POLICY_VERSION:
                raise ValueError("unsupported policyVersion")
            yield row


def audit_data(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    groups, positions, stages = set(), set(), {}
    mean, count, policies = 0., 0, 0
    for row in read_rows(path):
        count += 1
        mean += (row.get("weight", 1.) - mean) / count
        group = row.get("group", row.get("gameId"))
        if group is not None:
            if not isinstance(group, str):
                raise ValueError("group/gameId must be a string")
            groups.add(group)
        supervised_positions = [row["position"]]
        labels = row.get("lc0Policy", row.get("policy"))
        if labels is not None:
            if not isinstance(labels, list) or not 1 <= len(labels) <= 256:
                raise ValueError("policy must contain 1–256 prefixes")
            for label in labels:
                if not isinstance(label, dict):
                    raise ValueError("policy prefix must be an object")
                supervised_positions.append(label.get("position", row["position"]))
        for position in supervised_positions:
            identity = [position["action"] % 2, position.get("promotions"), position["board"]]
            positions.add(hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest())
        stage = str(row.get("stage", "replay"))
        stages[stage] = stages.get(stage, 0) + 1
        policies += bool(row.get("lc0Policy", row.get("policy")))
    if count == 0:
        raise ValueError("dataset has no examples")
    return {"sha256": digest.hexdigest(), "rows": count, "weightMean": mean,
            "policyRows": policies, "stages": stages, "groups": groups, "positions": positions}


def check_holdout(training, validation):
    if training["groups"] & validation["groups"]:
        raise ValueError("training/validation source groups overlap")
    if training["positions"] & validation["positions"]:
        raise ValueError("training/validation positions overlap")


def encoded_rows(path, model, rng=None, policy=True):
    for row in read_rows(path):
        position = model.encode_position(row["position"])
        example = None
        labels = row.get("lc0Policy", row.get("policy")) if policy else None
        if labels is not None:
            if not isinstance(labels, list) or not 1 <= len(labels) <= 256:
                raise ValueError("policy must contain 1–256 prefixes")
            # Validate every prefix before sampling, so malformed unused labels fail.
            validated = []
            for label in labels:
                if not isinstance(label, dict):
                    raise ValueError("policy prefix must be an object")
                prefix = label.get("position", row["position"])
                features = encode_transfer_moves(prefix, label.get("moves"))
                target = label.get("target")
                if type(target) is not int or not 0 <= target < len(features):
                    raise ValueError("policy target must index supplied candidates")
                validated.append((prefix, features, target))
            prefix, features, target = (rng.choice(validated) if rng else validated[0])
            example = (model.encode_position(prefix), features, target)
        yield position, math.tanh(row["value"] / 1000), row.get("weight", 1.), example


def training_stream(path, model, rng, buffer_size, policy):
    while True:
        buffer = []
        for row in encoded_rows(path, model, rng, policy):
            if len(buffer) < buffer_size:
                buffer.append(row)
            else:
                index = rng.randrange(len(buffer))
                yield buffer[index]
                buffer[index] = row
        rng.shuffle(buffer)
        yield from buffer
        if not buffer:
            raise ValueError("training file is empty")


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}, allow_nan=False), flush=True)


def train(args):
    # Also accepts neural/train.py's existing namespace during managed self-play.
    defaults = dict(weights=None, resume=None, unfreeze_last=None, backbone_lr=1e-6,
                    anchor_weight=0.01, max_boards=None, board_batch=None, width=None,
                    heads=None, layers=None, feedforward=None, max_tokens=4096, dropout=0.1,
                    policy="auto", policy_weight=1., weight_decay=.01, shuffle_buffer=16,
                    seed=42, threads=2, save_every=100, log_every=10, validation_data=None,
                    validation_batches=16, best_output=None,
                    label="LC0 transfer to 5D; experimental, strength unmeasured")
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    if bool(args.weights) == bool(args.resume):
        raise ValueError("supply exactly one of --weights or --resume")
    for name, low, high in (("steps", 1, 1000000), ("batch_size", 1, 128), ("threads", 1, 32),
                            ("shuffle_buffer", 1, 4096), ("save_every", 1, 1000000),
                            ("log_every", 1, 1000000), ("validation_batches", 1, 1000000)):
        value = getattr(args, name)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{name} must be {low}–{high}")
    for name in ("learning_rate", "backbone_lr", "policy_weight", "weight_decay", "anchor_weight"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name.endswith("rate") or name in ("backbone_lr", "policy_weight")) and value == 0:
            raise ValueError(f"invalid {name}")
    if args.best_output and (not args.validation_data or Path(args.output).resolve() == Path(args.best_output).resolve()):
        raise ValueError("--best-output needs validation data and a path distinct from output")
    inputs = {Path(p).resolve() for p in (args.data, args.validation_data, args.weights) if p}
    destinations = [Path(p).resolve() for p in (args.output, args.best_output) if p]
    temporary = [p.with_name(p.name + ".tmp") for p in destinations]
    protected = inputs | ({Path(args.resume).resolve()} if args.resume else set())
    if any(p in inputs for p in destinations) or any(p in protected for p in temporary):
        raise ValueError("checkpoint output cannot overwrite weights or datasets")
    if len(set(destinations + temporary)) != len(destinations + temporary):
        raise ValueError("checkpoint outputs and temporary paths must be distinct")
    info = audit_data(args.data)
    validation_info = audit_data(args.validation_data) if args.validation_data else None
    if validation_info:
        check_holdout(info, validation_info)
    train_policy = args.policy != "off" and info["policyRows"] > 0
    if args.policy == "on" and not train_policy:
        raise ValueError("--policy on requires policy labels")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = choose_device(args.device)
    checkpoint = {}
    if args.resume:
        model, checkpoint = load_checkpoint(args.resume, device, max_tokens=args.max_tokens)
        if checkpoint["architecture"] != ARCHITECTURE:
            raise ValueError("resume checkpoint is not an LC0 transfer model")
        for name in ("width", "heads", "layers", "feedforward"):
            if getattr(args, name) is not None and getattr(args, name) != getattr(model.config, name):
                raise ValueError(f"cannot change {name} when resuming a transfer checkpoint")
        budgets = {key: getattr(args, key) for key in ("max_boards", "board_batch") if getattr(args, key) is not None}
        model.config = TransferConfig(**{**asdict(model.config), **budgets})
    else:
        overrides = {key: getattr(args, key) for key in ("width", "heads", "layers", "feedforward") if getattr(args, key) is not None}
        config = TransferConfig(**overrides, max_boards=16 if args.max_boards is None else args.max_boards,
                                max_tokens=args.max_tokens, board_batch=8 if args.board_batch is None else args.board_batch,
                                dropout=args.dropout)
        model = create_transfer_model(args.weights, config).to(device)
    previous_unfreeze = model.config.unfreeze_last
    if args.unfreeze_last is not None:
        if type(args.unfreeze_last) is not int or not 0 <= args.unfreeze_last <= 15:
            raise ValueError("unfreeze_last must be 0–15")
        model.set_trainable_backbone(args.unfreeze_last)
    base_params = [(name, p) for name, p in model.backbone.named_parameters() if p.requires_grad]
    adapters = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith("backbone.")]
    groups = [{"params": adapters, "lr": args.learning_rate}]
    if base_params:
        groups.append({"params": [p for _, p in base_params], "lr": args.backbone_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    if "optimizer" in checkpoint and previous_unfreeze == model.config.unfreeze_last:
        optimizer.load_state_dict(checkpoint["optimizer"])
        for i, group in enumerate(optimizer.param_groups):
            group.update(lr=args.backbone_lr if i else args.learning_rate, weight_decay=args.weight_decay)
    # Anchor previously unfrozen weights to their first adaptation-stage values.
    saved_anchors = dict(checkpoint.get("backboneAnchors", {}))
    all_base = dict(model.backbone.named_parameters())
    for name, anchor in saved_anchors.items():
        if name not in all_base or not isinstance(anchor, torch.Tensor) or anchor.shape != all_base[name].shape or not torch.isfinite(anchor).all():
            raise ValueError("invalid saved backbone anchor")
    for name, parameter in base_params:
        if name not in saved_anchors:
            saved_anchors[name] = parameter.detach().cpu().clone()
    anchors = {name: saved_anchors[name].clone().to(device) for name, _ in base_params}
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    stream = training_stream(args.data, model, rng, args.shuffle_buffer, train_policy)
    previous_steps, examples = checkpoint.get("trainedSteps", 0), checkpoint.get("examplesSeen", 0)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def validate():
        if not args.validation_data:
            return None
        return evaluate_records(model, encoded_rows(args.validation_data, model, policy=False), device,
                                args.batch_size, args.validation_batches)

    baseline_validation = validate()
    best = baseline_validation if previous_steps else None
    best_step = previous_steps if best else None
    emit("start", architecture=ARCHITECTURE, config=asdict(model.config), device=str(device),
         parameters=sum(p.numel() for p in model.parameters()), trainableParameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
         baseline={key: model.baseline_metadata.get(key) for key in (
             "source_path", "source_sha256", "encoder_layers", "heads", "reused_tensor_count",
             "reused_parameter_count", "omitted_parameter_count", "parity_status")},
         additionalSteps=args.steps, previousSteps=previous_steps,
         data={k: v for k, v in info.items() if k not in ("groups", "positions")}, baselineValidation=baseline_validation)
    if args.best_output and best:
        write_checkpoint(args.best_output, {**checkpoint,
                         "config": {**checkpoint["config"], **{key: getattr(model.config, key)
                                    for key in ("max_tokens", "max_boards", "board_batch")}}})
    updates, attempts = 0, 0
    last_validation = baseline_validation
    while updates < args.steps:
        attempts += 1
        if attempts > args.steps * 10:
            raise RuntimeError("too many skipped AMP updates")
        model.train()
        rows = [next(stream) for _ in range(args.batch_size)]
        targets = torch.tensor([r[1] for r in rows], device=device)
        weights = torch.tensor([r[2] / info["weightMean"] for r in rows], device=device)
        optimizer.zero_grad(set_to_none=True)
        examples_policy = [(i, row[3]) for i, row in enumerate(rows) if row[3] is not None]
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            prediction = model(*model.collate([r[0] for r in rows], device))
            value_loss = weighted_mse(prediction, targets, weights)
            component_loss = prediction.sum() * 0
            if examples_policy:
                context = model.encode(*model.collate([e[0] for _, e in examples_policy], device))
                features, mask = collate_moves([e[1] for _, e in examples_policy], device)
                logits = model.score_moves(context, features, mask)
                choices = torch.tensor([e[2] for _, e in examples_policy], device=device)
                per_prefix = torch.nn.functional.cross_entropy(logits.float(), choices, reduction="none")
                component_loss = (per_prefix * weights[[i for i, _ in examples_policy]]).sum() / len(rows)
            anchor_loss = prediction.sum() * 0
            if anchors:
                anchor_loss = sum((p.float() - anchors[name]).square().sum() for name, p in base_params) / sum(p.numel() for _, p in base_params)
            loss = value_loss + args.policy_weight * component_loss + args.anchor_weight * anchor_loss
        if not torch.isfinite(loss):
            raise ValueError("nonfinite transfer training loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < scale:
            continue
        updates += 1
        if any(len(e[1]) > 1 for _, e in examples_policy):
            model.policy_trained_steps += 1
        examples += len(rows)
        step = previous_steps + updates
        if updates == 1 or updates % args.log_every == 0 or updates == args.steps:
            emit("train", step=step, loss=loss.item(), valueLoss=value_loss.item(), policyLoss=component_loss.item(),
                 anchorLoss=anchor_loss.item(), policyTrainedSteps=model.policy_trained_steps,
                 seconds=round(time.perf_counter() - started, 3))
        if updates % args.save_every == 0 or updates == args.steps:
            last_validation = validate()
            improved = last_validation is not None and (best is None or last_validation["normalizedMse"] < best["normalizedMse"])
            if improved:
                best, best_step = last_validation, step
            payload = {"architecture": ARCHITECTURE, "encodingVersion": TRANSFER_ENCODING_VERSION,
                       "config": asdict(model.config), "backboneConfig": model.backbone.config,
                       "baseline": model.baseline_metadata, "state_dict": model.state_dict(),
                       "trainedSteps": step, "policyVersion": POLICY_VERSION,
                       "policyTrainedSteps": model.policy_trained_steps, "examplesSeen": examples,
                       "optimizer": optimizer.state_dict(), "backboneAnchors": saved_anchors,
                       "label": args.label, "createdAt": datetime.now(timezone.utc).isoformat(),
                       "training": {"data": str(Path(args.data).resolve()), "dataSha256": info["sha256"],
                                    "validationSha256": validation_info["sha256"] if validation_info else None,
                                    "loss": loss.item(), "validation": last_validation, "seed": args.seed,
                                    "learningRate": args.learning_rate, "backboneLearningRate": args.backbone_lr,
                                    "anchorWeight": args.anchor_weight, "policyWeight": args.policy_weight,
                                    "torchVersion": str(torch.__version__), "strengthEstablished": False},
                       "selection": {"baselineValidation": baseline_validation, "bestValidation": best, "bestStep": best_step}}
            write_checkpoint(args.output, payload)
            if args.best_output and improved:
                write_checkpoint(args.best_output, payload)
            if last_validation:
                emit("validation", step=step, validation=last_validation, bestStep=best_step)
    result = dict(output=str(Path(args.output).resolve()), trainedSteps=previous_steps + updates,
                  policyTrainedSteps=model.policy_trained_steps, loss=loss.item(), validation=last_validation,
                  bestStep=best_step, seconds=round(time.perf_counter() - started, 3))
    if device.type == "cuda":
        result["peakAllocatedMiB"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    emit("complete", **result)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--weights", help="LCZero .pb.gz source; imports the spatial body")
    source.add_argument("--resume", help="self-contained trained transfer checkpoint")
    parser.add_argument("--data", default="artifacts/lc0/curriculum.jsonl")
    parser.add_argument("--output", default="artifacts/lc0/model.pt")
    parser.add_argument("--validation-data")
    parser.add_argument("--best-output")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    for name, default in (("steps", 1000), ("batch-size", 4), ("max-boards", None), ("board-batch", None),
                          ("max-tokens", 4096), ("shuffle-buffer", 16), ("seed", 42), ("threads", 2),
                          ("save-every", 100), ("log-every", 10), ("validation-batches", 16)):
        parser.add_argument("--" + name, type=int, default=default)
    for name in ("width", "heads", "layers", "feedforward", "unfreeze-last"):
        parser.add_argument("--" + name, type=int)
    for name, default in (("learning-rate", .0003), ("backbone-lr", .000001), ("anchor-weight", .01),
                          ("weight-decay", .01), ("policy-weight", 1.), ("dropout", .1)):
        parser.add_argument("--" + name, type=float, default=default)
    parser.add_argument("--policy", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--label", default="LC0 transfer to 5D; experimental, strength unmeasured")
    args = parser.parse_args()
    try:
        return train(args)
    except Exception as error:
        print(json.dumps({"event": "error", "error": str(error)}), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
