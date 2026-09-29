"""Compare trained checkpoints against a streamed held-out JSONL dataset.

Reports MSE in tanh(value / 1000) space and MAE after mapping predictions and
targets back to centipawns with both clipped to the model's ±~3800 cp range.
These are teacher-agreement metrics, not measured playing strength.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def evaluate_records(model, records, device, batch_size=16, max_batches=None):
    import torch
    try:
        from .model import collate_for_model
    except ImportError:
        from model import collate_for_model
    if not 1 <= batch_size <= 128 or (max_batches is not None and max_batches < 1):
        raise ValueError("batch-size must be 1–128 and max-batches must be positive")
    was_training = model.training
    model.eval()
    squared_error, absolute_cp_error, samples, batches = 0.0, 0.0, 0, 0
    truncated, frontier_truncated, batch = 0, 0, []

    def measure(items):
        nonlocal squared_error, absolute_cp_error, samples, truncated, frontier_truncated
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            prediction = model(*collate_for_model(model, [record[0] for record in items], device)).float()
            target = torch.tensor([record[1] for record in items], dtype=torch.float32, device=device)
            if not torch.isfinite(prediction).all():
                raise ValueError("model returned nonfinite validation predictions")
            squared_error += float((prediction - target).square().sum().item())
            predicted_cp = 1000 * torch.atanh(prediction.clamp(-0.999, 0.999))
            target_cp = 1000 * torch.atanh(target.clamp(-0.999, 0.999))
            absolute_cp_error += float((predicted_cp - target_cp).abs().sum().item())
        samples += len(items)
        truncated += sum(record[0].context["truncated"] for record in items)
        frontier_truncated += sum(record[0].context["frontierTruncated"] for record in items)

    try:
        for record in records:
            batch.append(record)
            if len(batch) == batch_size:
                measure(batch)
                batch = []
                batches += 1
                if max_batches is not None and batches >= max_batches:
                    break
        if batch:
            measure(batch)
        if not samples:
            raise ValueError("evaluation file has no examples")
        return {"normalizedMse": squared_error / samples,
                "maeClippedCp": absolute_cp_error / samples,
                "samples": samples, "contextTruncated": truncated,
                "frontierTruncated": frontier_truncated}
    finally:
        model.train(was_training)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", help="also write the complete JSON report to this file")
    args = parser.parse_args()
    try:
        import torch
        try:
            from .model import choose_device, encode_for_model, load_checkpoint, metadata
            from .train import records
        except ImportError:
            from model import choose_device, encode_for_model, load_checkpoint, metadata
            from train import records
        if not 1 <= args.threads <= 32 or not 1 <= args.batch_size <= 128:
            raise ValueError("threads must be 1–32 and batch-size must be 1–128")
        torch.set_num_threads(args.threads)
        device = choose_device(args.device)
        digest = hashlib.sha256()
        with open(args.data, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        report = {"data": str(Path(args.data).resolve()), "dataSha256": digest.hexdigest(),
                  "device": str(device), "batchSize": args.batch_size,
                  "target": "White-relative teacher centipawns, tanh(value / 1000)",
                  "cpClip": 3800.201167, "checkpoints": []}
        for path in args.checkpoints:
            model, checkpoint = load_checkpoint(path, device)
            start = time.perf_counter()
            rows = records(args.data, model.config.max_tokens, encoder=lambda position: encode_for_model(model, position))
            metrics = evaluate_records(model, rows, device, args.batch_size)
            report["checkpoints"].append({"model": metadata(model, checkpoint, path), **metrics,
                                          "seconds": round(time.perf_counter() - start, 3)})
            del model, checkpoint
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
