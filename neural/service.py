"""Persistent JSONL inference worker. Stdout is reserved for protocol messages."""

import argparse
import json
import sys

MAX_LINE_BYTES = 32 * 1024 * 1024


def emit(message):
    print(json.dumps(message, allow_nan=False, separators=(",", ":")), flush=True)


def inference_batch_size(parameters, requested=None):
    if requested is not None:
        if type(requested) is not int or not 1 <= requested <= 128:
            raise ValueError("batch-size must be 1–128")
        return requested
    # Long attention contexts need more memory with the larger architecture.
    return 1 if parameters >= 10_000_000 else 16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="artifacts/transformer/model.pt")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int,
                        help="inference batch, 1–128; default 1 for models with at least 10M parameters, otherwise 16")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    try:
        import torch
        try:
            from .model import choose_device, load_checkpoint, metadata, predict, predict_policy
        except ImportError:
            from model import choose_device, load_checkpoint, metadata, predict, predict_policy
        inference_batch_size(0, args.batch_size)  # Validate explicit limits before loading weights.
        if not 1 <= args.threads <= 32:
            raise ValueError("threads must be 1–32")
        torch.set_num_threads(args.threads)
        device = choose_device(args.device)
        model, checkpoint = load_checkpoint(args.checkpoint, device)
        model_info = metadata(model, checkpoint, args.checkpoint)
        batch_size = inference_batch_size(model_info["parameters"], args.batch_size)
        emit({"ready": True, "device": str(device), "batchSize": batch_size, "model": model_info})
    except ImportError as error:
        emit({"ready": False, "error": f"PyTorch is not installed in this Python environment: {error}. See docs/transformer.md."})
        return 1
    except Exception as error:
        emit({"ready": False, "error": str(error)})
        return 1
    while True:
        line = sys.stdin.buffer.readline(MAX_LINE_BYTES + 1)
        if not line:
            return 0
        request_id = None
        try:
            if len(line) > MAX_LINE_BYTES:
                emit({"id": None, "error": "request exceeds the 32 MiB protocol limit"})
                return 1
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be an object")
            request_id = request.get("id")
            if request.get("type") == "policy":
                scores = predict_policy(model, request.get("position"), request.get("moves"), device)
                emit({"id": request_id, "scores": scores, "device": str(device)})
                continue
            if request.get("type") not in (None, "evaluate"):
                raise ValueError("unknown inference request type")
            positions = request.get("positions")
            if not isinstance(positions, list) or not 1 <= len(positions) <= 128:
                raise ValueError("positions must contain between 1 and 128 positions")
            values, contexts = predict(model, positions, device, batch_size)
            emit({"id": request_id, "values": values, "context": contexts, "device": str(device)})
        except Exception as error:
            emit({"id": request_id, "error": str(error)})


if __name__ == "__main__":
    raise SystemExit(main())
