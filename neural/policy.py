"""Versioned component descriptors; legal candidate lists are supplied by rules.

The network never generates coordinates or authorizes a move. Logits are masked
to each supplied list. Four endpoints preserve castling and en-passant details,
and signed timeline/time coordinates distinguish spatially identical jumps.
"""

try:
    from .encoding import _integer, _timeline_coordinate, _validate
except ImportError:
    from encoding import _integer, _timeline_coordinate, _validate

POLICY_VERSION = 1
MOVE_FEATURES = 38
MAX_POLICY_MOVES = 16384


def encode_moves(position, moves):
    board, action, _, _, _, _ = _validate(position)
    if not isinstance(moves, list) or not 1 <= len(moves) <= MAX_POLICY_MOVES:
        raise ValueError(f"policy moves must contain 1–{MAX_POLICY_MOVES} candidates")
    even = board[0] is None
    result = []
    for move in moves:
        if not isinstance(move, list) or not 2 <= len(move) <= 4:
            raise ValueError("component move must contain 2–4 endpoints")
        features, endpoints = [], []
        for square in move:
            if not isinstance(square, list) or not 4 <= len(square) <= 5 or any(not _integer(x) for x in square):
                raise ValueError("move endpoints must contain 4 coordinates and an optional promotion")
            line, time, rank, file = square[:4]
            if not (0 <= line < len(board) and board[line] is not None and 0 <= time < len(board[line])
                    and board[line][time] is not None and 0 <= rank < len(board[line][time])
                    and 0 <= file < len(board[line][time][rank])):
                raise ValueError("move endpoint is outside position history")
            promotion = square[4] if len(square) == 5 else 0
            if not 0 <= promotion <= 24:
                raise ValueError("move promotion must be a piece code")
            piece = board[line][time][rank][file]
            coordinates = [_timeline_coordinate(line, even), time, rank, file]
            endpoints.append(coordinates)
            features.extend([*coordinates, abs(piece) / 24, float(piece < 0), promotion / 24, 1.0])
        features.extend([0.0] * (8 * (4 - len(move))))
        features.extend(endpoints[1][i] - endpoints[0][i] for i in range(4))
        features.extend([float(move[0][:2] != move[1][:2]), float(action % 2)])
        result.append(features)
    return result


def encode_policy_records(position, policy):
    """Validate optional prefix labels while retaining old value-only rows."""
    if policy is None:
        return []
    if not isinstance(policy, list) or not 1 <= len(policy) <= 256:
        raise ValueError("policy must contain 1–256 component prefix examples")
    result = []
    for item in policy:
        if not isinstance(item, dict):
            raise ValueError("policy example must be an object")
        prefix = item.get("position", position)
        features = encode_moves(prefix, item.get("moves"))
        target = item.get("target")
        if not _integer(target) or not 0 <= target < len(features):
            raise ValueError("policy target must index a supplied legal candidate")
        result.append((prefix, features, target))
    return result


def collate_moves(features, device):
    import torch
    count = max(len(moves) for moves in features)
    inputs = torch.zeros((len(features), count, MOVE_FEATURES), device=device)
    mask = torch.ones((len(features), count), dtype=torch.bool, device=device)
    for index, moves in enumerate(features):
        inputs[index, :len(moves)] = torch.tensor(moves, dtype=torch.float32, device=device)
        mask[index, :len(moves)] = False
    return inputs, mask


def policy_loss(model, examples, device, weights=None):
    import torch
    try:
        from .model import collate_for_model
    except ImportError:
        from model import collate_for_model
    if not examples:
        return next(model.parameters()).sum() * 0
    batch = collate_for_model(model, [item[0] for item in examples], device)
    features, mask = collate_moves([item[1] for item in examples], device)
    logits = model.score_moves(model.encode(*batch), features, mask)
    targets = torch.tensor([item[2] for item in examples], dtype=torch.long, device=device)
    losses = torch.nn.functional.cross_entropy(logits.float(), targets, reduction="none")
    if weights is not None:
        losses = losses * torch.tensor(weights, dtype=torch.float32, device=device)
    return losses.mean()
