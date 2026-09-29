"""Versioned 5D board projection for a transferred LCZero spatial encoder.

This is deliberately a 5D input adapter, not LC0's orthodox position encoder.
Chess repetition, rule-50, and en-passant planes are zero: none can be inferred
as orthodox rights from a multiverse snapshot. Full history still governs rules.
"""

from dataclasses import dataclass
import math

try:
    from .encoding import _validate, _timeline_coordinate, MAX_TOKENS
    from .policy import encode_moves, MOVE_FEATURES
except ImportError:
    from encoding import _validate, _timeline_coordinate, MAX_TOKENS
    from policy import encode_moves, MOVE_FEATURES

TRANSFER_ENCODING_VERSION = 1
BOARD_FEATURES = 10
# 5d-chess-js: pawn, bishop, knight, rook, queen, king; even=White.
PIECE_PLANES = {0: 0, 1: 2, 2: 1, 3: 3, 4: 4, 5: 5}


@dataclass
class TransferPosition:
    planes: object
    pieces: object
    unmoved: object
    valid: object
    coordinates: object
    global_features: object
    context: dict


def encode_transfer_position(position, max_boards=16, max_tokens=MAX_TOKENS):
    import torch
    if type(max_boards) is not int or not 1 <= max_boards <= 64:
        raise ValueError("max_boards must be 1–64")
    if type(max_tokens) is not int or not 16 <= max_tokens <= MAX_TOKENS:
        raise ValueError(f"max_tokens must be 16–{MAX_TOKENS}")
    board, action, promotions, latest, total, lines = _validate(position)
    even = board[0] is None
    for timeline in board:
        for squares in timeline or []:
            if squares is not None and (len(squares) > 8 or len(squares[0]) > 8):
                raise ValueError("LC0 transfer supports boards up to 8x8; larger boards need a different spatial adapter")
    anchors = [(line, time) for line, time in latest.items() if time % 2 == action % 2]
    anchors = anchors or list(latest.items())

    def priority(key):
        line, time = key
        coordinate = _timeline_coordinate(line, even)
        distance = min(abs(coordinate - _timeline_coordinate(l, even)) + abs(time - t) / 2 for l, t in anchors)
        return (time != latest[line], distance, -time, coordinate)

    available = [(line, time) for line in latest for time, squares in enumerate(board[line]) if squares is not None]
    selected = sorted(sorted(available, key=priority)[:min(max_boards, max_tokens - 1)])
    count = len(selected)
    planes = torch.zeros(count, 112, 8, 8)
    pieces = torch.zeros(count, 64, dtype=torch.long)
    unmoved = torch.zeros(count, 64)
    valid = torch.zeros(count, 64, dtype=torch.bool)
    coordinates = []
    for index, (line, time) in enumerate(selected):
        squares = board[line][time]
        height, width = len(squares), len(squares[0])
        black = time % 2
        # All local board tensors use the same side-to-move rank orientation.
        # History slots stay on this timeline and stop at gaps/branch origins.
        for age in range(8):
            past = time - age
            if past < 0 or board[line][past] is None:
                break
            for rank, row in enumerate(board[line][past]):
                for file, piece in enumerate(row):
                    if not piece:
                        continue
                    code = abs(piece)
                    kind = (code - 1) // 2
                    if kind in PIECE_PLANES:
                        color = 0 if code % 2 == 0 else 1
                        plane = age * 13 + (0 if color == black else 6) + PIECE_PLANES[kind]
                        planes[index, plane, 7 - rank if black else rank, file] = 1
        for rank, row in enumerate(squares):
            for file, piece in enumerate(row):
                square = (7 - rank if black else rank) * 8 + file
                pieces[index, square] = abs(piece)
                unmoved[index, square] = float(piece < 0)
                valid[index, square] = True
        # Classical112 color and constant edge planes. Orthodox rights absent.
        planes[index, 108] = float(black)
        planes[index, 111] = 1
        coordinates.append([_timeline_coordinate(line, even), time, latest[line] - time,
                            float(time == latest[line]), float(black), height, width,
                            float(time % 2 == action % 2), action, float(even)])
    frontier_missing = sum((line, time) not in selected for line, time in latest.items())
    globals_ = [float(action % 2), math.log1p(action), math.log1p(total), math.log1p(lines),
                float(even), count / total, float(frontier_missing > 0), math.log1p(max(latest.values()))]
    globals_.extend(float(code in {abs(p) for p in promotions}) for code in range(1, 25))
    return TransferPosition(planes, pieces, unmoved, valid, torch.tensor(coordinates), torch.tensor(globals_), {
        "tokens": count + 1, "totalTokens": total + 1, "boards": count, "totalBoards": total,
        "truncated": count < total, "frontierTruncated": frontier_missing > 0,
        "frontierBoardsOmitted": frontier_missing, "encoding": "5d-lc0-board-projection-v1",
    })


def collate_transfer(encoded, device):
    import torch
    if not encoded:
        raise ValueError("cannot collate an empty batch")
    length = max(len(item.planes) for item in encoded)
    batch = len(encoded)
    planes = torch.zeros(batch, length, 112, 8, 8)
    pieces = torch.zeros(batch, length, 64, dtype=torch.long)
    unmoved = torch.zeros(batch, length, 64)
    valid = torch.zeros(batch, length, 64, dtype=torch.bool)
    coordinates = torch.zeros(batch, length, BOARD_FEATURES)
    padding = torch.ones(batch, length, dtype=torch.bool)
    for row, item in enumerate(encoded):
        count = len(item.planes)
        planes[row, :count], pieces[row, :count] = item.planes, item.pieces
        unmoved[row, :count], valid[row, :count] = item.unmoved, item.valid
        coordinates[row, :count] = item.coordinates
        padding[row, :count] = False
    globals_ = torch.stack([item.global_features for item in encoded])
    return tuple(item.to(device) for item in (planes, pieces, unmoved, valid, coordinates, globals_, padding))


def encode_transfer_moves(position, moves):
    """null is a legal SUBMIT candidate supplied only by the rules/data layer."""
    if not isinstance(moves, list) or not 1 <= len(moves) <= 16384:
        raise ValueError("moves must be a nonempty bounded candidate list")
    _validate(position)
    normal = [move for move in moves if move is not None]
    features = iter(encode_moves(position, normal)) if normal else iter(())
    # A negative final field distinguishes SUBMIT from every component (0/1).
    return [[0.] * (MOVE_FEATURES - 1) + [-1.] if move is None else next(features) for move in moves]
