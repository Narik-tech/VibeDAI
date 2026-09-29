"""Hierarchical LC0 transfer: shared spatial tower, board router, legal policy."""

from dataclasses import asdict, dataclass
import math
import torch
from torch import nn

try:
    from .encoding import MAX_TOKENS
    from .lc0_encoding import (TRANSFER_ENCODING_VERSION, BOARD_FEATURES,
                               encode_transfer_position, collate_transfer, encode_transfer_moves)
    from .policy import MOVE_FEATURES, POLICY_VERSION
except ImportError:
    from encoding import MAX_TOKENS
    from lc0_encoding import (TRANSFER_ENCODING_VERSION, BOARD_FEATURES,
                              encode_transfer_position, collate_transfer, encode_transfer_moves)
    from policy import MOVE_FEATURES, POLICY_VERSION

ARCHITECTURE = "5d-lc0-transfer-v1"


@dataclass
class TransferConfig:
    width: int = 256
    heads: int = 8
    layers: int = 2
    feedforward: int = 768
    max_boards: int = 16
    max_tokens: int = MAX_TOKENS
    dropout: float = 0.1
    policy_head: bool = True
    unfreeze_last: int = 0
    board_batch: int = 8

    def __post_init__(self):
        for name in ("width", "heads", "layers", "feedforward", "max_boards", "max_tokens", "unfreeze_last", "board_batch"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"config.{name} must be an integer")
        if not 32 <= self.width <= 512 or self.heads not in (1, 2, 4, 8) or self.width % self.heads:
            raise ValueError("transfer width must be 32–512 and divisible by heads")
        if not 1 <= self.layers <= 8 or not self.width <= self.feedforward <= 2048:
            raise ValueError("invalid transfer router dimensions")
        if not 1 <= self.max_boards <= 64 or not 16 <= self.max_tokens <= MAX_TOKENS:
            raise ValueError("invalid board/context budget")
        if not 0 <= self.dropout < 1 or type(self.policy_head) is not bool or not self.policy_head:
            raise ValueError("transfer model requires a policy head and dropout in [0,1)")
        if not 0 <= self.unfreeze_last <= 15 or not 1 <= self.board_batch <= 64:
            raise ValueError("unfreeze_last must be 0–15 and board_batch 1–64")


class LC0Transfer(nn.Module):
    def __init__(self, backbone, config=None, baseline=None):
        super().__init__()
        self.config = config or TransferConfig()
        self.backbone = backbone
        self.baseline_metadata = baseline or {}
        c = self.config
        self.spatial_projection = nn.Linear(backbone.embedding_dim, c.width)
        self.piece_embedding = nn.Embedding(25, c.width)
        self.square_embedding = nn.Embedding(64, c.width)
        self.unmoved_projection = nn.Linear(1, c.width, bias=False)
        self.pool_query = nn.Parameter(torch.zeros(c.width))
        self.board_projection = nn.Linear(BOARD_FEATURES * 17, c.width)
        self.global_projection = nn.Linear(32, c.width)
        self.layers = nn.ModuleList(nn.TransformerEncoderLayer(
            c.width, c.heads, c.feedforward, c.dropout, activation="gelu", batch_first=True,
            norm_first=True) for _ in range(c.layers))
        self.head = nn.Sequential(nn.LayerNorm(c.width), nn.Linear(c.width, c.width // 2),
                                  nn.GELU(), nn.Linear(c.width // 2, 1), nn.Tanh())
        self.policy_query = nn.Sequential(nn.LayerNorm(c.width), nn.Linear(c.width, c.width))
        self.policy_moves = nn.Sequential(nn.Linear(MOVE_FEATURES * 17, c.width), nn.GELU(), nn.Linear(c.width, c.width))
        self.endpoint_projection = nn.Linear(c.width * 2, c.width, bias=False)
        self.register_buffer("frequencies", 2.0 ** -torch.arange(8, dtype=torch.float32))
        self.policy_trained_steps = 0
        self.set_trainable_backbone(c.unfreeze_last)

    def set_trainable_backbone(self, last):
        self.backbone.requires_grad_(False)
        blocks = self.backbone.encoders
        if last > len(blocks):
            raise ValueError("unfreeze_last exceeds imported encoder count")
        for block in list(blocks)[len(blocks) - last:] if last else []:
            block.requires_grad_(True)
        self.config.unfreeze_last = last

    def train(self, mode=True):
        super().train(mode)
        # The imported network has no training-time dropout/statistics changes.
        self.backbone.eval()
        return self

    def encode_position(self, position):
        return encode_transfer_position(position, self.config.max_boards, self.config.max_tokens)

    @staticmethod
    def collate(encoded, device):
        return collate_transfer(encoded, device)

    encode_moves = staticmethod(encode_transfer_moves)

    def geometry(self, coordinates):
        angles = coordinates.unsqueeze(-1) * self.frequencies
        return torch.cat((torch.sign(coordinates) * torch.log1p(coordinates.abs()) / 10,
                          angles.sin().flatten(-2), angles.cos().flatten(-2)), dim=-1)

    def encode(self, planes, pieces, unmoved, valid, coordinates, global_features, padding):
        batch, boards = planes.shape[:2]
        flat = planes.flatten(0, 1)
        keep = (~padding).flatten()
        spatial = []
        selected = flat[keep]
        # No padded boards pass through the expensive shared LC0 tower.
        for start in range(0, len(selected), self.config.board_batch):
            with torch.set_grad_enabled(torch.is_grad_enabled() and self.config.unfreeze_last > 0):
                spatial.append(self.backbone(selected[start:start + self.config.board_batch]))
        features = self.spatial_projection(torch.cat(spatial))
        square = torch.zeros(batch * boards, 64, self.config.width, device=features.device, dtype=features.dtype)
        square[keep] = features
        square = square.reshape(batch, boards, 64, self.config.width)
        square = (square + self.piece_embedding(pieces) + self.square_embedding.weight
                  + self.unmoved_projection(unmoved.unsqueeze(-1)))
        scores = (square * self.pool_query).sum(-1) / math.sqrt(self.config.width)
        # Padded boards receive a harmless valid square before the router masks them.
        safe_valid = valid | padding.unsqueeze(-1)
        pooled = (scores.masked_fill(~safe_valid, -torch.inf).softmax(-1).unsqueeze(-1) * square).sum(-2)
        board_tokens = pooled + self.board_projection(self.geometry(coordinates))
        x = torch.cat((self.global_projection(global_features).unsqueeze(1), board_tokens), dim=1)
        mask = torch.cat((torch.zeros(batch, 1, dtype=torch.bool, device=padding.device), padding), dim=1)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=mask)
        return {"global": x[:, 0], "boards": x[:, 1:], "squares": square,
                "coordinates": coordinates, "padding": padding}

    def forward(self, *batch):
        return self.head(self.encode(*batch)["global"]).squeeze(-1)

    def score_moves(self, context, features, padding_mask):
        endpoints = []
        boards, coords = context["squares"], context["coordinates"]
        for offset in (0, 8):
            endpoint = features[:, :, offset:offset + 4]
            matches = (endpoint[:, :, None, :2] == coords[:, None, :, :2]).all(-1) & ~context["padding"][:, None, :]
            board_indices = matches.to(torch.int64).argmax(-1)
            batch_indices = torch.arange(len(features), device=features.device)[:, None]
            black = coords[batch_indices, board_indices, 4] > 0.5
            rank = torch.where(black, 7 - endpoint[:, :, 2], endpoint[:, :, 2]).long()
            file = endpoint[:, :, 3].long()
            square_indices = (rank * 8 + file).clamp(0, 63)
            local = boards[batch_indices, board_indices, square_indices]
            routed = context["boards"][batch_indices, board_indices]
            present = matches.any(-1) & (features[:, :, -1] >= 0)
            endpoints.append((local + routed) * present.unsqueeze(-1))
        moves = self.policy_moves(self.geometry(features)) + self.endpoint_projection(torch.cat(endpoints, dim=-1))
        logits = (moves * self.policy_query(context["global"]).unsqueeze(1)).sum(-1) / math.sqrt(self.config.width)
        return logits.masked_fill(padding_mask, -torch.inf)


def create_transfer_model(path, config=None):
    try:
        from .lc0_backbone import load_lc0_backbone
    except ImportError:
        from lc0_backbone import load_lc0_backbone
    backbone, baseline = load_lc0_backbone(path)
    return LC0Transfer(backbone, config, baseline)


def load_transfer_checkpoint(checkpoint, device, max_tokens=MAX_TOKENS):
    try:
        from .lc0_backbone import LC0Backbone
    except ImportError:
        from lc0_backbone import LC0Backbone
    if checkpoint.get("architecture") != ARCHITECTURE or checkpoint.get("encodingVersion") != TRANSFER_ENCODING_VERSION:
        raise ValueError("incompatible LC0 transfer checkpoint encoding")
    if type(checkpoint.get("trainedSteps")) is not int or checkpoint["trainedSteps"] < 1:
        raise ValueError("transfer checkpoint has no completed training steps")
    config = TransferConfig(**{**checkpoint["config"], "max_tokens": max_tokens})
    # Validate structure on the meta device before dimensions can allocate RAM.
    with torch.device("meta"):
        backbone = LC0Backbone(checkpoint["backboneConfig"])
        model = LC0Transfer(backbone, config, checkpoint.get("baseline", {}))
    state = checkpoint.get("state_dict")
    expected = model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("transfer checkpoint state_dict keys do not match its architecture")
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or tensor.shape != expected[name].shape or tensor.dtype != expected[name].dtype:
            raise ValueError(f"transfer checkpoint tensor shape/dtype mismatch: {name}")
        if not torch.isfinite(tensor).all().item():
            raise ValueError("transfer checkpoint contains nonfinite weights")
    model.load_state_dict(state, strict=True, assign=True)
    steps = checkpoint.get("policyTrainedSteps", 0)
    if type(steps) is not int or steps < 0 or (steps and checkpoint.get("policyVersion") != POLICY_VERSION):
        raise ValueError("incompatible transfer policy metadata")
    model.policy_trained_steps = steps
    model.to(device)
    return model
