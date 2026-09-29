"""Pre-norm Transformer with explicit multiverse coordinate features."""

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

try:
    from .encoding import COORDINATE_FEATURES, ENCODING_VERSION, GLOBAL_FEATURES, MAX_TOKENS, encode_position
    from .policy import MOVE_FEATURES, POLICY_VERSION, collate_moves, encode_moves
except ImportError:
    from encoding import COORDINATE_FEATURES, ENCODING_VERSION, GLOBAL_FEATURES, MAX_TOKENS, encode_position
    from policy import MOVE_FEATURES, POLICY_VERSION, collate_moves, encode_moves

ARCHITECTURE = "5d-transformer-value-v1"


@dataclass
class ModelConfig:
    # Approximately 20M parameters with the optional component policy.
    width: int = 512
    heads: int = 8
    layers: int = 6
    feedforward: int = 2048
    max_tokens: int = MAX_TOKENS
    dropout: float = 0.1
    policy_head: bool = False

    def __post_init__(self):
        if type(self.policy_head) is not bool:
            raise ValueError("config.policy_head must be a boolean")
        for key in ("width", "heads", "layers", "feedforward", "max_tokens"):
            if type(getattr(self, key)) is not int:
                raise ValueError(f"config.{key} must be an integer")
        if not 32 <= self.width <= 512 or self.heads not in (1, 2, 4, 8) or self.width % self.heads:
            raise ValueError("width must be 32–512 and divisible by heads (1, 2, 4, or 8)")
        if not 1 <= self.layers <= 8 or not self.width <= self.feedforward <= 2048:
            raise ValueError("layers must be 1–8 and feedforward width must be width–2048")
        if not 16 <= self.max_tokens <= MAX_TOKENS or not 0 <= self.dropout < 1:
            raise ValueError(f"max_tokens must be 16–{MAX_TOKENS} and dropout must be in [0,1)")


class TransformerValue(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.embeddings = nn.ModuleList(nn.Embedding(size, c.width) for size in (3, 25, 2, 2, 2))
        self.register_buffer("frequencies", 2.0 ** -torch.arange(8, dtype=torch.float32))
        self.coordinate_projection = nn.Linear(COORDINATE_FEATURES * 17, c.width)
        self.global_projection = nn.Linear(GLOBAL_FEATURES, c.width)
        # Construct each layer independently; copied TransformerEncoder layers
        # otherwise start from identical initial parameters.
        self.layers = nn.ModuleList(nn.TransformerEncoderLayer(
            d_model=c.width, nhead=c.heads, dim_feedforward=c.feedforward,
            dropout=c.dropout, activation="gelu", batch_first=True, norm_first=True,
        ) for _ in range(c.layers))
        self.head = nn.Sequential(nn.LayerNorm(c.width), nn.Linear(c.width, c.width // 2),
                                  nn.GELU(), nn.Linear(c.width // 2, 1), nn.Tanh())
        self.policy_trained_steps = 0
        if c.policy_head:
            self.add_policy_head()

    def add_policy_head(self):
        if hasattr(self, "policy_query"):
            return
        self.config.policy_head = True
        width = self.config.width
        self.policy_query = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU())
        self.policy_moves = nn.Sequential(nn.Linear(MOVE_FEATURES * 17, width), nn.GELU(), nn.Linear(width, width))

    def encode(self, categories, coordinates, global_features, padding_mask):
        x = sum(embedding(categories[:, :, index]) for index, embedding in enumerate(self.embeddings))
        angles = coordinates.unsqueeze(-1) * self.frequencies
        signed_log = torch.sign(coordinates) * torch.log1p(coordinates.abs()) / 10
        geometry = torch.cat((signed_log, angles.sin().flatten(-2), angles.cos().flatten(-2)), dim=-1)
        x = x + self.coordinate_projection(geometry)
        x = x + self.global_projection(global_features).unsqueeze(1)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=padding_mask)
        return x[:, 0]

    def forward(self, categories, coordinates, global_features, padding_mask):
        return self.head(self.encode(categories, coordinates, global_features, padding_mask)).squeeze(-1)

    def score_moves(self, context, features, padding_mask):
        if not self.config.policy_head:
            raise ValueError("model has no component policy head")
        angles = features.unsqueeze(-1) * self.frequencies
        signed_log = torch.sign(features) * torch.log1p(features.abs()) / 10
        geometry = torch.cat((signed_log, angles.sin().flatten(-2), angles.cos().flatten(-2)), dim=-1)
        moves = self.policy_moves(geometry)
        logits = (moves * self.policy_query(context).unsqueeze(1)).sum(-1) / math.sqrt(self.config.width)
        return logits.masked_fill(padding_mask, -torch.inf)


def collate(encoded, device):
    if not encoded:
        raise ValueError("cannot collate an empty batch")
    length = max(len(position.categories) for position in encoded)
    categories = torch.zeros((len(encoded), length, 5), dtype=torch.long)
    coordinates = torch.zeros((len(encoded), length, COORDINATE_FEATURES))
    padding = torch.ones((len(encoded), length), dtype=torch.bool)
    for index, position in enumerate(encoded):
        count = len(position.categories)
        categories[index, :count] = torch.tensor(position.categories, dtype=torch.long)
        coordinates[index, :count] = torch.tensor(position.coordinates, dtype=torch.float32)
        padding[index, :count] = False
    global_features = torch.tensor([position.global_features for position in encoded], dtype=torch.float32)
    return tuple(tensor.to(device) for tensor in (categories, coordinates, global_features, padding))


def choose_device(requested="auto"):
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Install a CUDA-enabled PyTorch wheel and check the NVIDIA driver.")
    return torch.device("cuda" if requested != "cpu" and torch.cuda.is_available() else "cpu")


def load_checkpoint(path, device, max_tokens=MAX_TOKENS):
    """Load weights with the current context budget without modifying the file.

    Context length does not affect parameter shapes. Checkpoints trained with an
    older budget therefore use the current runtime limit, while resumed training
    can explicitly request a smaller budget.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Transformer checkpoint not found: {path}. Generate training data and run neural/train.py; see docs/transformer.md.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("checkpoint architecture is incompatible with this engine")
    if checkpoint.get("encodingVersion") != ENCODING_VERSION:
        raise ValueError("checkpoint encoding version is incompatible with this engine")
    if type(checkpoint.get("trainedSteps")) is not int or checkpoint["trainedSteps"] < 1:
        raise ValueError("checkpoint has no completed training steps; random weights cannot be used as an engine")
    config = ModelConfig(**{**checkpoint["config"], "max_tokens": max_tokens})
    model = TransformerValue(config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    policy_steps = checkpoint.get("policyTrainedSteps", 0)
    if type(policy_steps) is not int or policy_steps < 0:
        raise ValueError("checkpoint policyTrainedSteps must be a nonnegative integer")
    if policy_steps and (not config.policy_head or checkpoint.get("policyVersion") != POLICY_VERSION):
        raise ValueError("checkpoint component policy is incompatible with this engine")
    model.policy_trained_steps = policy_steps
    if any(not torch.isfinite(parameter).all().item() for parameter in model.parameters()):
        raise ValueError("checkpoint contains nonfinite weights")
    model.to(device)
    return model, checkpoint


def metadata(model, checkpoint, path):
    return {"architecture": ARCHITECTURE, "config": asdict(model.config),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainedSteps": checkpoint["trainedSteps"], "checkpoint": str(Path(path).resolve()),
            "policyAvailable": model.policy_trained_steps > 0,
            "policyTrainedSteps": model.policy_trained_steps,
            "policyVersion": POLICY_VERSION if model.config.policy_head else None,
            "label": checkpoint.get("label", "Locally trained experimental value model")}


def predict(model, positions, device, batch_size=16):
    model.eval()
    values, contexts = [], []
    with torch.inference_mode():
        for start in range(0, len(positions), batch_size):
            encoded = [encode_position(position, model.config.max_tokens) for position in positions[start:start + batch_size]]
            batch = collate(encoded, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                prediction = model(*batch)
            if not torch.isfinite(prediction).all():
                raise RuntimeError("model returned nonfinite values")
            # Network scores never claim a forced mate; rules/search own that.
            cp = (1000 * torch.atanh(prediction.float().clamp(-0.999, 0.999))).cpu().tolist()
            values.extend(round(value, 3) for value in cp)
            contexts.extend(position.context for position in encoded)
    return values, contexts


def predict_policy(model, position, moves, device):
    # Validate the protocol even for a legacy model, but never return random
    # policy logits from an absent or merely initialized head.
    features = encode_moves(position, moves)
    if not model.config.policy_head or model.policy_trained_steps < 1:
        return None
    model.eval()
    encoded = encode_position(position, model.config.max_tokens)
    with torch.inference_mode():
        inputs, mask = collate_moves([features], device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            context = model.encode(*collate([encoded], device))
            scores = model.score_moves(context, inputs, mask)[0]
        if not torch.isfinite(scores).all():
            raise RuntimeError("model returned nonfinite policy scores")
        return scores.float().cpu().tolist()
