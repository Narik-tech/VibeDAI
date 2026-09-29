"""Strict, bounded reader for LCZero's numeric protobuf weights.

No generated protobuf code or executable checkpoint deserialization is needed.
Field numbers follow the public LCZero schema:
https://github.com/LeelaChessZero/lc0/blob/master/proto/net.proto
"""
from __future__ import annotations

import gzip
import hashlib
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


class LC0FormatError(ValueError):
    """An invalid or unsupported LCZero model was supplied."""


def _varint(data: memoryview, offset: int) -> tuple[int, int]:
    value = 0
    for shift in range(0, 70, 7):
        if offset >= len(data):
            raise LC0FormatError("Truncated protobuf varint")
        byte = data[offset]
        offset += 1
        if shift == 63 and byte > 1:
            raise LC0FormatError("Protobuf varint exceeds 64 bits")
        value |= (byte & 127) << shift
        if not byte & 128:
            return value, offset
    raise LC0FormatError("Invalid protobuf varint")


def wire_fields(data: bytes | memoryview) -> Iterator[tuple[int, int, int | memoryview]]:
    """Read protobuf scalar/byte fields, rejecting truncation and groups."""
    view = memoryview(data)
    offset = 0
    count = 0
    while offset < len(view):
        count += 1
        if count > 100_000:
            raise LC0FormatError("Too many protobuf fields")
        tag, offset = _varint(view, offset)
        number, wire = tag >> 3, tag & 7
        if number == 0 or number > 536870911:
            raise LC0FormatError("Invalid protobuf field number")
        if wire == 0:
            value, offset = _varint(view, offset)
        elif wire in (1, 2, 5):
            if wire == 2:
                size, offset = _varint(view, offset)
            else:
                size = 8 if wire == 1 else 4
            end = offset + size
            if end > len(view):
                raise LC0FormatError("Truncated protobuf field")
            value, offset = view[offset:end], end
        else:
            raise LC0FormatError(f"Unsupported protobuf wire type {wire}")
        yield number, wire, value


def _message(data: bytes | memoryview) -> dict[int, list[tuple[int, int | memoryview]]]:
    result: dict[int, list[tuple[int, int | memoryview]]] = {}
    for number, wire, value in wire_fields(data):
        result.setdefault(number, []).append((wire, value))
    return result


def _single(fields: dict, number: int, wire: int, default=None):
    values = fields.get(number, [])
    if not values:
        return default
    if len(values) != 1 or values[0][0] != wire:
        raise LC0FormatError(f"Invalid or duplicate protobuf field {number}")
    return values[0][1]


@dataclass(frozen=True)
class WeightLayer:
    minimum: float
    maximum: float
    data: memoryview
    encoding: int
    dims: tuple[int, ...] = ()

    @property
    def count(self) -> int:
        itemsize = 4 if self.encoding == 4 else 2
        if self.encoding not in (1, 2, 3, 4) or len(self.data) % itemsize:
            raise LC0FormatError("Invalid LCZero tensor encoding or byte length")
        return len(self.data) // itemsize

    def tensor(self, shape: tuple[int, ...]):
        import torch

        if math.prod(shape) != self.count:
            raise LC0FormatError(f"Tensor has {self.count} values; expected shape {shape}")
        if self.dims and self.dims != shape:
            raise LC0FormatError(f"Declared tensor dimensions {self.dims} disagree with {shape}")
        if self.encoding == 1:
            values = torch.frombuffer(bytearray(self.data), dtype=torch.uint16).to(torch.float32)
            values.mul_((self.maximum - self.minimum) / 65535.0).add_(self.minimum)
        else:
            dtype = {2: torch.float16, 3: torch.bfloat16, 4: torch.float32}[self.encoding]
            values = torch.frombuffer(bytearray(self.data), dtype=dtype).to(torch.float32)
        if not bool(torch.isfinite(values).all()):
            raise LC0FormatError("LCZero tensor contains nonfinite weights")
        return values.reshape(shape)


def _layer(data: memoryview, default_encoding: int) -> WeightLayer:
    fields = _message(data)
    if set(fields) - {1, 2, 3, 4, 5}:
        raise LC0FormatError("Unsupported LCZero tensor fields")
    raw = _single(fields, 3, 2)
    if raw is None or len(raw) == 0:
        raise LC0FormatError("Empty LCZero weight tensor")
    minimum = struct.unpack('<f', _single(fields, 1, 5, b'\0' * 4))[0]
    maximum = struct.unpack('<f', _single(fields, 2, 5, b'\0' * 4))[0]
    encoding = _single(fields, 4, 0, default_encoding)
    if encoding == 1 and (not math.isfinite(minimum) or not math.isfinite(maximum) or maximum < minimum):
        raise LC0FormatError("Invalid LINEAR16 quantization range")
    dims = []
    for wire, value in fields.get(5, []):
        if wire == 0:
            dims.append(value)
        elif wire == 2:
            offset = 0
            while offset < len(value):
                dimension, offset = _varint(value, offset)
                dims.append(dimension)
        else:
            raise LC0FormatError("Invalid tensor dimensions")
    layer = WeightLayer(minimum, maximum, raw, encoding, tuple(dims))
    if dims and (len(dims) > 8 or any(d <= 0 for d in dims) or math.prod(dims) != layer.count):
        raise LC0FormatError("Invalid tensor dimensions")
    _ = layer.count
    return layer


_WEIGHT_NAMES = {
    4: 'ip_pol_w', 5: 'ip_pol_b', 7: 'ip1_val_w', 8: 'ip1_val_b',
    9: 'ip2_val_w', 10: 'ip2_val_b', 13: 'ip1_mov_w', 14: 'ip1_mov_b',
    15: 'ip2_mov_w', 16: 'ip2_mov_b', 17: 'ip2_pol_w', 18: 'ip2_pol_b',
    19: 'ip3_pol_w', 20: 'ip3_pol_b', 22: 'ip4_pol_w', 25: 'ip_emb_w',
    26: 'ip_emb_b', 29: 'ip_val_w', 30: 'ip_val_b', 31: 'ip_mov_w',
    32: 'ip_mov_b', 33: 'ip_mult_gate', 34: 'ip_add_gate',
    35: 'smolgen_w', 36: 'smolgen_b', 37: 'ip_emb_preproc_w',
    38: 'ip_emb_preproc_b', 39: 'ip_emb_ln_gammas', 40: 'ip_emb_ln_betas',
    42: 'ip_emb_ffn_ln_gammas', 43: 'ip_emb_ffn_ln_betas',
}
_ENCODER_NAMES = {2: 'ln1_gammas', 3: 'ln1_betas', 5: 'ln2_gammas', 6: 'ln2_betas'}
_MHA_NAMES = {1: 'q_w', 2: 'q_b', 3: 'k_w', 4: 'k_b', 5: 'v_w', 6: 'v_b', 7: 'dense_w', 8: 'dense_b'}
_FFN_NAMES = {1: 'dense1_w', 2: 'dense1_b', 3: 'dense2_w', 4: 'dense2_b'}
_SMOLGEN_NAMES = {1: 'compress', 2: 'dense1_w', 3: 'dense1_b', 4: 'ln1_gammas', 5: 'ln1_betas', 6: 'dense2_w', 7: 'dense2_b', 8: 'ln2_gammas', 9: 'ln2_betas'}


def _named_layers(fields, names, prefix, encoding, destination):
    for number, name in names.items():
        value = _single(fields, number, 2)
        if value is not None:
            destination[f'{prefix}{name}'] = _layer(value, encoding)


@dataclass
class LC0Weights:
    tensors: dict[str, WeightLayer]
    metadata: dict


def read_lc0_weights(path: str | Path, *, max_uncompressed_bytes: int = 512 * 1024 * 1024) -> LC0Weights:
    """Read the attention-body format; never load pickle or execute model content."""
    source = Path(path).expanduser().resolve()
    if max_uncompressed_bytes <= 0:
        raise ValueError("max_uncompressed_bytes must be positive")
    if source.stat().st_size > max_uncompressed_bytes + 1024 * 1024:
        raise LC0FormatError("LCZero weights exceed the input file size limit")
    digest = hashlib.sha256()
    with source.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    try:
        with source.open('rb') as handle:
            compressed = handle.read(2) == b'\x1f\x8b'
        with (gzip.open(source, 'rb') if compressed else source.open('rb')) as handle:
            data = handle.read(max_uncompressed_bytes + 1)
    except (OSError, EOFError) as exc:
        raise LC0FormatError(f"Cannot decompress LCZero weights: {exc}") from exc
    if len(data) > max_uncompressed_bytes:
        raise LC0FormatError("LCZero weights exceed the uncompressed size limit")
    root = _message(data)
    magic_raw = _single(root, 1, 5)
    if magic_raw is None:
        raise LC0FormatError("Missing LCZero magic number")
    magic = struct.unpack('<I', magic_raw)[0]
    if magic != 0x1C0:
        raise LC0FormatError(f"Invalid LCZero magic: {magic:#x}")
    fmt = _message(_single(root, 4, 2, b''))
    network_format = _message(_single(fmt, 2, 2, b''))
    format_names = {1: 'input', 2: 'output', 3: 'network', 4: 'policy', 5: 'value', 6: 'moves_left', 7: 'default_activation', 8: 'smolgen_activation', 9: 'ffn_activation', 10: 'input_embedding'}
    formats = {name: _single(network_format, number, 0, 0) for number, name in format_names.items()}
    if set(network_format) - set(format_names):
        raise LC0FormatError("Unsupported LCZero network format fields")
    if formats['network'] not in (6, 7):
        raise LC0FormatError(f"Only LCZero attention-body networks (6/7) are supported, got {formats['network']}")
    if 11 in root:
        raise LC0FormatError("ONNX weights are unsupported")
    weights_raw = _single(root, 10, 2)
    if weights_raw is None:
        raise LC0FormatError("Missing LCZero weights")
    weights = _message(weights_raw)
    encoding = _single(fmt, 1, 0, 0)
    tensors = {}
    _named_layers(weights, _WEIGHT_NAMES, '', encoding, tensors)
    declared_formats = dict(formats)
    # Official loader.cc FixOlderWeightsFile compatibility rule for early T3.
    if formats['network'] == 6 and 44 in weights and 45 in weights:
        formats['network'] = 7
        formats['input_embedding'] = 2
    elif formats['network'] == 6 and 10 not in network_format:
        formats['input_embedding'] = 1
    for number in (1, 2, 3, 6, 11, 12, 21):
        if number in weights:
            raise LC0FormatError(f"Unsupported attention-body weight field {number}")
    unknown = set(weights) - set(_WEIGHT_NAMES) - {24, 27, 28, 41, 44, 45}
    if unknown:
        raise LC0FormatError(f"Unknown LCZero weights fields: {sorted(unknown)}")
    if 41 in weights:
        fields = _message(_single(weights, 41, 2))
        _named_layers(fields, _FFN_NAMES, 'embedding_ffn.', encoding, tensors)
        if set(fields) - set(_FFN_NAMES):
            raise LC0FormatError("Unsupported embedding FFN fields")
    # Inventory the original orthodox heads so transfer reports account for
    # deliberately omitted parameters as well as every reused body tensor.
    value_names = {1: 'ip_val_w', 2: 'ip_val_b', 3: 'ip1_val_w', 4: 'ip1_val_b', 5: 'ip2_val_w', 6: 'ip2_val_b', 7: 'ip_val_err_w', 8: 'ip_val_err_b', 9: 'ip_val_cat_w', 10: 'ip_val_cat_b'}
    policy_names = {1: 'ip_pol_w', 2: 'ip_pol_b', 3: 'ip2_pol_w', 4: 'ip2_pol_b', 5: 'ip3_pol_w', 6: 'ip3_pol_b', 7: 'ip4_pol_w'}
    for container_number, container_name, head_names, child_names in (
        (44, 'value_heads', {1: 'winner', 2: 'q', 3: 'st'}, value_names),
        (45, 'policy_heads', {3: 'vanilla', 4: 'optimistic_st', 5: 'soft', 6: 'opponent'}, policy_names),
    ):
        if container_number not in weights:
            continue
        container = _message(_single(weights, container_number, 2))
        allowed = set(head_names) | ({1, 2} if container_number == 45 else set())
        if set(container) - allowed:
            raise LC0FormatError(f"Unsupported named maps in {container_name}")
        if container_number == 45:
            _named_layers(container, {1: 'ip_pol_w', 2: 'ip_pol_b'}, container_name + '.', encoding, tensors)
        for number, name in head_names.items():
            if number in container:
                child = _message(_single(container, number, 2))
                if set(child) - set(child_names):
                    raise LC0FormatError(f"Unsupported fields in {container_name}.{name}")
                _named_layers(child, child_names, f'{container_name}.{name}.', encoding, tensors)
    encoders = weights.get(27, [])
    if not 1 <= len(encoders) <= 256:
        raise LC0FormatError("Invalid LCZero encoder count")
    for index, (wire, encoder_raw) in enumerate(encoders):
        if wire != 2:
            raise LC0FormatError("Invalid LCZero encoder")
        encoder = _message(encoder_raw)
        if set(encoder) - {1, 2, 3, 4, 5, 6}:
            raise LC0FormatError("Unsupported encoder fields")
        prefix = f'encoder.{index}.'
        _named_layers(encoder, _ENCODER_NAMES, prefix, encoding, tensors)
        for number, subname, names in ((1, 'mha.', _MHA_NAMES), (4, 'ffn.', _FFN_NAMES)):
            fields = _message(_single(encoder, number, 2, b''))
            allowed = set(names) | ({9} if number == 1 else set())
            if set(fields) - allowed:
                raise LC0FormatError(f"Unsupported {subname} fields (including relative positional embeddings)")
            _named_layers(fields, names, prefix + subname, encoding, tensors)
            if number == 1 and 9 in fields:
                smolgen = _message(_single(fields, 9, 2))
                if set(smolgen) - set(_SMOLGEN_NAMES):
                    raise LC0FormatError("Unsupported smolgen fields")
                _named_layers(smolgen, _SMOLGEN_NAMES, prefix + 'smolgen.', encoding, tensors)
    license_raw = _single(root, 2, 2, b'')
    training = _message(_single(root, 5, 2, b''))
    metadata = {
        'source_path': str(source), 'source_sha256': digest.hexdigest(),
        'compressed_bytes': source.stat().st_size, 'uncompressed_bytes': len(data),
        'license': bytes(license_raw).decode('utf-8', errors='replace'),
        'format': formats, 'declared_format': declared_formats, 'weights_encoding': encoding,
        'encoder_layers': len(encoders), 'heads': _single(weights, 28, 0, 0),
        'training_steps': _single(training, 1, 0, 0),
        'tensor_count': len(tensors), 'parameter_count': sum(t.count for t in tensors.values()),
        'tensor_sizes': {name: tensor.count for name, tensor in tensors.items()},
    }
    return LC0Weights(tensors, metadata)


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description='Inspect an LCZero attention-body baseline safely.')
    parser.add_argument('weights')
    args = parser.parse_args()
    print(json.dumps(read_lc0_weights(args.weights).metadata, indent=2))


if __name__ == '__main__':
    main()
