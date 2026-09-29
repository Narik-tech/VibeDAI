"""PyTorch transfer of LCZero's T3 dense-position attention body.

The tensor layout and inference equations are checked against LCZero's public
schema and BLAS backend (accessed 2026-09-29):
https://github.com/LeelaChessZero/lc0/blob/master/proto/net.proto
https://github.com/LeelaChessZero/lc0/blob/master/src/neural/loader.cc
https://github.com/LeelaChessZero/lc0/blob/master/src/neural/backends/blas/network_blas.cc
https://github.com/LeelaChessZero/lc0/blob/master/src/neural/backends/blas/encoder.h

This independently written implementation vendors no upstream source. LCZero's
code is GPL-3.0-or-later; weight licensing is recorded from the supplied file,
which may contain no license declaration. Source provenance is kept separately
from the new, deliberately different 5D input projection and output heads.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import torch
from torch import nn
try:
    from .lc0_weights import LC0FormatError, LC0Weights, read_lc0_weights
except ImportError:
    from lc0_weights import LC0FormatError, LC0Weights, read_lc0_weights

BACKBONE_VERSION = 'lc0-t3-pe-dense-v1'


def _activation(code: int) -> nn.Module:
    constructors = {1: nn.Mish, 2: nn.ReLU, 3: nn.Identity, 4: nn.Tanh,
                    5: nn.Sigmoid, 6: nn.SELU, 7: nn.SiLU, 8: _SquareReLU}
    if code not in constructors:
        raise LC0FormatError(f'Unsupported LCZero activation {code}')
    return constructors[code]()


class _SquareReLU(nn.Module):
    def forward(self, x):
        return x.relu().square()


class _FFN(nn.Module):
    def __init__(self, width: int, hidden: int, activation: int):
        super().__init__()
        self.dense1 = nn.Linear(width, hidden)
        self.activation = _activation(activation)
        self.dense2 = nn.Linear(hidden, width)

    def forward(self, x):
        return self.dense2(self.activation(self.dense1(x)))


class _Smolgen(nn.Module):
    def __init__(self, width: int, heads: int, spec: dict, activation: int):
        super().__init__()
        self.heads = heads
        self.generated = spec['generated']
        self.compress = nn.Linear(width, spec['channels'], bias=False)
        self.dense1 = nn.Linear(64 * spec['channels'], spec['hidden'])
        self.ln1 = nn.LayerNorm(spec['hidden'], eps=1e-3)
        self.dense2 = nn.Linear(spec['hidden'], heads * spec['generated'])
        self.ln2 = nn.LayerNorm(heads * spec['generated'], eps=1e-3)
        self.activation = _activation(activation)

    def forward(self, x, shared_projection):
        compressed = self.compress(x).flatten(1)
        hidden = self.ln1(self.activation(self.dense1(compressed)))
        generated = self.ln2(self.activation(self.dense2(hidden)))
        generated = generated.reshape(x.shape[0], self.heads, self.generated)
        return shared_projection(generated).reshape(x.shape[0], self.heads, 64, 64)


class _Encoder(nn.Module):
    def __init__(self, width: int, heads: int, layers: int, spec: dict,
                 ffn_activation: int, smolgen_activation: int):
        super().__init__()
        attention_dim = spec['attention_dim']
        self.heads = heads
        self.depth = attention_dim // heads
        self.residual_scale = (2.0 * layers) ** -0.25
        self.q = nn.Linear(width, attention_dim)
        self.k = nn.Linear(width, attention_dim)
        self.v = nn.Linear(width, attention_dim)
        self.out = nn.Linear(attention_dim, width)
        self.ln1 = nn.LayerNorm(width, eps=1e-3)
        self.ffn = _FFN(width, spec['ffn_dim'], ffn_activation)
        self.ln2 = nn.LayerNorm(width, eps=1e-3)
        self.smolgen = (_Smolgen(width, heads, spec['smolgen'], smolgen_activation)
                        if spec.get('smolgen') else None)

    def forward(self, x, shared_projection):
        def split_heads(tensor):
            return tensor.reshape(x.shape[0], 64, self.heads, self.depth).transpose(1, 2)

        q, k, v = (split_heads(projection(x)) for projection in (self.q, self.k, self.v))
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.depth)
        if self.smolgen is not None:
            logits = logits + self.smolgen(x, shared_projection)
        attention = torch.matmul(logits.softmax(-1), v)
        attention = attention.transpose(1, 2).reshape(x.shape[0], 64, self.heads * self.depth)
        x = self.ln1(x + self.residual_scale * self.out(attention))
        return self.ln2(x + self.residual_scale * self.ffn(x))


def _validate_config(config: dict) -> None:
    expected = {'version', 'embedding_dim', 'positional_channels', 'embedding_ffn_dim',
                'heads', 'default_activation', 'ffn_activation', 'smolgen_activation', 'encoders'}
    if not isinstance(config, dict) or set(config) != expected or config['version'] != BACKBONE_VERSION:
        raise LC0FormatError('Unsupported LCZero backbone configuration')
    def size(value, maximum=8192):
        if type(value) is not int or not 1 <= value <= maximum:
            raise LC0FormatError(f'Invalid LCZero backbone dimension {value!r}')
    for name in ('embedding_dim', 'positional_channels', 'embedding_ffn_dim'):
        size(config[name])
    size(config['heads'], 256)
    for name in ('default_activation', 'ffn_activation', 'smolgen_activation'):
        _activation(config[name])
    if not isinstance(config['encoders'], list) or not 1 <= len(config['encoders']) <= 256:
        raise LC0FormatError('Invalid LCZero backbone encoder list')
    generated_sizes = set()
    for spec in config['encoders']:
        if not isinstance(spec, dict) or set(spec) != {'attention_dim', 'ffn_dim', 'smolgen'}:
            raise LC0FormatError('Invalid LCZero encoder configuration')
        size(spec['attention_dim'])
        size(spec['ffn_dim'])
        if spec['attention_dim'] % config['heads']:
            raise LC0FormatError('LCZero attention width is not divisible by its head count')
        if spec['smolgen'] is not None:
            if set(spec['smolgen']) != {'channels', 'hidden', 'generated'}:
                raise LC0FormatError('Invalid smolgen configuration')
            for value in spec['smolgen'].values():
                size(value)
            generated_sizes.add(spec['smolgen']['generated'])
    if len(generated_sizes) > 1:
        raise LC0FormatError('Smolgen projection must be shared across all encoders')


class LC0Backbone(nn.Module):
    """Reconstructible body: [batch,112,8,8] -> [batch,64,embedding_dim].

    Constructing from a config alone initializes an architecture. Production
    callers must load a complete state_dict or use load_lc0_backbone; the loader
    has no partial-transfer or random-fallback path.
    """
    def __init__(self, config: dict):
        super().__init__()
        _validate_config(config)
        self.config = copy.deepcopy(config)
        self.metadata = {}
        self.embedding_dim = width = config['embedding_dim']
        positional_channels = config['positional_channels']
        self.preprocess = nn.Linear(64 * 12, 64 * positional_channels)
        self.embedding = nn.Linear(112 + positional_channels, width)
        self.activation = _activation(config['default_activation'])
        self.embedding_norm = nn.LayerNorm(width, eps=1e-3)
        self.mult_gate = nn.Parameter(torch.ones(64, width))
        self.add_gate = nn.Parameter(torch.zeros(64, width))
        self.embedding_ffn = _FFN(width, config['embedding_ffn_dim'], config['ffn_activation'])
        self.embedding_ffn_norm = nn.LayerNorm(width, eps=1e-3)
        self.residual_scale = (2.0 * len(config['encoders'])) ** -0.25
        generated = next((e['smolgen']['generated'] for e in config['encoders'] if e['smolgen']), None)
        self.smolgen_projection = nn.Linear(generated, 4096, bias=False) if generated else None
        self.encoders = nn.ModuleList([
            _Encoder(width, config['heads'], len(config['encoders']), spec,
                     config['ffn_activation'], config['smolgen_activation'])
            for spec in config['encoders']
        ])

    def forward(self, planes: torch.Tensor) -> torch.Tensor:
        if planes.ndim != 4 or tuple(planes.shape[1:]) != (112, 8, 8):
            raise ValueError('LCZero backbone input must have shape [batch,112,8,8]')
        if not planes.is_floating_point():
            raise ValueError('LCZero backbone inputs must be floating point')
        squares = planes.flatten(2).transpose(1, 2)
        positional = self.preprocess(squares[:, :, :12].flatten(1))
        positional = positional.reshape(planes.shape[0], 64, self.config['positional_channels'])
        x = self.embedding_norm(self.activation(self.embedding(torch.cat((squares, positional), dim=-1))))
        x = x * self.mult_gate + self.add_gate
        x = self.embedding_ffn_norm(x + self.residual_scale * self.embedding_ffn(x))
        for encoder in self.encoders:
            x = encoder(x, self.smolgen_projection)
        return x


def _infer_config(weights: LC0Weights) -> dict:
    tensors, metadata = weights.tensors, weights.metadata
    formats = metadata['format']
    if formats['input'] != 1:
        raise LC0FormatError('This transfer supports INPUT_CLASSICAL_112_PLANE (1) only')
    if formats['input_embedding'] != 2:
        raise LC0FormatError('This transfer supports T3 INPUT_EMBEDDING_PE_DENSE (2) only')
    if formats['default_activation'] not in (0, 1):
        raise LC0FormatError('Unsupported default activation')
    default_activation = 1 if formats['default_activation'] == 1 else 2
    def count(name):
        if name not in tensors:
            raise LC0FormatError(f'Missing required LCZero tensor {name}')
        return tensors[name].count
    width = count('ip_emb_b')
    positional_count = count('ip_emb_preproc_b')
    if positional_count % 64:
        raise LC0FormatError('Dense positional encoding must have 64 squares')
    heads = metadata['heads']
    if type(heads) is not int or not 1 <= heads <= 256:
        raise LC0FormatError('Invalid LCZero head count')
    config = {
        'version': BACKBONE_VERSION, 'embedding_dim': width,
        'positional_channels': positional_count // 64,
        'embedding_ffn_dim': count('embedding_ffn.dense1_b'),
        'heads': heads, 'default_activation': default_activation,
        'ffn_activation': formats['ffn_activation'] or default_activation,
        'smolgen_activation': formats['smolgen_activation'] or default_activation,
        'encoders': [],
    }
    for index in range(metadata['encoder_layers']):
        prefix = f'encoder.{index}.'
        smolgen = None
        if prefix + 'smolgen.compress' in tensors:
            compressed_count = count(prefix + 'smolgen.compress')
            generated_count = count(prefix + 'smolgen.dense2_b')
            if compressed_count % width or generated_count % heads:
                raise LC0FormatError('Invalid smolgen dimensions')
            smolgen = {'channels': compressed_count // width,
                       'hidden': count(prefix + 'smolgen.dense1_b'),
                       'generated': generated_count // heads}
        config['encoders'].append({'attention_dim': count(prefix + 'mha.q_b'),
                                   'ffn_dim': count(prefix + 'ffn.dense1_b'),
                                   'smolgen': smolgen})
    _validate_config(config)
    return config


def load_lc0_backbone(path: str | Path, *, max_uncompressed_bytes: int = 512 * 1024 * 1024) -> tuple[LC0Backbone, dict]:
    """Transfer every T3 body parameter, with a strict and auditable manifest."""
    weights = read_lc0_weights(path, max_uncompressed_bytes=max_uncompressed_bytes)
    # Validate all shapes before allocating: a malformed tiny file must not
    # induce a huge random model allocation just by declaring large biases.
    with torch.device('meta'):
        model = LC0Backbone(_infer_config(weights))
    mapping = {
        'preprocess.weight': 'ip_emb_preproc_w', 'preprocess.bias': 'ip_emb_preproc_b',
        'embedding.weight': 'ip_emb_w', 'embedding.bias': 'ip_emb_b',
        'embedding_norm.weight': 'ip_emb_ln_gammas', 'embedding_norm.bias': 'ip_emb_ln_betas',
        'mult_gate': 'ip_mult_gate', 'add_gate': 'ip_add_gate',
        'embedding_ffn_norm.weight': 'ip_emb_ffn_ln_gammas',
        'embedding_ffn_norm.bias': 'ip_emb_ffn_ln_betas',
    }
    def ffn_mapping(target, source):
        for dense in ('dense1', 'dense2'):
            mapping[f'{target}.{dense}.weight'] = f'{source}.{dense}_w'
            mapping[f'{target}.{dense}.bias'] = f'{source}.{dense}_b'
    ffn_mapping('embedding_ffn', 'embedding_ffn')
    if model.smolgen_projection is not None:
        mapping['smolgen_projection.weight'] = 'smolgen_w'
    for index, encoder in enumerate(model.encoders):
        target, source = f'encoders.{index}', f'encoder.{index}'
        for projection, key in (('q', 'q'), ('k', 'k'), ('v', 'v'), ('out', 'dense')):
            mapping[f'{target}.{projection}.weight'] = f'{source}.mha.{key}_w'
            mapping[f'{target}.{projection}.bias'] = f'{source}.mha.{key}_b'
        for norm in ('ln1', 'ln2'):
            mapping[f'{target}.{norm}.weight'] = f'{source}.{norm}_gammas'
            mapping[f'{target}.{norm}.bias'] = f'{source}.{norm}_betas'
        ffn_mapping(f'{target}.ffn', f'{source}.ffn')
        if encoder.smolgen is not None:
            mapping[f'{target}.smolgen.compress.weight'] = f'{source}.smolgen.compress'
            ffn_mapping(f'{target}.smolgen', f'{source}.smolgen')
            for norm in ('ln1', 'ln2'):
                mapping[f'{target}.smolgen.{norm}.weight'] = f'{source}.smolgen.{norm}_gammas'
                mapping[f'{target}.smolgen.{norm}.bias'] = f'{source}.smolgen.{norm}_betas'
    target_parameters = dict(model.named_parameters())
    if set(mapping) != set(target_parameters):
        raise LC0FormatError('Internal transfer mapping does not cover every backbone parameter')
    for target, source in mapping.items():
        if source not in weights.tensors:
            raise LC0FormatError(f'Missing required LCZero tensor {source}')
        shape = ((model.embedding_dim, 64) if target in ('mult_gate', 'add_gate')
                 else tuple(target_parameters[target].shape))
        tensor = weights.tensors[source]
        if tensor.count != math.prod(shape):
            raise LC0FormatError(f'{source} has {tensor.count} values; expected shape {shape}')
        if tensor.dims and tensor.dims != shape:
            raise LC0FormatError(f'{source} declares dimensions {tensor.dims}; expected shape {shape}')
    reused = set(mapping.values())
    unused = set(weights.tensors) - reused
    # Heads are intentionally replaced for 5D candidate moves and multiverse value.
    permitted_omissions = lambda name: (name.startswith(('policy_heads.', 'value_heads.'))
                                        or name.startswith(('ip_pol_', 'ip2_pol_', 'ip3_pol_', 'ip4_pol_',
                                                            'ip_val_', 'ip1_val_', 'ip2_val_',
                                                            'ip_mov_', 'ip1_mov_', 'ip2_mov_')))
    unsupported = sorted(name for name in unused if not permitted_omissions(name))
    if unsupported:
        raise LC0FormatError(f'Unconsumed LCZero body tensors: {unsupported}')
    model.to_empty(device='cpu')
    target_parameters = dict(model.named_parameters())
    with torch.no_grad():
        for target, source in mapping.items():
            parameter = target_parameters[target]
            if target in ('mult_gate', 'add_gate'):
                values = weights.tensors[source].tensor((model.embedding_dim, 64)).t()
            else:
                values = weights.tensors[source].tensor(tuple(parameter.shape))
            parameter.copy_(values)
    metadata = copy.deepcopy(weights.metadata)
    metadata.update({
        'backbone_config': model.config,
        'reused_tensor_count': len(reused),
        'reused_parameter_count': sum(weights.tensors[n].count for n in reused),
        'omitted_parameter_count': sum(weights.tensors[n].count for n in unused),
        'omitted_tensor_names': sorted(unused),
        'transfer_mapping': mapping,
        'transfer_scope': 'All dense positional embedding, gating, embedding FFN, encoder and smolgen body tensors; orthodox policy/value/moves-left heads replaced.',
        'parity_status': 'Equations and tensor layout follow LCZero BLAS; no native LCZero output parity benchmark has been run. 5D input projection is a separate adaptation.',
    })
    model.metadata = metadata
    model.eval()
    return model, metadata
