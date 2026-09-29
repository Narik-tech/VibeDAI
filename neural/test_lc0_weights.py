"""Portable LCZero import regressions, using a small synthetic protobuf."""
import gzip
import hashlib
import json
import math
import os
import struct
import tempfile
import unittest
from pathlib import Path

import torch

from .lc0_backbone import BACKBONE_VERSION, LC0Backbone, _Encoder, load_lc0_backbone
from .lc0_weights import LC0FormatError, WeightLayer, read_lc0_weights, wire_fields


def _varint(value):
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def _field(number, value):
    if isinstance(value, int):
        return _varint(number << 3) + _varint(value)
    return _varint((number << 3) | 2) + _varint(len(value)) + value


def _tensor(values):
    values = list(values)
    return _field(3, struct.pack('<' + 'f' * len(values), *values)) + _field(4, 4)


def make_tiny_weights(*, missing_gate=False, bad_shape=False, network=7, legacy=False):
    """Create a schema-level fixture without using the production serializer."""
    def layer(number, count, offset=0.0):
        return _field(number, _tensor((offset + (i % 19 - 9) * .005 for i in range(count))))
    width, positional, hidden, heads = 4, 2, 6, 2
    ffn = layer(1, width * hidden) + layer(2, hidden) + layer(3, width * hidden) + layer(4, width)
    smolgen = (layer(1, width) + layer(2, 3 * 64) + layer(3, 3)
               + layer(4, 3, 1.) + layer(5, 3) + layer(6, 3 * heads * 2)
               + layer(7, heads * 2) + layer(8, heads * 2, 1.) + layer(9, heads * 2))
    mha = b''.join(layer(number, width * width if number % 2 else width)
                   for number in range(1, 9)) + _field(9, smolgen)
    encoder = (_field(1, mha) + layer(2, width, 1.) + layer(3, width)
               + _field(4, ffn) + layer(5, width, 1.) + layer(6, width))
    weights = (layer(25, width * (112 + positional) - int(bad_shape)) + layer(26, width)
               + _field(27, encoder) + _field(28, heads)
               + layer(33, width * 64, 1.) + layer(35, 4096 * 2)
               + layer(37, 64 * positional * 768) + layer(38, 64 * positional)
               + layer(39, width, 1.) + layer(40, width) + _field(41, ffn)
               + layer(42, width, 1.) + layer(43, width)
               + _field(44, _field(1, layer(6, 3)))
               + _field(45, _field(3, layer(7, 4))))
    if not missing_gate:
        weights += layer(34, width * 64)
    formats = (b''.join(_field(number, value) for number, value in
                       ((1, 1), (2, 2), (3, 6 if legacy else network), (4, 3),
                        (5, 2), (7, 1), (8, 7), (9, 0))))
    if not legacy:
        formats += _field(10, 2)
    root = (b'\x0d' + struct.pack('<I', 0x1c0) + _field(4, _field(1, 1) + _field(2, formats))
            + _field(5, _field(1, 123)) + _field(10, weights))
    return root


class WireReaderTests(unittest.TestCase):
    def test_rejects_truncated_or_invalid_fields(self):
        for malformed in (b'\x80', b'\x00', b'\x0a\x05abc', b'\x0dabc', b'\x0b', b'\x08' + b'\xff' * 10):
            with self.subTest(malformed=malformed), self.assertRaises(LC0FormatError):
                list(wire_fields(malformed))

    def test_linear16_decoding_has_exact_endpoints(self):
        layer = WeightLayer(-2., 3., memoryview(struct.pack('<3H', 0, 32768, 65535)), 1)
        actual = layer.tensor((3,))
        self.assertEqual(actual[0].item(), -2.)
        self.assertEqual(actual[-1].item(), 3.)
        self.assertAlmostEqual(actual[1].item(), -2 + 5 * 32768 / 65535, places=6)

    def test_float16_float32_and_bfloat16_decoding(self):
        cases = ((2, struct.pack('<2e', 1.5, -2.)),
                 (3, struct.pack('<2H', 0x3fc0, 0xc000)),
                 (4, struct.pack('<2f', 1.5, -2.)))
        for encoding, raw in cases:
            torch.testing.assert_close(WeightLayer(0., 0., memoryview(raw), encoding).tensor((2,)),
                                       torch.tensor([1.5, -2.]))

    def test_nonfinite_and_shape_mismatch_fail(self):
        layer = WeightLayer(0., 0., memoryview(struct.pack('<f', math.inf)), 4)
        with self.assertRaisesRegex(LC0FormatError, 'nonfinite'):
            layer.tensor((1,))
        with self.assertRaisesRegex(LC0FormatError, 'expected shape'):
            layer.tensor((2,))

    def test_uncompressed_limit_and_bad_magic(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'weights.pb.gz'
            path.write_bytes(gzip.compress(b'x' * 100))
            with self.assertRaisesRegex(LC0FormatError, 'size limit'):
                read_lc0_weights(path, max_uncompressed_bytes=32)
            path.write_bytes(b'\x0d' + struct.pack('<I', 123))
            with self.assertRaisesRegex(LC0FormatError, 'magic'):
                read_lc0_weights(path)


class BackboneTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def _load(self, **kwargs):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / 'tiny.pb.gz'
        path.write_bytes(gzip.compress(make_tiny_weights(**kwargs)))
        model, metadata = load_lc0_backbone(path)
        return model, metadata, path

    def test_complete_transfer_manifest_and_quantities(self):
        model, metadata, path = self._load()
        self.assertEqual(metadata['source_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(metadata['training_steps'], 123)
        self.assertEqual(metadata['reused_parameter_count'], sum(p.numel() for p in model.parameters()))
        self.assertEqual(metadata['omitted_parameter_count'], 7)
        self.assertEqual(metadata['reused_tensor_count'], len(list(model.parameters())))
        self.assertEqual(set(metadata['transfer_mapping']), set(dict(model.named_parameters())))
        json.dumps(metadata)

    def test_exact_gate_transpose_and_linear_layout(self):
        model, _, path = self._load()
        weights = read_lc0_weights(path)
        gate = weights.tensors['ip_add_gate'].tensor((4, 64))
        self.assertEqual(model.add_gate[23, 2].item(), gate[2, 23].item())
        raw = weights.tensors['ip_emb_w'].tensor((4, 114))
        torch.testing.assert_close(model.embedding.weight, raw, rtol=0, atol=0)

    def test_legacy_t3_compatibility_uses_official_upgrade(self):
        _, metadata, _ = self._load(legacy=True)
        self.assertEqual(metadata['declared_format']['network'], 6)
        self.assertEqual(metadata['declared_format']['input_embedding'], 0)
        self.assertEqual(metadata['format']['network'], 7)
        self.assertEqual(metadata['format']['input_embedding'], 2)

    def test_missing_or_wrong_shape_never_falls_back(self):
        for kwargs, message in (({'missing_gate': True}, 'Missing required'),
                                ({'bad_shape': True}, 'expected shape'),
                                ({'network': 3}, 'attention-body')):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(LC0FormatError, message):
                self._load(**kwargs)

    def test_dense_position_input_is_square_major_current_twelve_planes(self):
        model, _, _ = self._load()
        captured = []
        hook = model.preprocess.register_forward_pre_hook(lambda _, args: captured.append(args[0].clone()))
        inputs = torch.arange(112 * 64, dtype=torch.float32).reshape(1, 112, 8, 8) / 1000.
        with torch.no_grad():
            model(inputs)
        hook.remove()
        for square in (0, 1, 7, 8, 63):
            for plane in (0, 5, 11):
                self.assertEqual(captured[0][0, square * 12 + plane].item(), inputs[0, plane, square // 8, square % 8].item())

    def test_checkpoint_reconstruction_and_batch_independence(self):
        model, _, _ = self._load()
        clone = LC0Backbone(json.loads(json.dumps(model.config)))
        clone.load_state_dict(model.state_dict(), strict=True)
        x = torch.rand(2, 112, 8, 8)
        with torch.no_grad():
            expected, actual = model(x), clone(x)
            single = model(x[:1])
        self.assertEqual(tuple(actual.shape), (2, 64, 4))
        self.assertTrue(bool(torch.isfinite(actual).all()))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(expected[:1], single, rtol=2e-5, atol=2e-6)

    def test_gradients_reach_attention_and_smolgen(self):
        model, _, _ = self._load()
        output = model(torch.rand(1, 112, 8, 8))
        output.square().mean().backward()
        for parameter in (model.encoders[0].q.weight,
                          model.encoders[0].smolgen.compress.weight,
                          model.smolgen_projection.weight,
                          model.preprocess.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertGreater(parameter.grad.abs().sum().item(), 0.)

    def test_deepnorm_scales_branch_and_uses_population_variance(self):
        encoder = _Encoder(4, 2, 15, {'attention_dim': 4, 'ffn_dim': 6, 'smolgen': None}, 2, 7)
        with torch.no_grad():
            for parameter in encoder.parameters():
                parameter.zero_()
            encoder.ln1.weight.fill_(1.)
            encoder.ln2.weight.fill_(1.)
            encoder.out.bias.copy_(torch.tensor([.4, -.3, .9, -.2]))
        row = [.1, .5, -.6, .8]
        def norm(values):
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            return [(v - mean) / math.sqrt(variance + 1e-3) for v in values]
        expected = norm(norm([x + (30 ** -.25) * b for x, b in zip(row, [.4, -.3, .9, -.2])]))
        actual = encoder(torch.tensor(row).repeat(1, 64, 1), None)
        torch.testing.assert_close(actual[0, 0], torch.tensor(expected), rtol=2e-6, atol=2e-6)

    def test_invalid_input_shape_is_actionable(self):
        model, _, _ = self._load()
        with self.assertRaisesRegex(ValueError, r'\[batch,112,8,8\]'):
            model(torch.zeros(1, 111, 8, 8))

    @unittest.skipUnless(os.environ.get('LC0_TEST_WEIGHTS'), 'Set LC0_TEST_WEIGHTS for the external baseline integration test')
    def test_external_baseline(self):
        model, metadata = load_lc0_backbone(os.environ['LC0_TEST_WEIGHTS'])
        model.requires_grad_(False)
        with torch.no_grad():
            actual = model(torch.zeros(1, 112, 8, 8))
        self.assertEqual(actual.shape, (1, 64, model.embedding_dim))
        self.assertTrue(bool(torch.isfinite(actual).all()))
        self.assertEqual(metadata['reused_parameter_count'], sum(p.numel() for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
