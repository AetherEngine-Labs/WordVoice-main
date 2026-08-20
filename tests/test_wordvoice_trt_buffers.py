import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parents[1] / "CosyVoice"))

from cosyvoice.llm.wordvoice_trt import (
    WordVoiceTensorRTDecoder,
    _flat_cache_view,
    _flat_layer_cache_view,
    _flat_token_major_view,
)
from cosyvoice.bin import export_wordvoice_trt_decoder as exporter


class WordVoiceTensorRTBufferTest(unittest.TestCase):
    def test_flat_exporter_round_trips_legacy_cache_layout(self):
        class IdentityDynamicCache:
            @classmethod
            def from_legacy_cache(cls, legacy_cache):
                return legacy_cache

        class AppendOneToken(nn.Module):
            def forward(self, inputs_embeds, past_key_values, **_kwargs):
                present = tuple(
                    (
                        torch.cat(
                            (key, torch.full_like(key[:, :, :1, :], 10.0)),
                            dim=2,
                        ),
                        torch.cat(
                            (value, torch.full_like(value[:, :, :1, :], 20.0)),
                            dim=2,
                        ),
                    )
                    for key, value in past_key_values
                )
                return SimpleNamespace(
                    last_hidden_state=inputs_embeds,
                    past_key_values=present,
                )

        original_dynamic_cache = exporter.DynamicCache
        exporter.DynamicCache = IdentityDynamicCache
        try:
            wrapper = exporter.Qwen2DecodeStep(
                AppendOneToken(),
                num_layers=1,
                num_kv_heads=2,
                precision="fp32",
                cache_layout="token-major-flat-v1",
            )
            flat_input = torch.arange(1 * 3 * 4 * 2, dtype=torch.float32).reshape(
                1, 3, 4, 2
            )
            _hidden, flat_output = wrapper(
                torch.zeros(1, 1, 8), flat_input
            )
        finally:
            exporter.DynamicCache = original_dynamic_cache

        self.assertEqual(tuple(flat_output.shape), (1, 4, 4, 2))
        self.assertTrue(torch.equal(flat_output[:, :3], flat_input))
        self.assertTrue(torch.equal(flat_output[:, 3, :2], torch.full((1, 2, 2), 10.0)))
        self.assertTrue(torch.equal(flat_output[:, 3, 2:], torch.full((1, 2, 2), 20.0)))

    def test_token_major_flat_cache_reconstructs_legacy_views(self):
        buffer = torch.arange(2 * 6 * 4 * 3, dtype=torch.float32)

        flat = _flat_token_major_view(
            buffer,
            length=6,
            channels=4,
            head_dim=3,
        )
        key = _flat_layer_cache_view(
            buffer,
            length=6,
            layer=0,
            kind_index=0,
            num_layers=1,
            num_kv_heads=2,
            head_dim=3,
        )
        value = _flat_layer_cache_view(
            buffer,
            length=6,
            layer=0,
            kind_index=1,
            num_layers=1,
            num_kv_heads=2,
            head_dim=3,
        )

        self.assertTrue(flat.is_contiguous())
        self.assertEqual(tuple(key.shape), (1, 2, 6, 3))
        self.assertEqual(tuple(value.shape), (1, 2, 6, 3))
        self.assertTrue(torch.equal(key, flat[:, :, :2, :].permute(0, 2, 1, 3)))
        self.assertTrue(torch.equal(value, flat[:, :, 2:, :].permute(0, 2, 1, 3)))

    def test_cache_slots_are_reused_and_identifiable(self):
        decoder = object.__new__(WordVoiceTensorRTDecoder)
        decoder.num_kv_heads = 2
        decoder.head_dim = 3
        decoder.hidden_size = 8
        decoder.num_layers = 2
        decoder.max_past_tokens = 5
        decoder.dtype = torch.float32
        decoder._cache_buffer_slots = [None, None]
        decoder._cache_buffer_allocations = 0

        first = decoder._ensure_cache_buffer_slot(
            0, device=torch.device("cpu")
        )
        self.assertIs(first, decoder._ensure_cache_buffer_slot(0, device=torch.device("cpu")))
        self.assertEqual(decoder._cache_buffer_allocations, 1)
        self.assertEqual(
            first.numel(), 2 * 2 * 2 * (5 + 1) * 3
        )

        first_cache = _flat_cache_view(
            first,
            length=5,
            num_kv_heads=2,
            head_dim=3,
        )
        self.assertEqual(
            decoder._cache_slot_for_legacy_cache(((first_cache, first_cache),)),
            0,
        )

        second = decoder._ensure_cache_buffer_slot(
            1, device=torch.device("cpu")
        )
        self.assertEqual(decoder._cache_buffer_allocations, 2)
        self.assertNotEqual(first.data_ptr(), second.data_ptr())

    def test_flat_cache_prefix_view_remains_contiguous(self):
        buffer = torch.empty(2 * 3 * 5 * 7)

        view = _flat_cache_view(
            buffer,
            length=5,
            num_kv_heads=3,
            head_dim=7,
        )

        self.assertEqual(tuple(view.shape), (1, 3, 5, 7))
        self.assertTrue(view.is_contiguous())
        self.assertEqual(view.data_ptr(), buffer.data_ptr())

        offset_view = _flat_cache_view(
            buffer,
            offset=3 * 5 * 7,
            length=5,
            num_kv_heads=3,
            head_dim=7,
        )
        self.assertTrue(offset_view.is_contiguous())
        self.assertEqual(offset_view.storage_offset(), 3 * 5 * 7)

        with self.assertRaisesRegex(RuntimeError, "cache-buffer-capacity"):
            _flat_cache_view(
                buffer,
                offset=3 * 5 * 7,
                length=6,
                num_kv_heads=3,
                head_dim=7,
            )


if __name__ == "__main__":
    unittest.main()
