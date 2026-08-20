import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "CosyVoice"))

from cosyvoice.llm.wordvoice_trt import WordVoiceTensorRTDecoder, _flat_cache_view


class WordVoiceTensorRTBufferTest(unittest.TestCase):
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
