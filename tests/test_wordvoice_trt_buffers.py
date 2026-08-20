import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "CosyVoice"))

from cosyvoice.llm.wordvoice_trt import _flat_cache_view


class WordVoiceTensorRTBufferTest(unittest.TestCase):
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
