import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from cosyvoice.llm.llm import Qwen2Encoder
from cosyvoice.llm.wordvoice_llm import _cache_tuple


class TransformersCacheAdapterTest(unittest.TestCase):
    def test_wordvoice_decode_adapts_stored_legacy_cache(self):
        config = Qwen2Config(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=32,
        )
        model = Qwen2ForCausalLM(config)
        wrapper = type("Wrapper", (), {"model": model, "native_decoder": None})()

        prefill = torch.randn(1, 2, config.hidden_size)
        prefill_mask = torch.ones(1, 2, 2, dtype=torch.bool)
        _, cache = Qwen2Encoder.forward_one_step(wrapper, prefill, prefill_mask)

        legacy_cache = _cache_tuple(cache)
        decode = torch.randn(1, 1, config.hidden_size)
        decode_mask = torch.ones(1, 1, 3, dtype=torch.bool)
        hidden, next_cache = Qwen2Encoder.forward_one_step(
            wrapper,
            decode,
            decode_mask,
            legacy_cache,
        )

        self.assertEqual(hidden.shape, (1, 1, config.hidden_size))
        self.assertIsNotNone(next_cache)
