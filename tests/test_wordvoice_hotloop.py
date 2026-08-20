import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "CosyVoice"))

from cosyvoice.llm.wordvoice_llm import WordVoiceLM


class WordVoiceHotLoopTest(unittest.TestCase):
    def test_cached_decode_masks_match_original_rules(self):
        model = object.__new__(WordVoiceLM)
        torch.nn.Module.__init__(model)
        model.llm_decoder = torch.nn.Linear(10, 64, bias=False)
        model.silent_tokens = [1, 2, 7]
        model.bound_token = 60
        model.eos_token = 61
        model._wordvoice_mask_cache_device = None
        model._wordvoice_mask_indices = None

        for allows_silence, allows_terminal in (
            (True, True),
            (True, False),
            (False, True),
            (False, False),
        ):
            expected = torch.randn(1, 64)
            original = expected.clone()
            mask = torch.ones(64, dtype=torch.bool)
            if allows_silence:
                mask[model.silent_tokens] = False
            if allows_terminal:
                mask[[model.bound_token, model.eos_token]] = False
            if not allows_silence and not allows_terminal:
                mask[model.bound_token] = False
            expected[:, mask] = -float("inf")

            index = (
                1
                if allows_silence and allows_terminal
                else 0
                if allows_silence
                else 2
                if allows_terminal
                else 3
            )
            indices = model._wordvoice_decode_mask_indices(torch.device("cpu"))[index]
            actual = original.clone()
            actual.index_fill_(1, indices, -float("inf"))
            self.assertTrue(torch.equal(expected, actual))

        self.assertIs(
            model._wordvoice_decode_mask_indices(torch.device("cpu")),
            model._wordvoice_mask_indices,
        )


if __name__ == "__main__":
    unittest.main()
