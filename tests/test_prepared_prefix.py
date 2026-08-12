import importlib
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "CosyVoice"))

wordvoice_llm = importlib.import_module("cosyvoice.llm.wordvoice_llm")
WordVoiceLM = wordvoice_llm.WordVoiceLM


class TinyCausalModel(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(32, hidden_size)


class ImmutableCachedDecoder(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.model = TinyCausalModel(hidden_size)
        self.native_decoder = object()
        self.prefill_calls = 0

    def forward_one_step(self, inputs, masks, cache=None):
        if cache is None:
            self.prefill_calls += 1
        hidden = inputs[:, -1:, :]
        key = torch.zeros(1, 1, inputs.shape[1], 1)
        return hidden, ((key, key.clone()),)


def select_terminal(weighted_scores, decoded_tokens, sampling):
    finite = torch.isfinite(weighted_scores)
    return int(torch.arange(weighted_scores.shape[0])[finite][-1].item())


class PreparedPrefixTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.decoder = ImmutableCachedDecoder(hidden_size=8)
        self.model = WordVoiceLM(
            llm_input_size=8,
            llm_output_size=8,
            speech_token_size=2500,
            llm=self.decoder,
            sampling=select_terminal,
        ).eval()
        self.inputs = {
            "text": torch.tensor([[1]], dtype=torch.int32),
            "text_len": torch.tensor([1], dtype=torch.int32),
            "prompt_text": torch.tensor([[2]], dtype=torch.int32),
            "prompt_text_len": torch.tensor([1], dtype=torch.int32),
            "prompt_speech_token": torch.tensor([[3]], dtype=torch.int32),
            "prompt_speech_token_len": torch.tensor([1], dtype=torch.int32),
            "word_list": [torch.tensor([[4]], dtype=torch.int32)],
            "start_list": [0],
            "dur_list": [1],
            "bnd_list": [0],
            "tone_list": [0],
            "eng_list": [0],
            "f0_list": [0],
            "embedding": torch.zeros(1, 4),
        }

    def infer(self, key):
        values = {
            name: list(value) if isinstance(value, list) else value.clone()
            for name, value in self.inputs.items()
        }
        values["word_list"] = [word.clone() for word in self.inputs["word_list"]]
        return self.model.base_inference(
            **values,
            prepared_prefix_key=key,
        )

    def test_reuses_only_an_exact_prepared_prefix(self):
        first = self.infer("request-a")
        first_metrics = self.model.consume_prepared_prefix_metrics()
        second = self.infer("request-a")
        second_metrics = self.model.consume_prepared_prefix_metrics()
        third = self.infer("request-b")
        third_metrics = self.model.consume_prepared_prefix_metrics()

        self.assertEqual(first, second)
        self.assertEqual(self.decoder.prefill_calls, 2)
        self.assertEqual(first_metrics["prepared_prefix_cache"], "miss")
        self.assertEqual(second_metrics["prepared_prefix_cache"], "hit")
        self.assertEqual(third_metrics["prepared_prefix_cache"], "miss")
        self.assertGreater(second_metrics["prepared_prefix_retained_bytes"], 0)

    def test_fingerprint_changes_with_reference_embedding(self):
        args = self.inputs
        first = self.model.prepared_prefix_key(
            args["text"], args["prompt_text"], args["prompt_speech_token"],
            args["word_list"], args["start_list"], args["dur_list"],
            args["bnd_list"], args["tone_list"], args["eng_list"],
            args["f0_list"], args["embedding"],
        )
        changed_embedding = torch.ones_like(args["embedding"])
        second = self.model.prepared_prefix_key(
            args["text"], args["prompt_text"], args["prompt_speech_token"],
            args["word_list"], args["start_list"], args["dur_list"],
            args["bnd_list"], args["tone_list"], args["eng_list"],
            args["f0_list"], changed_embedding,
        )
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
