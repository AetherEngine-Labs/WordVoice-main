import importlib
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "CosyVoice"))

common = importlib.import_module("cosyvoice.utils.common")
nucleus_sampling = common.nucleus_sampling
ras_sampling = common.ras_sampling


def reference_nucleus_sampling(weighted_scores, top_p=0.8, top_k=25):
    probabilities = []
    indices = []
    cumulative = 0.0
    sorted_probabilities, sorted_indices = weighted_scores.softmax(dim=0).sort(
        descending=True, stable=True
    )
    for index in range(len(sorted_indices)):
        if cumulative < top_p and len(probabilities) < top_k:
            cumulative += sorted_probabilities[index]
            probabilities.append(sorted_probabilities[index])
            indices.append(sorted_indices[index])
        else:
            break
    probabilities = torch.tensor(probabilities).to(weighted_scores)
    indices = torch.tensor(indices, dtype=torch.long).to(weighted_scores.device)
    return indices[probabilities.multinomial(1, replacement=True)].item()


class NucleusSamplingTest(unittest.TestCase):
    def test_matches_original_seeded_selection(self):
        generator = torch.Generator().manual_seed(37)
        for seed in range(50):
            scores = torch.randn(6563, generator=generator)
            torch.manual_seed(seed)
            expected = reference_nucleus_sampling(scores.clone())
            torch.manual_seed(seed)
            self.assertEqual(nucleus_sampling(scores.clone()), expected)

    def test_retains_first_candidate_that_reaches_top_p(self):
        scores = torch.tensor([2.0, 1.0, 0.0, -1.0])
        selected = set()
        for seed in range(100):
            torch.manual_seed(seed)
            selected.add(nucleus_sampling(scores.clone(), top_p=0.7, top_k=4))
        self.assertEqual(selected, {0, 1})

    def test_repetition_aware_retry_excludes_repeated_candidate(self):
        scores = torch.tensor([9.0, 1.0, 0.0])
        torch.manual_seed(0)
        selected = ras_sampling(
            scores,
            decoded_tokens=[0] * 10,
            sampling=25,
            top_p=0.8,
            top_k=3,
            win_size=10,
            tau_r=0.1,
        )
        self.assertNotEqual(selected, 0)
        self.assertTrue(torch.isneginf(scores[0]))


if __name__ == "__main__":
    unittest.main()
