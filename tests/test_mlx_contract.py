import tempfile
import unittest
from pathlib import Path

import numpy as np

from mlx_wordvoice.contract import ControlPlan, PreparedRequest


class PreparedRequestTest(unittest.TestCase):
    def request(self):
        return PreparedRequest(
            text_tokens=np.array([[11, 12]], dtype=np.int32),
            prompt_text_tokens=np.array([[21, 22]], dtype=np.int32),
            prompt_speech_tokens=np.array([[31, 32, 33]], dtype=np.int32),
            prompt_mel=np.arange(12, dtype=np.float32).reshape(1, 3, 4),
            speaker_embedding=np.arange(4, dtype=np.float32).reshape(1, 4),
            word_tokens=(
                np.array([[41]], dtype=np.int32),
                np.array([[42, 43]], dtype=np.int32),
            ),
            controls=ControlPlan.from_sequences(
                starts=[0],
                durations=[2, 35],
                boundaries=[1, 5],
                tones=[2, 7],
                pitches=[10, 20],
                energies=[11, 20],
            ),
            metadata={
                "base_model_revision": "base",
                "reference_audio_sha256": "audio",
                "wordvoice_model_revision": "model",
                "wordvoice_source_revision": "source",
            },
        )

    def test_round_trip_is_fingerprint_exact(self):
        request = self.request()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "request"
            request.save(destination)
            loaded = PreparedRequest.load(destination)
        self.assertEqual(request.fingerprint(), loaded.fingerprint())
        np.testing.assert_array_equal(request.prompt_mel, loaded.prompt_mel)

    def test_fingerprint_changes_with_control(self):
        first = self.request()
        changed = PreparedRequest(
            **{
                **first.__dict__,
                "controls": ControlPlan.from_sequences(
                    starts=[0],
                    durations=[2, 34],
                    boundaries=[1, 5],
                    tones=[2, 7],
                    pitches=[10, 20],
                    energies=[11, 20],
                ),
            }
        )
        self.assertNotEqual(first.fingerprint(), changed.fingerprint())

    def test_rejects_misaligned_control_vectors(self):
        with self.assertRaisesRegex(ValueError, "identical"):
            ControlPlan.from_sequences(
                starts=[0],
                durations=[1, 2],
                boundaries=[1],
                tones=[1, 2],
                pitches=[1, 2],
                energies=[1, 2],
            )


if __name__ == "__main__":
    unittest.main()
