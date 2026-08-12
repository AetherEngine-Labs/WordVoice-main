import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from safetensors.numpy import load_file, save_file

from mlx_wordvoice.convert import (
    PYTORCH_RAND_NOISE_SHAPE,
    PYTORCH_RAND_NOISE_VALUES_SHA256,
    build_pytorch_rand_noise,
    convert_flow_state,
    convert_llm_state,
)


class ConversionTest(unittest.TestCase):
    def test_reproduces_pytorch_cfm_fixed_diffusion_noise(self):
        import hashlib

        noise = build_pytorch_rand_noise()

        self.assertEqual(noise.shape, PYTORCH_RAND_NOISE_SHAPE)
        self.assertEqual(noise.dtype, np.float32)
        self.assertEqual(
            hashlib.sha256(noise.tobytes()).hexdigest(),
            PYTORCH_RAND_NOISE_VALUES_SHA256,
        )

    def test_splits_wordvoice_llm_controls_from_base_weights(self):
        state = {
            "llm.model.model.layers.0.self_attn.q_proj.weight": np.zeros((2, 2)),
            "speech_embedding.weight": np.zeros((3, 2)),
            "llm_decoder.weight": np.zeros((3, 2)),
            "duration_embedding.weight": np.zeros((4, 2)),
            "duration_predictor.bias": np.zeros((4,)),
            "style_loss_module.duration_weights": np.zeros((4,)),
        }
        base, controls = convert_llm_state(state)
        self.assertIn("qwen2.model.layers.0.self_attn.q_proj.weight", base)
        self.assertIn("llm.speech_embedding.weight", base)
        self.assertIn("wordvoice_llm.duration_embedding.weight", controls)
        self.assertNotIn("wordvoice_llm.style_loss_module.duration_weights", controls)

    def test_transposes_mlx_convolutions_and_splits_flow_controls(self):
        conv = np.arange(24).reshape(2, 3, 4)
        state = {
            "pre_lookahead_layer.conv1.weight": conv,
            "decoder.estimator.transformer_blocks.0.attn.to_out.0.weight": np.zeros((2, 2)),
            "control_modulator.2.weight": np.zeros((4, 2)),
        }
        base, controls = convert_flow_state(state)
        converted = base["flow.pre_lookahead_layer.conv1.weight"]
        self.assertEqual(converted.shape, (2, 4, 3))
        self.assertTrue(converted.flags.c_contiguous)
        np.testing.assert_array_equal(converted, np.swapaxes(conv, 1, 2))
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "weights.safetensors"
            save_file({"weight": converted}, path)
            np.testing.assert_array_equal(
                load_file(path)["weight"], np.swapaxes(conv, 1, 2)
            )
        self.assertIn(
            "flow.decoder.estimator.transformer_blocks.0.attn.to_out_0.weight", base
        )
        self.assertIn("wordvoice_flow.control_modulator.layers.2.weight", controls)


if __name__ == "__main__":
    unittest.main()
