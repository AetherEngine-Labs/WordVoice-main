import unittest

from mlx_wordvoice.runtime_options import (
    DEFAULT_FLOW_STEPS,
    QUALITY_REFERENCE_FLOW_STEPS,
    resolve_flow_steps,
)


class RuntimeOptionsTest(unittest.TestCase):
    def test_uses_eight_step_evaluation_default(self):
        self.assertEqual(DEFAULT_FLOW_STEPS, 8)
        self.assertEqual(resolve_flow_steps(None), 8)

    def test_retains_explicit_ten_step_quality_reference(self):
        self.assertEqual(QUALITY_REFERENCE_FLOW_STEPS, 10)
        self.assertEqual(resolve_flow_steps(10), 10)

    def test_rejects_non_positive_override(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            resolve_flow_steps(0)


if __name__ == "__main__":
    unittest.main()
