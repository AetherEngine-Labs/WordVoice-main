import importlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "CosyVoice"))

WordVoiceTensorRTDecoder = importlib.import_module(
    "cosyvoice.llm.wordvoice_trt"
).WordVoiceTensorRTDecoder


class FakeEvent:
    def __init__(self, milliseconds: float):
        self.milliseconds = milliseconds
        self.synchronized = False

    def synchronize(self):
        self.synchronized = True

    def elapsed_time(self, _other):
        return _other.milliseconds


class WordVoiceTensorRTTimingTest(unittest.TestCase):
    def test_collects_all_recorded_events_and_syncs_only_last_end(self):
        decoder = object.__new__(WordVoiceTensorRTDecoder)
        first_start = FakeEvent(0.0)
        first_end = FakeEvent(2.5)
        second_start = FakeEvent(0.0)
        second_end = FakeEvent(3.5)
        decoder._timing_events = (
            (first_start, first_end),
            (second_start, second_end),
        )
        decoder._timing_event_index = 2

        self.assertEqual(decoder._collect_device_seconds(), 0.006)
        self.assertFalse(first_end.synchronized)
        self.assertTrue(second_end.synchronized)


if __name__ == "__main__":
    unittest.main()
