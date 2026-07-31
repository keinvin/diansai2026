import json
import tempfile
import unittest
from pathlib import Path

from main.timing import STAGE_ORDER, StageTimings, append_timing_log


class StageTimingsTest(unittest.TestCase):
    def test_accumulates_and_merges_all_main_stages(self):
        timings = StageTimings.from_dict({"recognition": 1.25})
        timings.add("solve", 0.5)
        timings.add("xy", 0.25)
        values = timings.to_dict()

        self.assertEqual(tuple(values), STAGE_ORDER)
        self.assertEqual(values["recognition"], 1.25)
        self.assertEqual(values["solve"], 0.5)
        self.assertEqual(values["xy"], 0.25)
        self.assertAlmostEqual(timings.total_seconds(), 2.0)

    def test_context_manager_records_failed_stage(self):
        timings = StageTimings()
        with self.assertRaisesRegex(RuntimeError, "failure"):
            with timings.measure("recognition"):
                raise RuntimeError("failure")
        self.assertGreater(timings.to_dict()["recognition"], 0.0)

    def test_jsonl_log_contains_stage_values_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timing.jsonl"
            append_timing_log(
                path,
                "recognition_completed",
                {"recognition": 0.25, "solve": 0.75},
                piece_count=4,
            )
            record = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(record["event"], "recognition_completed")
        self.assertEqual(record["piece_count"], 4)
        self.assertEqual(record["timings_seconds"]["recognition"], 0.25)
        self.assertEqual(record["stage_total_seconds"], 1.0)


if __name__ == "__main__":
    unittest.main()
