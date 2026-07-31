import unittest

from motion.core_xy import CoreXY


class StatusSequenceCoreXY(CoreXY):
    def __init__(self, reports):
        self.reports = list(reports)
        self.read_count = 0

    def status(self, timeout=1.0):
        index = min(self.read_count, len(self.reports) - 1)
        self.read_count += 1
        return self.reports[index]


class CoreXYPositionWaitTest(unittest.TestCase):
    def test_transient_idle_before_run_does_not_finish_move(self):
        machine = StatusSequenceCoreXY(
            [
                "<Idle|WPos:0.000,0.000,0.000|FS:0,0>",
                "<Run|WPos:4.000,8.000,0.000|FS:1000,0>",
                "<Idle|WPos:10.000,20.000,0.000|FS:0,0>",
            ]
        )

        result = machine.wait_until_position(
            x=10.0,
            y=20.0,
            interval=0.0,
        )

        self.assertEqual(result, "<Idle|WPos:10.000,20.000,0.000|FS:0,0>")
        self.assertEqual(machine.read_count, 3)

    def test_zero_distance_move_can_finish_on_first_idle(self):
        machine = StatusSequenceCoreXY(
            ["<Idle|WPos:10.000,20.000,3.000|FS:0,0>"]
        )

        machine.wait_until_position(z=3.0, interval=0.0)

        self.assertEqual(machine.read_count, 1)

    def test_machine_position_reports_use_cached_work_offset(self):
        machine = StatusSequenceCoreXY(
            [
                "<Run|MPos:8.000,16.000,0.000|WCO:4.000,8.000,0.000>",
                "<Idle|MPos:14.000,28.000,0.000>",
            ]
        )

        result = machine.wait_until_position(
            x=10.0,
            y=20.0,
            interval=0.0,
        )

        self.assertEqual(result, "<Idle|MPos:14.000,28.000,0.000>")
        self.assertEqual(machine.read_count, 2)


if __name__ == "__main__":
    unittest.main()
