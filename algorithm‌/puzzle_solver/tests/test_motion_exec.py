import unittest

import numpy as np

from motion.motion_exec import MotionExecutor, PickPlaceConfig, build_pick_place_plan


class InitialTransform:
    matrix = np.asarray(
        [[0.0, 1.0, -29.0], [-1.0, 0.0, 9.5], [0.0, 0.0, 1.0]],
        dtype=float,
    )

    def to_grbl(self, points):
        points = np.asarray(points, dtype=float)
        homogeneous = np.column_stack((points, np.ones(len(points))))
        return (homogeneous @ self.matrix.T)[:, :2]


class IdentityTransform:
    def to_grbl(self, points):
        return np.asarray(points, dtype=float)


class FakeCoreXY:
    def __init__(self, event_log=None):
        self.uart = object()
        self.moves = []
        self.commands = []
        self.event_log = event_log

    def command(self, line):
        self.commands.append(line)
        return ["ok"]

    def set_work_position(self, x=None, y=None, z=None):
        self.commands.append(("set_work_position", x, y, z))
        return ["ok"]

    def move_to(self, **kwargs):
        self.moves.append(kwargs)
        if self.event_log is not None:
            self.event_log.append(("move", kwargs))

    def wait_until_position(self, **targets):
        if self.event_log is not None:
            self.event_log.append(("arrive", targets))
        return "<Idle>"

    def close(self):
        pass


class FakeServo:
    def __init__(self, event_log=None):
        self.uart = object()
        self.current = 0.0
        self.moves = []
        self.event_log = event_log

    def angle(self, multi_turn=False):
        return self.current

    def move(self, angle, **kwargs):
        self.current = float(angle)
        self.moves.append(self.current)
        if self.event_log is not None:
            self.event_log.append(("servo", self.current))
        return self.current

    def close(self):
        pass


class FakeMag:
    def __init__(self):
        self.events = []

    def on(self):
        self.events.append("on")

    def off(self):
        self.events.append("off")

    def close(self):
        pass


class MotionPlanTest(unittest.TestCase):
    def test_solution_points_and_rotation_become_motion_plan(self):
        solution = {
            "pieces": [
                {
                    "id": "piece_0",
                    "pickup_source_mm": [120.0, 130.0],
                    "pickup_target_mm": [100.0, 50.0],
                    "rotation_deg": 90.0,
                }
            ]
        }
        plan = build_pick_place_plan(solution, InitialTransform(), PickPlaceConfig())
        self.assertEqual(plan[0]["pickup_source_grbl_mm"], [101.0, -110.5])
        self.assertEqual(plan[0]["pickup_target_grbl_mm"], [21.0, -90.5])
        self.assertEqual(plan[0]["servo_target_deg"], 90.0)

    def test_servo_direction_is_configurable(self):
        solution = {
            "pieces": [
                {
                    "pickup_source_mm": [120.0, 130.0],
                    "pickup_target_mm": [100.0, 50.0],
                    "rotation_deg": 90.0,
                }
            ]
        }
        config = PickPlaceConfig(servo_direction=-1.0)
        plan = build_pick_place_plan(solution, InitialTransform(), config)
        self.assertEqual(plan[0]["servo_target_deg"], -90.0)

    def test_piece_order_minimizes_xy_route(self):
        solution = {
            "pieces": [
                {
                    "id": "far",
                    "pickup_source_mm": [100.0, 0.0],
                    "pickup_target_mm": [90.0, 0.0],
                    "rotation_deg": 0.0,
                },
                {
                    "id": "near",
                    "pickup_source_mm": [10.0, 0.0],
                    "pickup_target_mm": [20.0, 0.0],
                    "rotation_deg": 0.0,
                },
            ]
        }
        plan = build_pick_place_plan(solution, IdentityTransform(), PickPlaceConfig())
        self.assertEqual([step["id"] for step in plan], ["near", "far"])

    def test_servo_keeps_angle_between_pieces(self):
        solution = {
            "pieces": [
                {
                    "id": "a",
                    "pickup_source_mm": [10.0, 10.0],
                    "pickup_target_mm": [20.0, 20.0],
                    "rotation_deg": 30.0,
                },
                {
                    "id": "b",
                    "pickup_source_mm": [30.0, 30.0],
                    "pickup_target_mm": [40.0, 40.0],
                    "rotation_deg": 40.0,
                },
            ]
        }
        corexy = FakeCoreXY()
        servo = FakeServo()
        mag = FakeMag()
        executor = MotionExecutor(corexy=corexy, servo=servo, mag=mag)
        executor.execute_solution(
            solution,
            IdentityTransform(),
            PickPlaceConfig(
                optimize_piece_order=False,
                return_xy_zero=False,
                magnet_settle_s=0.0,
            ),
        )
        self.assertEqual(servo.moves, [30.0, 70.0, 0.0])
        self.assertEqual(mag.events, ["on", "off", "on", "off"])

    def test_piece_rotates_only_after_arriving_at_target_xy(self):
        events = []
        solution = {
            "pieces": [
                {
                    "id": "a",
                    "pickup_source_mm": [10.0, 10.0],
                    "pickup_target_mm": [20.0, 20.0],
                    "rotation_deg": 30.0,
                }
            ]
        }
        executor = MotionExecutor(
            corexy=FakeCoreXY(events),
            servo=FakeServo(events),
            mag=FakeMag(),
        )
        executor.execute_solution(
            solution,
            IdentityTransform(),
            PickPlaceConfig(return_xy_zero=False, magnet_settle_s=0.0),
        )
        target_move_index = next(
            index
            for index, event in enumerate(events)
            if event[0] == "move"
            and event[1].get("x") == 20.0
            and event[1].get("y") == 20.0
        )
        rotation_index = events.index(("servo", 30.0))
        self.assertLess(target_move_index, rotation_index)

    def test_motion_stage_timings_are_reported(self):
        solution = {
            "pieces": [
                {
                    "id": "a",
                    "pickup_source_mm": [10.0, 10.0],
                    "pickup_target_mm": [20.0, 20.0],
                    "rotation_deg": 30.0,
                }
            ]
        }
        durations = {"xy": 0.0, "z": 0.0, "servo": 0.0, "magnet_wait": 0.0}

        def record(stage, elapsed):
            durations[stage] += elapsed

        executor = MotionExecutor(
            corexy=FakeCoreXY(),
            servo=FakeServo(),
            mag=FakeMag(),
        )
        executor.execute_solution(
            solution,
            IdentityTransform(),
            PickPlaceConfig(return_xy_zero=False, magnet_settle_s=0.0),
            timing_callback=record,
        )
        self.assertGreater(durations["xy"], 0.0)
        self.assertGreater(durations["z"], 0.0)
        self.assertGreater(durations["servo"], 0.0)
        self.assertGreaterEqual(durations["magnet_wait"], 0.0)

    def test_grbl_is_initialized_once_and_released_during_close(self):
        solution = {
            "pieces": [
                {
                    "id": "a",
                    "pickup_source_mm": [10.0, 10.0],
                    "pickup_target_mm": [20.0, 20.0],
                    "rotation_deg": 0.0,
                }
            ]
        }
        corexy = FakeCoreXY()
        executor = MotionExecutor(
            corexy=corexy,
            servo=FakeServo(),
            mag=FakeMag(),
            grbl_step_idle_delay_ms=255,
        )

        executor.open()
        executor.execute_solution(
            solution,
            IdentityTransform(),
            PickPlaceConfig(return_xy_zero=False, magnet_settle_s=0.0),
        )
        self.assertEqual(
            corexy.commands,
            ["$1=255", ("set_work_position", 0.0, 0.0, 0.0)],
        )

        executor.close()
        self.assertEqual(
            corexy.commands,
            [
                "$1=255",
                ("set_work_position", 0.0, 0.0, 0.0),
                "$1=0",
            ],
        )


if __name__ == "__main__":
    unittest.main()
