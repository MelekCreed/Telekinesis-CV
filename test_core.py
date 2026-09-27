"""Deterministic CV math/interaction/physics tests: python -m unittest -v test_core."""
import math
import unittest

import numpy as np

from hand_tracker import estimate_velocity, palm_geometry, update_pinch, smooth_pointer
from interaction import Interaction
from physics import Object, collide, step_physics


def observation(x, y, pinch=False, label="Left"):
    points = np.tile([float(x), float(y)], (21, 1))
    points[0] += [0, 110]
    points[9] += [0, 50]
    points[5] += [-30, 55]
    points[17] += [30, 55]
    points[4] += [12 if pinch else 65, 0]
    return dict(points=points, z=np.zeros(21), label=label)


def open_palm(x=250, y=200, closed=False):
    points = np.tile([float(x), float(y)], (21, 1))
    points[0] += [0, 80]
    for lateral, base in zip((-30, -10, 10, 30), (5, 9, 13, 17)):
        points[base] += [lateral, 30]
        points[base + 1] += [lateral, 0]
        points[base + 2] += [lateral, 10 if closed else -15]
        points[base + 3] += [lateral, 35 if closed else -40]
    points[4] += [-80, 0]
    return dict(points=points, z=np.zeros(21), label="Left")


def scene(radius=45):
    obj = Object(np.array([250., 200.]), radius, (255, 180, 80), 4)
    return Interaction([obj], 640, 416, smoothing=0), obj


class MathTests(unittest.TestCase):
    def test_velocity_irregular_timestamps(self):
        times = [0, 0.018, 0.047, 0.077, 0.10]
        history = [(t, 80 + 300 * t, 70 - 120 * t) for t in times]
        np.testing.assert_allclose(estimate_velocity(history, 0.10), [300, -120])
        np.testing.assert_array_equal(estimate_velocity(history, 0.4), [0, 0])

    def test_smoothing_frame_rate_independence(self):
        for fps in (20, 30, 60, 120):
            point = np.zeros(2)
            for _ in range(fps):
                point = smooth_pointer(point, np.ones(2), 1 / fps, 0.15)
            np.testing.assert_allclose(point, np.ones(2) * (1 - math.exp(-1 / 0.15)))

    def test_pinch_scale_and_hysteresis(self):
        points = observation(250, 200, True)["points"]
        for scale in (0.3, 1, 3):
            self.assertTrue(update_pinch(points * scale + 50, False)[0])
        points[4] = points[8] + [60 * 0.38, 0]
        self.assertFalse(update_pinch(points, False)[0])
        self.assertTrue(update_pinch(points, True)[0])
        self.assertEqual(update_pinch(None, True), (False, None))

    def test_palm_and_fist_geometry(self):
        self.assertEqual(palm_geometry(open_palm()["points"])[2:], (True, False))
        self.assertEqual(palm_geometry(open_palm(closed=True)["points"])[2:], (False, True))


class InteractionTests(unittest.TestCase):
    def test_grab_offset_throw_and_stationary_release(self):
        game, obj = scene()
        game.update([observation(240, 200)], 0, 1 / 30)
        game.update([observation(240, 200, True)], 1 / 30, 1 / 30)
        np.testing.assert_allclose(obj.position, [250, 200])
        self.assertEqual(obj.holders, {0})
        for i in range(2, 8):
            game.update([observation(240 + (i - 1) * 10, 200, True)], i / 30, 1 / 30)
        game.update([observation(300, 200)], 8 / 30, 1 / 30)
        self.assertFalse(obj.holders)
        np.testing.assert_allclose(obj.velocity, [375, 0], atol=1e-5)
        # Regrasp, stop for longer than the velocity window, release: no impulse.
        x, y = obj.position
        game.update([observation(x, y, True)], 9 / 30, 1 / 30)
        for i in range(10, 20):
            game.update([observation(x, y, True)], i / 30, 1 / 30)
        game.update([observation(x, y)], 20 / 30, 1 / 30)
        np.testing.assert_array_equal(obj.velocity, [0, 0])

    def test_tracking_loss_drops_without_throw_and_requires_open(self):
        game, obj = scene()
        game.update([observation(250, 200)], 0, .03)
        game.update([observation(250, 200, True)], .03, .03)
        game.update([], .06, .03)
        self.assertFalse(obj.holders)
        np.testing.assert_array_equal(obj.velocity, [0, 0])
        game.update([observation(250, 200, True)], .09, .03)
        self.assertFalse(obj.holders)

    def test_pinch_outside_cannot_grab_by_sliding_in(self):
        game, obj = scene()
        game.update([observation(100, 200)], 0, .03)
        game.update([observation(100, 200, True)], .03, .03)
        game.update([observation(250, 200, True)], .06, .03)
        self.assertFalse(obj.holders)

    def test_invalid_landmarks_release_safely(self):
        game, obj = scene()
        game.update([observation(250, 200)], 0, .03)
        game.update([observation(250, 200, True)], .03, .03)
        bad = observation(250, 200, True)
        bad["points"][8, 0] = np.nan
        game.update([bad], .06, .03)
        self.assertFalse(obj.holders)
        np.testing.assert_array_equal(obj.velocity, [0, 0])

    def test_two_hand_transform_release_and_result_reordering(self):
        game, obj = scene()
        def pair(a, b, p=True, q=True):
            return [observation(*a, p, "Left"), observation(*b, q, "Right")]
        game.update(pair((235, 200), (265, 200), False, False), 0, .03)
        game.update(pair((235, 200), (265, 200), True, False), .03, .03)
        game.update(pair((235, 200), (265, 200)), .06, .03)
        self.assertEqual(obj.holders, {0, 1})
        game.update(pair((220, 200), (280, 200))[::-1], .09, .03)
        self.assertAlmostEqual(obj.radius, 90)
        game.update(pair((250, 170), (250, 230)), .12, .03)
        self.assertAlmostEqual(obj.angle, math.pi / 2)
        position = obj.position.copy()
        game.update(pair((250, 170), (250, 230), False, True), .15, .03)
        self.assertEqual(obj.holders, {1})
        np.testing.assert_allclose(obj.position, position)
        game.update([], .18, .03)
        self.assertFalse(obj.holders)

    def test_depth_scale_rebase(self):
        game, obj = scene()
        game.update([observation(250, 200)], 0, .03)
        game.update([observation(250, 200, True)], .03, .03)
        game.toggle_depth()
        hand = observation(250, 200, True)
        hand["points"] = (hand["points"] - [250, 200]) * 1.5 + [250, 200]
        game.update([hand], .06, .03)
        self.assertAlmostEqual(obj.radius, 67.5)

    def test_fist_dwell_toggle_once(self):
        game, obj = scene()
        game.advanced = True
        for i in range(25):
            game.update([open_palm(closed=True)], i / 30, 1 / 30)
        self.assertTrue(obj.frozen)
        self.assertEqual(game.stats["freezes"], 1)
        game.update([open_palm()], 25 / 30, 1 / 30)
        for i in range(26, 44):
            game.update([open_palm(closed=True)], i / 30, 1 / 30)
        self.assertFalse(obj.frozen)

    def test_open_palm_push(self):
        game, obj = scene()
        game.advanced = True
        for i in range(7):
            game.update([open_palm(100 + 30 * i, 200)], i / 30, 1 / 30)
        self.assertEqual(game.stats["pushes"], 1)
        self.assertGreater(np.linalg.norm(obj.velocity), 100)


class PhysicsTests(unittest.TestCase):
    def test_elastic_collision_conserves_momentum_energy(self):
        a = Object(np.array([100., 100.]), 20, (0, 0, 0), 4, np.array([100., 0.]))
        b = Object(np.array([139., 100.]), 20, (0, 0, 0), 4, np.array([-50., 0.]))
        collide(a, b, restitution=1)
        np.testing.assert_allclose(a.velocity, [-50, 0])
        np.testing.assert_allclose(b.velocity, [100, 0])
        self.assertGreaterEqual(np.linalg.norm(b.position - a.position), 40)

    def test_dt_independence_and_floor_settling(self):
        outcomes = []
        for fps in (30, 60):
            obj = Object(np.array([250., 80.]), 20, (0, 0, 0), 4, np.array([150., 0.]))
            for _ in range(fps * 10):
                step_physics([obj], 1 / fps, 640, 416)
            outcomes.append(obj.position)
            self.assertAlmostEqual(obj.position[1], 396, places=1)
            self.assertLess(np.linalg.norm(obj.velocity), 1)
        np.testing.assert_allclose(*outcomes, atol=0.2)

    def test_fast_motion_boundaries_and_frozen_body(self):
        obj = Object(np.array([600., 100.]), 20, (0, 0, 0), 4, np.array([1800., -500.]))
        step_physics([obj], .1, 640, 416)
        self.assertLess(obj.velocity[0], 0)
        self.assertTrue(20 <= obj.position[0] <= 620)
        obj.frozen = True
        pos = obj.position.copy()
        step_physics([obj], .1, 640, 416)
        np.testing.assert_array_equal(obj.position, pos)


if __name__ == "__main__":
    unittest.main()
