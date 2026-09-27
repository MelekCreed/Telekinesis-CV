"""Hand geometry, gesture state machines, manipulation, physics and compositing.
No webcam or model files needed."""
import math
import unittest

import numpy as np

from hand_tracker import estimate_velocity, smooth_pointer, update_pinch
from manipulation import (Group, ManipulatedObject, Manipulator, Sprite, finger_pose,
                          render_sprite, step_physics)
from selection import HoldGesture, PinchGesture, aim_point


def make_hand(tip=(320, 240), pinch=False, palm=80.0, angle=-90.0, fist=False, spread=False):
    """Synthetic 21-landmark hand in image pixels. Index points along `angle` (deg)."""
    a = math.radians(angle)
    forward = np.array([math.cos(a), math.sin(a)])
    side = np.array([-forward[1], forward[0]])
    tip = np.asarray(tip, float)
    finger = .9 * palm
    mcp = tip - forward * finger
    wrist = mcp - forward * palm
    p = np.zeros((21, 2))
    p[0] = wrist
    for k, (i, lateral) in enumerate(((5, 0), (9, -.25), (13, -.5), (17, -.75))):
        base = mcp + side * lateral * palm * (1.6 if spread else 1)
        p[i] = base
        if fist or (k > 0 and not spread):
            # Curled: tip folds back toward the wrist.
            p[i], p[i + 1] = base, base + forward * .35 * palm
            p[i + 2], p[i + 3] = base + forward * .25 * palm - side * .05 * palm, base - forward * .1 * palm
        else:
            for j in range(1, 4):
                p[i + j] = base + forward * finger * j / 3
    if fist:
        p[5:9] = [p[5], p[5] + forward * .35 * palm, p[5] + forward * .25 * palm, p[5] - forward * .1 * palm]
    thumb_base = wrist + side * .45 * palm + forward * .3 * palm
    p[1], p[2], p[3] = thumb_base, thumb_base + forward * .3 * palm, thumb_base + forward * .55 * palm
    p[4] = p[8] + side * .05 * palm if pinch else thumb_base + side * .5 * palm + forward * .8 * palm
    return p


def obs(points, label="Right"):
    return {"points": points, "z": np.zeros(21), "label": label}


def sprite(w=40, h=20):
    bgr = np.zeros((h, w, 3), np.uint8)
    bgr[:, : w // 2] = (0, 0, 255)     # left half red, right half green: orientation check
    bgr[:, w // 2:] = (0, 255, 0)
    return Sprite(bgr, np.ones((h, w), np.float32), np.array([w / 2, h / 2]))


class FakeTracker:
    offset = np.zeros(2)
    state = "TRACKING"


def make_object(manip, origin=(200, 200)):
    group = Group(1, np.zeros((480, 640), bool), np.array(origin, float), FakeTracker(), None)
    return manip.add(group, sprite())


class GeometryTests(unittest.TestCase):
    def test_velocity_irregular_timestamps(self):
        history = [(t, 100 + 300 * t, 50 - 120 * t) for t in (0, .03, .045, .08, .1, .13)]
        np.testing.assert_allclose(estimate_velocity(history, .13), [300, -120], atol=1e-6)

    def test_smoothing_frame_rate_independence(self):
        a = b = np.zeros(2)
        for _ in range(10):
            a = smooth_pointer(a, np.array([100.0, 0]), .01, .05)
        for _ in range(5):
            b = smooth_pointer(b, np.array([100.0, 0]), .02, .05)
        np.testing.assert_allclose(a, b, atol=1e-9)

    def test_pinch_ratio_is_scale_invariant(self):
        _, small = update_pinch(make_hand(palm=60), False)
        _, large = update_pinch(make_hand(palm=120), False)
        self.assertAlmostEqual(small, large, places=6)
        self.assertLess(update_pinch(make_hand(pinch=True), False)[1], .30)

    def test_aim_point_is_ahead_of_fingertip_and_clipped(self):
        hand = make_hand(tip=(320, 240), angle=-90)
        aim = aim_point(hand, (480, 640))
        self.assertLess(aim[1], 240 - 10)             # ahead (up) along the finger
        self.assertAlmostEqual(aim[0], 320, delta=1)
        edge = aim_point(make_hand(tip=(320, 5), angle=-90), (480, 640))
        self.assertGreaterEqual(edge[1], 0)

    def test_finger_pose_fist_open_and_pointing(self):
        self.assertEqual(finger_pose(make_hand(fist=True))[1], 4)
        extended, curled, spread, _ = finger_pose(make_hand(spread=True))
        self.assertEqual(extended, 4)
        self.assertGreater(spread, 1.0)
        self.assertLess(finger_pose(make_hand())[1], 4)   # pointing is not a fist


class GestureTests(unittest.TestCase):
    def feed(self, g, ratios, start=0.0, step=1 / 30):
        events = []
        for i, r in enumerate(ratios):
            e = g.update(r, start + i * step)
            if e:
                events.append(e)
        return events

    def test_single_noisy_frame_and_hysteresis_band_never_press(self):
        g = PinchGesture()
        self.assertEqual(self.feed(g, [.6, .6, .2, .6, .6]), [])
        self.assertEqual(self.feed(g, [.2, .38, .2, .38, .2, .38], start=1), [])

    def test_press_requires_arming_and_sustained_close(self):
        g = PinchGesture()
        self.assertEqual(self.feed(g, [.2] * 10), [])              # entered already pinched
        self.assertEqual(self.feed(g, [.6, .6, .2, .2, .2, .2], start=1), ["press"])

    def test_release_requires_sustained_open_and_rearms(self):
        g = PinchGesture()
        self.feed(g, [.6, .2, .2, .2, .2])
        self.assertEqual(self.feed(g, [.5, .2, .5, .2], start=1), [])
        self.assertEqual(self.feed(g, [.5, .5, .5, .5], start=2), ["release"])
        self.assertEqual(self.feed(g, [.2, .2, .2, .2], start=3), ["press"])

    def test_tracking_loss_reports_lost_and_needs_rearm(self):
        g = PinchGesture()
        self.feed(g, [.6, .2, .2, .2, .2])
        self.assertEqual(g.update(None, 1), "lost")
        self.assertEqual(self.feed(g, [.2] * 6, start=2), [])

    def test_hold_gesture_fires_once_with_cooldown(self):
        g = HoldGesture(.5, cooldown=1.0)
        fired = [g.update(True, t / 10) for t in range(12)]
        self.assertEqual(sum(fired), 1)
        g.update(False, 1.2)
        self.assertFalse(any(g.update(True, 1.2 + t / 10) for t in range(4)))  # cooldown


class ManipulationTests(unittest.TestCase):
    def setUp(self):
        self.m = Manipulator(640, 480)
        self.obj = make_object(self.m)

    def run_hand(self, frames, start=0.0, step=1 / 30, label="Right"):
        events, t = [], start
        for points in frames:
            t += step
            events += self.m.update_hands([obs(points, label)] if points is not None else [], t, step)
            for slot, event in events[-2:]:
                if event == "release":
                    self.m.release(slot, t)
            self.m.update_objects(t, step)
        return events, t

    def grab_at(self, point):
        _, t = self.run_hand([make_hand(point)] * 4)
        hand = self.m.hands[0]
        events = []
        for _ in range(4):
            t += 1 / 30
            events += self.m.update_hands([obs(make_hand(point, pinch=True))], t, 1 / 30)
        self.assertIn((0, "press"), events)
        self.m.grab(0, self.obj, t)
        return hand, t

    def test_grab_preserves_offset_no_snap(self):
        hand, t = self.grab_at((230, 215))
        before = self.obj.position.copy()
        self.m.update_objects(t, 1 / 30)
        np.testing.assert_allclose(self.obj.position, before, atol=1e-6)
        offset = self.obj.position - hand.grip
        for i in range(10):
            t += 1 / 30
            self.m.update_hands([obs(make_hand((230 + 5 * i, 215), pinch=True))], t, 1 / 30)
            self.m.update_objects(t, 1 / 30)
        np.testing.assert_allclose(self.obj.position - hand.grip, offset, atol=1e-6)

    def test_stationary_release_floats_and_fast_release_throws(self):
        hand, t = self.grab_at((230, 215))
        for _ in range(8):
            t += 1 / 30
            self.m.update_hands([obs(make_hand((230, 215), pinch=True))], t, 1 / 30)
            self.m.update_objects(t, 1 / 30)
        hand.pinch.release_started = t
        self.assertEqual(self.m.release(0, t), "drop")
        self.assertEqual(self.obj.mode, "floating")
        np.testing.assert_allclose(self.obj.velocity, 0)
        # Fast swipe to the right, then release.
        m2 = Manipulator(640, 480)
        obj = make_object(m2)
        t = 0
        for i in range(4):
            t += 1 / 30
            m2.update_hands([obs(make_hand((200, 215)))], t, 1 / 30)
        for i in range(4):
            t += 1 / 30
            m2.update_hands([obs(make_hand((200, 215), pinch=True))], t, 1 / 30)
        m2.grab(0, obj, t)
        for i in range(8):
            t += 1 / 30
            m2.update_hands([obs(make_hand((200 + 30 * i, 215), pinch=True))], t, 1 / 30)
            m2.update_objects(t, 1 / 30)
        m2.hands[0].pinch.release_started = t
        self.assertEqual(m2.release(0, t), "throw")
        self.assertEqual(obj.mode, "flying")
        self.assertGreater(obj.velocity[0], 500)

    def test_hand_loss_releases_without_throw(self):
        hand, t = self.grab_at((230, 215))
        for i in range(6):
            t += 1 / 30
            self.m.update_hands([obs(make_hand((230 + 40 * i, 215), pinch=True))], t, 1 / 30)
            self.m.update_objects(t, 1 / 30)
        events = []
        for i in range(10):                  # gone for 0.33 s > 0.25 s grace
            t += 1 / 30
            events += self.m.update_hands([], t, 1 / 30)
            self.m.update_objects(t, 1 / 30)
        self.assertIn((0, "lost"), events)
        self.assertEqual(self.obj.mode, "floating")
        np.testing.assert_allclose(self.obj.velocity, 0)
        self.assertEqual(self.m.stats["throws"], 0)

    def test_brief_dropout_keeps_holding_without_jump(self):
        hand, t = self.grab_at((230, 215))
        position = self.obj.position.copy()
        for _ in range(4):                   # 0.13 s landmark dropout
            t += 1 / 30
            self.assertEqual(self.m.update_hands([], t, 1 / 30), [])
            self.m.update_objects(t, 1 / 30)
        np.testing.assert_allclose(self.obj.position, position)
        self.assertEqual(self.obj.mode, "held")
        for i in range(6):                   # reacquired a little further on
            t += 1 / 30
            self.m.update_hands([obs(make_hand((240, 215), pinch=True))], t, 1 / 30)
            self.m.update_objects(t, 1 / 30)
        self.assertEqual(self.obj.mode, "held")
        np.testing.assert_allclose(self.obj.position, position + (10, 0), atol=1)

    def test_two_hand_scale_rotate_clamp_and_handover_without_jump(self):
        m, obj = self.m, self.obj
        t = 0.0
        left, right = make_hand((150, 250)), make_hand((250, 250))
        for _ in range(4):
            t += 1 / 30
            m.update_hands([obs(left, "Left"), obs(right, "Right")], t, 1 / 30)
        for _ in range(4):
            t += 1 / 30
            m.update_hands([obs(make_hand((150, 250), True), "Left"),
                            obs(make_hand((250, 250), True), "Right")], t, 1 / 30)
        slots = sorted(s for s in (0, 1) if m.hands[s].pinch.pinched)
        m.grab(slots[0], obj, t)
        m.grab(slots[1], obj, t)
        m.update_objects(t, 1 / 30)      # same frame as the grab, as in App.step()
        # Spread to 2x separation and rotate the hand line by 30 degrees.
        c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
        a = np.array([200 - 100 * c, 250 - 100 * s])
        b = np.array([200 + 100 * c, 250 + 100 * s])
        for _ in range(40):
            t += 1 / 30
            m.update_hands([obs(make_hand(a, True), "Left"), obs(make_hand(b, True), "Right")], t, 1 / 30)
            m.update_objects(t, 1 / 30)
        self.assertAlmostEqual(obj.scale, 2.0, delta=.08)
        self.assertAlmostEqual(obj.angle, 30, delta=2)
        # Clamp at max scale.
        far_a, far_b = np.array([0.0, 250]), np.array([639.0, 250])
        m.max_scale = 2.5
        for _ in range(40):
            t += 1 / 30
            m.update_hands([obs(make_hand(far_a, True), "Left"), obs(make_hand(far_b, True), "Right")], t, 1 / 30)
            m.update_objects(t, 1 / 30)
        self.assertLessEqual(obj.scale, 2.5 + 1e-6)
        # One hand opens: the other keeps holding with no jump.
        before = obj.position.copy()
        m.release(slots[1], t)
        m.update_objects(t, 1 / 30)
        np.testing.assert_allclose(obj.position, before, atol=1e-6)
        self.assertEqual(obj.mode, "held")

    def test_detection_order_swap_keeps_hand_identity(self):
        m = self.m
        left, right = make_hand((150, 250)), make_hand((450, 250))
        m.update_hands([obs(left, "Left"), obs(right, "Right")], .03, .03)
        wrist0 = m.hands[0].points[0].copy()
        m.update_hands([obs(right, "Right"), obs(left, "Left")], .06, .03)
        np.testing.assert_allclose(m.hands[0].points[0], wrist0)

    def test_hide_show_reset_duplicate(self):
        m, obj = self.m, self.obj
        obj.position, obj.scale, obj.angle, obj.mode = np.array([400.0, 100]), 2.0, 45.0, "floating"
        self.assertEqual(m.toggle_visibility(obj), "hidden")
        for i in range(10):
            m.update_objects(i / 30, 1 / 30)
        self.assertLess(obj.fade, .01)
        self.assertEqual((obj.scale, obj.angle), (2.0, 45.0))     # state preserved while hidden
        m.toggle_visibility(obj)
        copy = m.duplicate(obj)
        self.assertIs(copy.group, obj.group)
        m.reset(obj)
        np.testing.assert_allclose(obj.position, obj.group.origin)
        self.assertEqual((obj.scale, obj.angle, obj.visible, obj.mode), (1.0, 0.0, True, "home"))

    def test_freeze_and_unfreeze(self):
        obj = self.obj
        obj.mode, obj.velocity = "flying", np.array([300.0, 0])
        self.assertEqual(self.m.toggle_freeze(obj), "frozen")
        self.assertEqual(obj.mode, "floating")
        self.assertIn("unfrozen", self.m.toggle_freeze(obj))


class PhysicsAndRenderTests(unittest.TestCase):
    def test_gravity_bounce_and_settle(self):
        obj = ManipulatedObject(1, None, sprite(), position=np.array([320.0, 100]),
                                velocity=np.array([200.0, 0]), mode="flying")
        bounced = False
        for _ in range(600):
            v = obj.velocity[1]
            step_physics(obj, 1 / 60, 640, 480)
            bounced |= v > 0 and obj.velocity[1] < 0
        self.assertTrue(bounced)
        self.assertEqual(obj.mode, "floating")
        self.assertLessEqual(obj.position[1], 480 - obj.half_extent()[1] + 1e-6)

    def test_long_stall_does_not_teleport(self):
        obj = ManipulatedObject(1, None, sprite(), position=np.array([320.0, 100]),
                                velocity=np.array([0.0, 0]), mode="flying")
        step_physics(obj, 5.0, 640, 480)
        self.assertLess(obj.position[1] - 100, 20)

    def test_rotation_and_scale_move_pixels_and_alpha_together(self):
        obj = ManipulatedObject(1, None, sprite(40, 20), position=np.array([100.0, 100]))
        obj.scale, obj.angle = 2.0, 90.0
        out = np.full((200, 200, 3), 255, np.uint8)
        render_sprite(out, obj)
        covered = (out != 255).any(axis=2)
        ys, xs = np.nonzero(covered)
        # 40x20 sprite, x2 scale, rotated 90 deg -> ~40 wide, ~80 tall around (100, 100).
        self.assertAlmostEqual(xs.max() - xs.min() + 1, 40, delta=3)
        self.assertAlmostEqual(ys.max() - ys.min() + 1, 80, delta=3)
        # +90 deg in image coordinates (y down) maps the sprite's left (red) half to the top.
        np.testing.assert_array_equal(out[75, 100], (0, 0, 255))
        np.testing.assert_array_equal(out[125, 100], (0, 255, 0))
        self.assertTrue(obj.contains((100, 70)))
        self.assertFalse(obj.contains((60, 100), margin=2))

    def test_fade_blends_partially(self):
        obj = ManipulatedObject(1, None, sprite(), position=np.array([50.0, 50]))
        obj.fade = .5
        out = np.zeros((100, 100, 3), np.uint8)
        render_sprite(out, obj)
        self.assertAlmostEqual(int(out[50, 40, 2]), 127, delta=2)


if __name__ == "__main__":
    unittest.main()
