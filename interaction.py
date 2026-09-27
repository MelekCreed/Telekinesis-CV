"""Landmark geometry -> hand state -> grab/throw and optional advanced gestures."""
from collections import deque
from dataclasses import dataclass, field
from itertools import permutations
import math

import numpy as np

from hand_tracker import smooth_pointer, update_pinch, palm_geometry, estimate_velocity
from physics import hit_test, limit_speed, constrain


@dataclass
class Hand:
    points: object = None
    z: object = None
    label: str = ""
    pointer: object = None
    palm: object = None
    palm_size: float = 0.0
    ratio: object = None
    pinched: bool = False
    armed: bool = False
    pressed: bool = False
    held: object = None
    offset: np.ndarray = field(default_factory=lambda: np.zeros(2))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    history: deque = field(default_factory=lambda: deque(maxlen=30))
    palm_history: deque = field(default_factory=lambda: deque(maxlen=30))
    size_history: deque = field(default_factory=lambda: deque(maxlen=30))
    open_since: object = None
    fist_since: object = None
    fist_target: object = None
    fist_fired: bool = False
    cooldown: float = 0.0
    depth_size: float = 1.0
    depth_radius: float = 1.0


class Interaction:
    def __init__(self, objects, width, height, smoothing=0.035, pinch=0.30,
                 release=0.45, throw_gain=1.25):
        self.objects, self.width, self.height = objects, width, height
        self.smoothing, self.pinch_threshold, self.release_threshold = smoothing, pinch, release
        self.throw_gain = throw_gain
        self.hands = [Hand(), Hand()]
        self.dual = {}
        self.advanced = False
        self.depth = False
        self.effects = []  # Small ring bursts: [x, y, start time, BGR color].
        self.message = "Point at a crystal, then pinch to grab"
        self.message_until = 6.0
        self.stats = dict(grabs=0, throws=0, tracking_drops=0, transforms=0, pushes=0, freezes=0)

    def event(self, message, position, now, color=(130, 255, 255)):
        self.message, self.message_until = message, now + 1.5
        self.effects.append((position.copy(), now, color))
        self.effects = self.effects[-16:]

    def release_hand(self, slot, now, velocity=None):
        hand = self.hands[slot]
        if hand.held is None:
            return
        index = hand.held
        obj = self.objects[index]
        obj.holders.discard(slot)
        hand.held = None
        self.dual.pop(index, None)
        if obj.holders:
            # Going from two hands to one must not snap to the original offset.
            other = self.hands[next(iter(obj.holders))]
            other.offset = obj.position - other.pointer
            other.depth_size, other.depth_radius = other.palm_size, obj.radius
        else:
            obj.velocity = (np.zeros(2) if velocity is None else
                            limit_speed(velocity * self.throw_gain))
            if np.linalg.norm(obj.velocity) < 45:
                obj.velocity[:] = 0  # Stationary landmark noise should not throw.
            obj.spin = float(obj.velocity[0]) / 260
            self.stats["tracking_drops" if velocity is None else "throws"] += 1
            self.event("Tracking lost - released safely" if velocity is None else "Released",
                       obj.position, now, obj.color)

    def match_hands(self, observations):
        # MediaPipe result ordering can change. Find the lowest-cost assignment
        # to the two previous wrists instead of using the detection list index.
        observations = observations[:2]
        if not observations:
            return {}

        def cost(assignment):
            total = 0.0
            for detection, slot in zip(observations, assignment):
                hand = self.hands[slot]
                total += (180 if hand.points is None else
                          np.linalg.norm(detection["points"][0] - hand.points[0]))
                if hand.points is not None and hand.label != detection["label"]:
                    total += 55  # Soft evidence; handedness alone can flicker.
            return total

        slots = min(permutations(range(2), len(observations)), key=cost)
        return dict(zip(slots, observations))

    def update(self, observations, now, dt):
        # Treat corrupt/nonfinite predictions like lost tracking, never as motion.
        observations = [o for o in observations
                        if o["points"].shape == (21, 2) and np.isfinite(o["points"]).all()]
        matches = self.match_hands(observations)
        for slot, hand in enumerate(self.hands):
            observed = matches.get(slot)
            points = observed["points"] if observed else None
            discontinuity = (dt > 0.25 or (points is not None and hand.points is not None
                             and np.linalg.norm(points[0] - hand.points[0]) > self.width * 0.45))
            if points is None or discontinuity:
                self.release_hand(slot, now)
                self.hands[slot] = hand = Hand()
                if points is None:
                    continue
            old_pinched = hand.pinched
            hand.points, hand.z, hand.label = points, observed["z"], observed["label"]
            hand.pointer = smooth_pointer(hand.pointer, points[8], dt, self.smoothing)
            hand.palm, hand.palm_size, palm_open, fist = palm_geometry(points)
            hand.pinched, hand.ratio = update_pinch(
                points, old_pinched, self.pinch_threshold, self.release_threshold)
            hand.pressed = hand.pinched and not old_pinched and hand.armed
            # Do not include the opening frame in throw velocity: extending the
            # index to release should not produce a fake final flick/impulse.
            if old_pinched and not hand.pinched:
                velocity = estimate_velocity(hand.history, now)
                self.release_hand(slot, now, velocity if hand.ratio is not None else None)
            if not hand.pinched and hand.ratio is not None:
                hand.armed = True
            hand.history.append((now, *hand.pointer))
            hand.palm_history.append((now, *hand.palm))
            hand.size_history.append((now, math.log(max(hand.palm_size, 1e-6)), 0))
            hand.velocity = estimate_velocity(hand.history, now)

            if hand.pressed and hand.held is None:
                target = hit_test(self.objects, hand.pointer)
                if target is not None:
                    obj = self.objects[target]
                    hand.held = target
                    hand.offset = obj.position - hand.pointer
                    hand.depth_size, hand.depth_radius = hand.palm_size, obj.radius
                    obj.holders.add(slot)
                    obj.frozen = False
                    obj.velocity[:] = 0
                    obj.spin = 0
                    obj.trail.clear()
                    self.stats["grabs"] += 1
                    self.event("Grabbed - move, then open to throw", obj.position, now, obj.color)

            if self.advanced and not hand.pinched and hand.held is None:
                self.advanced_gestures(hand, palm_open, fist, now)
            else:
                hand.fist_since = hand.open_since = None
                hand.fist_fired = False

        for index, obj in enumerate(self.objects):
            if not obj.holders:
                continue
            old_position = obj.position.copy()
            if len(obj.holders) == 2:
                first, second = [self.hands[i] for i in sorted(obj.holders)]
                line = second.pointer - first.pointer
                distance = float(np.linalg.norm(line))
                midpoint = (first.pointer + second.pointer) / 2
                angle = math.atan2(line[1], line[0])
                if index not in self.dual and distance >= 20:
                    self.dual[index] = (distance, angle, obj.radius, obj.angle,
                                        obj.position - midpoint)
                    self.stats["transforms"] += 1
                    self.event("Two hands: spread to scale, twist to rotate", obj.position, now)
                if index in self.dual:
                    base_distance, base_angle, radius, orientation, offset = self.dual[index]
                    scale = float(np.clip(distance / base_distance, 16 / radius,
                                          min(95, self.height / 4) / radius))
                    rotation = angle - base_angle
                    c, s = math.cos(rotation), math.sin(rotation)
                    # Rotate/scale the original midpoint offset so joining a
                    # second hand does not teleport the object's center.
                    matrix = np.array([[c, -s], [s, c]])
                    obj.position = midpoint + matrix @ offset * scale
                    obj.radius, obj.angle = radius * scale, orientation + rotation
            else:
                self.dual.pop(index, None)
                hand = self.hands[next(iter(obj.holders))]
                obj.position = hand.pointer + hand.offset
                if self.depth:
                    # Apparent palm scale is a relative depth cue, not meters.
                    ratio = hand.palm_size / max(hand.depth_size, 1e-6)
                    obj.radius = float(np.clip(hand.depth_radius * ratio, 16,
                                               min(95, self.height / 4)))
            constrain(obj, self.width, self.height)
            obj.velocity = limit_speed((obj.position - old_position) / max(dt, 1e-3))
        self.effects = [effect for effect in self.effects if now - effect[1] < 0.45]

    def advanced_gestures(self, hand, palm_open, fist, now):
        if palm_open:
            if hand.open_since is None:
                hand.open_since = now
            velocity = estimate_velocity(hand.palm_history, now)
            approach_rate = estimate_velocity(hand.size_history, now)[0]
            if (now - hand.open_since > 0.10 and now > hand.cooldown
                    and (np.linalg.norm(velocity) > 800 or approach_rate > 1.8)):
                affected = 0
                for obj in self.objects:
                    delta = obj.position - hand.palm
                    distance = np.linalg.norm(delta)
                    if distance < 220 and not obj.holders and not obj.frozen:
                        normal = delta / distance if distance > 1 else np.array([0., -1.])
                        strength = 1050 * (1 - distance / 260)
                        obj.velocity = limit_speed(obj.velocity + normal * strength + velocity * 0.35)
                        affected += 1
                if affected:
                    self.stats["pushes"] += 1
                    self.event("Force push", hand.palm, now)
                    hand.cooldown = now + 0.7
        else:
            hand.open_since = None

        if fist:
            candidates = [(np.linalg.norm(obj.position - hand.palm), i)
                          for i, obj in enumerate(self.objects) if not obj.holders]
            nearest = min(candidates) if candidates else (float("inf"), None)
            target = nearest[1] if nearest[0] < 130 else None
            if target != hand.fist_target or hand.fist_since is None:
                hand.fist_target, hand.fist_since = target, now
            if target is not None and not hand.fist_fired and now - hand.fist_since >= 0.4:
                obj = self.objects[target]
                obj.frozen = not obj.frozen
                obj.velocity[:] = 0
                obj.spin = 0
                hand.fist_fired = True
                self.stats["freezes"] += 1
                self.event("Frozen" if obj.frozen else "Unfrozen", obj.position, now, obj.color)
        else:
            hand.fist_since = hand.fist_target = None
            hand.fist_fired = False

    def toggle_depth(self):
        self.depth = not self.depth
        for hand in self.hands:
            if hand.held is not None:
                hand.depth_size = hand.palm_size
                hand.depth_radius = self.objects[hand.held].radius
