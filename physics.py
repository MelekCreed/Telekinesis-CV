"""Small 2D physics: pixels, seconds, circular collision bounds. No engine."""
from collections import deque
from dataclasses import dataclass, field
import math

import numpy as np


@dataclass
class Object:
    position: np.ndarray
    radius: float
    color: tuple
    sides: int
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    angle: float = 0.0
    spin: float = 0.0
    frozen: bool = False
    holders: set = field(default_factory=set)
    trail: deque = field(default_factory=lambda: deque(maxlen=12))

    @property
    def inverse_mass(self):
        # A held/frozen object is kinematic: collisions cannot knock it away.
        return 0.0 if self.holders or self.frozen else 1.0 / (self.radius ** 2)


def spawn_objects(width, height):
    return [Object(np.array([width * x, height * 0.32], dtype=float), radius,
                   color, sides, angle=angle)
            for x, radius, color, sides, angle in (
                (0.19, 32, (255, 220, 70), 4, math.pi / 4),
                (0.40, 38, (180, 100, 255), 6, 0),
                (0.62, 30, (70, 220, 255), 3, -math.pi / 2),
                (0.81, 35, (170, 255, 90), 5, 0))]


def hit_test(objects, pointer):
    # Radial distance is the circle hit test; choose the nearest overlapping one.
    candidates = [(float(np.linalg.norm(obj.position - pointer)), i)
                  for i, obj in enumerate(objects)
                  if np.linalg.norm(obj.position - pointer) <= obj.radius + 8]
    return min(candidates)[1] if candidates else None


def limit_speed(velocity, maximum=1800.0):
    speed = np.linalg.norm(velocity)
    return velocity * min(1.0, maximum / max(speed, 1e-9))


def constrain(obj, width, height, restitution=0.72):
    for axis, extent in enumerate((width, height)):
        lo, hi = obj.radius, extent - obj.radius
        if obj.position[axis] < lo:
            obj.position[axis] = lo
            if obj.velocity[axis] < 0:
                obj.velocity[axis] *= -restitution
        elif obj.position[axis] > hi:
            obj.position[axis] = hi
            if obj.velocity[axis] > 0:
                obj.velocity[axis] *= -restitution


def collide(a, b, restitution=0.72):
    delta = b.position - a.position
    distance = float(np.linalg.norm(delta))
    overlap = a.radius + b.radius - distance
    inv_a, inv_b = a.inverse_mass, b.inverse_mass
    total = inv_a + inv_b
    if overlap <= 0 or total == 0:
        return
    normal = delta / distance if distance > 1e-8 else np.array([1.0, 0.0])
    # Correct penetration in proportion to inverse mass (lighter objects move more).
    a.position -= normal * overlap * inv_a / total
    b.position += normal * overlap * inv_b / total
    closing_speed = float(np.dot(b.velocity - a.velocity, normal))
    if closing_speed < 0:
        # Impulse reverses only the approaching normal component; tangential stays.
        impulse = -(1 + restitution) * closing_speed / total
        a.velocity -= impulse * inv_a * normal
        b.velocity += impulse * inv_b * normal


def step_physics(objects, dt, width, height, gravity=650.0, restitution=0.72,
                 damping=0.18, floor_friction=4.0):
    # Fixed small substeps reduce tunneling. Cap long stalls to avoid simulation
    # explosions; real-time interaction deliberately discards time beyond 100 ms.
    remaining = min(max(dt, 0.0), 0.10)
    while remaining > 1e-9:
        step = min(remaining, 1 / 240)
        remaining -= step
        for obj in objects:
            if obj.holders or obj.frozen:
                constrain(obj, width, height, restitution)
                continue
            obj.velocity = limit_speed(obj.velocity)
            obj.velocity[1] += gravity * step
            obj.velocity *= math.exp(-damping * step)
            obj.position += obj.velocity * step
            obj.angle += obj.spin * step
            obj.spin *= math.exp(-1.5 * step)
            constrain(obj, width, height, restitution)
            if obj.position[1] >= height - obj.radius - 0.5:
                # Small bounces stop; horizontal drag is per-second, not per-frame.
                if abs(obj.velocity[1]) < 35:
                    obj.velocity[1] = 0
                obj.velocity[0] *= math.exp(-floor_friction * step)
        # A few projection passes handle short stacks without a complex solver.
        for _ in range(3):
            for i, a in enumerate(objects):
                for b in objects[i + 1:]:
                    collide(a, b, restitution)
            for obj in objects:
                constrain(obj, width, height, restitution)
    for obj in objects:
        obj.trail.append(obj.position.copy())
