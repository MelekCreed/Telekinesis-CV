"""Temporal selection logic, separate from neural inference and drawing."""
import cv2
import numpy as np


class PinchConfirmation:
    """Open dwell arms; continuous close evidence confirms; noise resets dwell."""
    def __init__(self, close=.30, release=.45, dwell=.20):
        self.close, self.release, self.dwell = close, release, dwell
        self.reset()

    def reset(self):
        self.token = None
        self.open_since = self.close_since = self.last_time = None
        self.armed = self.fired = False
        self.state = "OPEN"

    def update(self, ratio, now, token):
        if (token is None or ratio is None or not np.isfinite(ratio)
                or token != self.token or (self.last_time is not None and now - self.last_time > .25)):
            self.reset()
            self.token = token
        self.last_time = now
        if token is None or ratio is None or not np.isfinite(ratio):
            return False
        if ratio >= self.release:
            self.close_since = None
            self.state = "RELEASE_CANDIDATE" if self.fired else "OPEN"
            if self.open_since is None:
                self.open_since = now
            if now - self.open_since >= .12:
                self.armed, self.fired, self.state = True, False, "OPEN"
        else:
            self.open_since = None
            if self.fired:
                self.state = "PINCHED"
            elif self.armed and ratio <= self.close:
                if self.close_since is None:
                    self.close_since = now
                self.state = "PINCH_CANDIDATE"
                if now - self.close_since >= self.dwell:
                    self.fired, self.armed, self.state = True, False, "PINCHED"
                    return True
            else:
                # Crucially, the hysteresis band alone cannot finish a pinch.
                self.close_since = None
                self.state = "OPEN"
        return False


class Selection:
    def __init__(self, pinch=.30, release=.45):
        self.reference = None
        self.reference_id = self.prompt_id = self.candidate_version = 0
        self.result = None
        self.choice = 0
        self.anchor = self.aim_since = None
        self.sent = False
        self.confirmed = False
        self.pending_confirmation = False
        self.pending_since = self.clear_since = None
        self.gesture = PinchConfirmation(pinch, release)
        self.message = "Lower your hand briefly to capture the reference scene"

    @property
    def mask(self):
        return self.result.masks[self.choice] if self.result and self.result.masks else None

    def clear(self):
        self.prompt_id += 1
        self.candidate_version += 1
        self.result = None
        self.choice = 0
        self.anchor = self.aim_since = None
        self.sent = False
        self.confirmed = self.pending_confirmation = False
        self.pending_since = self.clear_since = None
        self.gesture.reset()

    def set_reference(self, frame):
        self.clear()
        self.reference_id += 1
        self.reference = frame.copy()
        self.message = "Scene captured. Hold your fingertip over an object."

    def invalidate(self):
        self.clear()
        self.reference_id += 1
        self.reference = None
        self.message = "Scene changed. Lower your hand and press B to recapture."

    def reset_observation_dwell(self, now):
        """Unobserved camera time cannot count as steady aim or confirmation."""
        self.gesture.reset()
        self.clear_since = None
        if self.anchor is not None and not self.sent:
            self.aim_since = now

    def aim(self, pointer, now):
        if self.reference is None or self.confirmed or self.pending_confirmation:
            return None
        if pointer is None:
            if self.anchor is not None:
                self.clear()
            return None
        point = np.asarray(pointer, dtype=float)
        height, width = self.reference.shape[:2]
        if not np.isfinite(point).all() or not (0 <= point[0] < width and 0 <= point[1] < height):
            self.clear()
            return None
        ix, iy = int(point[0]), int(point[1])
        # Once an object is previewed, moving within its silhouette (plus a
        # forgiving margin) keeps the same target while the fingers close.
        on_preview = self.mask is not None and self.mask[
            max(0, iy - 22):min(height, iy + 23), max(0, ix - 22):min(width, ix + 23)].any()
        if self.anchor is None or (np.linalg.norm(point - self.anchor) > 22 and not on_preview):
            self.clear()
            self.anchor, self.aim_since = point.copy(), now
        if not self.sent and now - self.aim_since >= .35:
            self.sent = True
            self.message = "Segmenting the reference image; live camera continues"
            return tuple(self.anchor)
        return None

    def accept(self, result):
        if result.reference_id != self.reference_id or result.prompt_id != self.prompt_id:
            return False
        self.result, self.choice = result, 0
        self.candidate_version += 1
        self.gesture.reset()
        self.message = ("Preview ready. Open fingers, then hold a pinch to confirm."
                        if result.masks else "No usable silhouette. Aim elsewhere, or press R to retry.")
        return True

    def cycle(self):
        if self.result and self.result.masks and not self.confirmed:
            self.choice = (self.choice + 1) % len(self.result.masks)
            self.candidate_version += 1
            self.pending_confirmation = False
            self.gesture.reset()
            self.message = "Alternate mask. Open fingers, then pinch to confirm."

    def update_confirmation(self, ratio, now, visible_fraction, target_changed):
        if self.mask is None or self.confirmed:
            return False
        token = (self.reference_id, self.prompt_id, self.candidate_version)
        if not self.pending_confirmation and self.gesture.update(ratio, now, token):
            self.pending_confirmation = True
            self.pending_since = now
            self.clear_since = None
        if self.pending_confirmation:
            self.message = "Pinch received. Move your hand aside to verify the silhouette."
            if now - self.pending_since > 6:
                self.pending_confirmation = False
                self.gesture.reset()
                self.message = "Confirmation timed out. Open fingers and try again."
            elif visible_fraction >= .65 and not target_changed:
                if self.clear_since is None:
                    self.clear_since = now
                if now - self.clear_since >= .25:
                    self.confirmed, self.pending_confirmation = True, False
                    self.message = "MASK CONFIRMED (static reference). R selects another object."
                    return True
            else:
                self.clear_since = None
        return False


def hand_exclusion(shape, points):
    excluded = np.zeros(shape, np.uint8)
    if points is not None and np.isfinite(points).all():
        hull = cv2.convexHull(np.rint(points).astype(np.int32))
        cv2.fillConvexPoly(excluded, hull, 255)
        size = int(np.clip(np.linalg.norm(points[0] - points[9]) * .7, 25, 100))
        excluded = cv2.dilate(excluded, np.ones((size | 1, size | 1), np.uint8))
        # Wrist continuation excludes an approximate forearm, not a person mask.
        wrist = points[0]
        direction = wrist - points[9]
        cv2.line(excluded, tuple(wrist.astype(int)), tuple((wrist + direction * 5).astype(int)),
                 255, max(35, size))
    return excluded.astype(bool)


def compare_scene(reference, current, excluded, target=None):
    """Simple static-scene validation, NOT tracking or camera registration."""
    old = cv2.GaussianBlur(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), (7, 7), 0).astype(np.float32)
    new = cv2.GaussianBlur(cv2.cvtColor(current, cv2.COLOR_BGR2GRAY), (7, 7), 0).astype(np.float32)
    visible = ~excluded
    if visible.mean() < .35:
        return True, 0.0, True
    delta = new - old
    # Modest uniform lighting drift is tolerated; it is not object motion.
    offset = float(np.clip(np.median(delta[visible]), -15, 15))
    changed = np.abs(delta - offset) > 24
    scene_changed = float(changed[visible].mean()) > .18
    visible_fraction, target_changed = 1.0, False
    if target is not None:
        region = cv2.dilate(target.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        exposed = region & visible
        visible_fraction = float(exposed.sum() / max(region.sum(), 1))
        if exposed.any():
            target_changed = float(changed[exposed].mean()) > .20
    return scene_changed, visible_fraction, target_changed
