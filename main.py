"""Telekinesis CV: camera -> landmarks -> gestures -> objects -> physics -> frame."""
import argparse
from collections import deque
import math
import sys
import time

import cv2
import numpy as np

from hand_tracker import HandTracker, draw_landmarks, draw_pointer, draw_pinch_indicator
from interaction import Interaction
from physics import spawn_objects, step_physics, hit_test

WINDOW = "Telekinesis CV | Virtual crystals"


def create_window(width, height):
    # Enlarge only the display: inference still runs on the original camera
    # frame, so a bigger window doesn't add landmark-model work.
    scale = 2.0
    if sys.platform == "win32":
        import ctypes
        screen = ctypes.windll.user32
        scale = min(scale, screen.GetSystemMetrics(0) * 0.90 / width,
                    screen.GetSystemMetrics(1) * 0.85 / height)
    size = (int(width * scale), int(height * scale))
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW, *size)
    return size


def open_camera(index, backend):
    choices = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}
    backends = ([choices[backend]] if backend != "auto" else
                [cv2.CAP_DSHOW, cv2.CAP_MSMF] if sys.platform == "win32" else
                [cv2.CAP_ANY])
    for api in backends:
        camera = cv2.VideoCapture(index, api)
        if camera.isOpened():
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            for _ in range(10):
                ok, frame = camera.read()
                if ok and frame is not None and frame.size:
                    print(f"Camera {index}: {camera.getBackendName()}, "
                          f"{frame.shape[1]}x{frame.shape[0]}", flush=True)
                    return camera, frame
        camera.release()
    raise RuntimeError(
        f"Cannot read webcam {index}. Close other camera apps, check the shutter "
        "and Windows Settings > Privacy & security > Camera > Let desktop apps "
        "access your camera. Try --camera 1 or --backend msmf."
    )


def text(frame, message, y, color=(235, 235, 235), x=12, scale=0.46):
    for thickness, shade in ((3, (0, 0, 0)), (1, color)):
        cv2.putText(frame, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, shade, thickness, cv2.LINE_AA)


def pixel(point):
    return tuple(np.rint(point).astype(int))


def draw_scene(frame, game, now, fps, debug=False, landmarks=True, inference_ms=0, dt=0):
    targets = {hit_test(game.objects, h.pointer) for h in game.hands if h.pointer is not None}
    cv2.line(frame, (0, game.height), (game.width - 1, game.height), (120, 130, 140), 1)
    for i, obj in enumerate(game.objects):
        color = obj.color
        if np.linalg.norm(obj.velocity) > 100:
            trail = list(obj.trail)
            for j in range(1, len(trail)):
                shade = tuple(int(c * j / len(trail) * 0.65) for c in color)
                cv2.line(frame, pixel(trail[j - 1]), pixel(trail[j]), shade, 2, cv2.LINE_AA)
        center = pixel(obj.position)
        if obj.holders or i in targets:
            cv2.circle(frame, center, int(obj.radius + 7), color,
                       3 if obj.holders else 1, cv2.LINE_AA)
        # Polygon crystals use circular collision bounds. Facets and a bright
        # vertex make two-hand rotation visible even on symmetric shapes.
        angles = obj.angle + np.arange(obj.sides) * (2 * math.pi / obj.sides)
        vertices = np.rint(obj.position + obj.radius * np.column_stack(
            (np.cos(angles), np.sin(angles)))).astype(np.int32)
        cv2.fillConvexPoly(frame, vertices, tuple(int(c * 0.18) for c in color), cv2.LINE_AA)
        cv2.polylines(frame, [vertices], True, color, 2, cv2.LINE_AA)
        for vertex in vertices:
            cv2.line(frame, center, tuple(vertex), tuple(int(c * 0.55) for c in color), 1, cv2.LINE_AA)
        cv2.circle(frame, tuple(vertices[0]), 4, (250, 250, 250), -1, cv2.LINE_AA)
        if obj.frozen:
            text(frame, "LOCK", center[1] + 5, x=center[0] - 18, scale=0.40)
        if debug:
            cv2.circle(frame, center, int(obj.radius), (130, 130, 130), 1)
            text(frame, f"{i + 1}: {np.linalg.norm(obj.velocity):.0f}px/s",
                 center[1] - int(obj.radius) - 10, x=max(2, center[0] - 35), scale=0.35)

    for slot, hand in enumerate(game.hands):
        if hand.points is None:
            continue
        if landmarks:
            draw_landmarks(frame, hand.points, debug)
        draw_pointer(frame, hand.pointer, hand.points[8], debug)
        state = ("PINCHED" if hand.pinched else "APPROACHING"
                 if hand.ratio is not None and hand.ratio < game.release_threshold + 0.20 else "OPEN")
        draw_pinch_indicator(frame, (22 + 180 * slot, 49), state)
        ratio = f"{hand.ratio:.2f}" if hand.ratio is not None else "--"
        text(frame, f"{state} {ratio}", 54, x=38 + 180 * slot, scale=0.40)
        if hand.held is not None:
            obj = game.objects[hand.held]
            cv2.line(frame, pixel(hand.pointer), pixel(obj.position), obj.color, 2, cv2.LINE_AA)
        if debug:
            end = hand.pointer + hand.velocity * 0.10
            cv2.arrowedLine(frame, pixel(hand.pointer), pixel(end), (90, 200, 255), 2,
                           cv2.LINE_AA, tipLength=0.2)
            y = 130 + slot * 66
            text(frame, f"H{slot + 1} index {hand.points[8].round(1)} thumb {hand.points[4].round(1)}",
                 y, scale=0.38)
            text(frame, f"v {hand.velocity.round(0)} px/s | palm {hand.palm_size:.0f}px | "
                 f"tip z {hand.z[8]:+.3f} (wrist-relative)", y + 19, scale=0.38)
    for position, started, color in game.effects:
        age = (now - started) / 0.45
        cv2.circle(frame, pixel(position), int(12 + 40 * age),
                   tuple(int(c * (1 - age)) for c in color), 2, cv2.LINE_AA)
    text(frame, f"TELEKINESIS CV   {fps:.0f} FPS", 24, scale=0.55)
    if not any(hand.points is not None for hand in game.hands):
        text(frame, "Raise a hand. Aim at a crystal, then pinch.", 54)
    if now < game.message_until:
        text(frame, game.message, 78, color=(130, 250, 255), scale=0.43)
    elif not debug:
        text(frame, "Pinch + move + release to throw | Two pinches: resize / rotate", 78, scale=0.40)
    if debug:
        text(frame, f"dt {dt * 1000:.1f}ms | infer {inference_ms:.1f}ms | "
             f"smoothing {game.smoothing * 1000:.0f}ms", 96, scale=0.40)
        text(frame, f"Pinch <= {game.pinch_threshold:.2f} | release >= {game.release_threshold:.2f}",
             113, scale=0.40)
    text(frame, f"R reset  G gravity  A palm/fist {'ON' if game.advanced else 'OFF'}  "
         f"Z depth cue {'ON' if game.depth else 'OFF'}", frame.shape[0] - 34, scale=0.40)
    text(frame, "F fullscreen  D debug  L skeleton  [ / ] smoothing  Q quit", frame.shape[0] - 13, scale=0.40)


def run(args):
    tracker = camera = None
    count = detected = 0
    inference_total = capture_total = 0.0
    timestamps = deque(maxlen=31)
    debug, landmarks = args.debug, True
    fullscreen = False
    started = None
    game = None
    next_report = 5.0
    gravity = args.gravity
    try:
        tracker = HandTracker(num_hands=args.hands)
        camera, frame = open_camera(args.camera, args.backend)
        height, width = frame.shape[:2]
        # Keep the virtual floor high enough to aim a finger at resting crystals
        # while the wrist is still visible to the camera.
        arena_height = height - 110
        objects = spawn_objects(width, arena_height)[:args.objects]
        game = Interaction(objects, width, arena_height, args.smoothing_ms / 1000,
                           args.pinch_threshold, args.release_threshold, args.throw_gain)
        game.advanced, game.depth = args.advanced, args.depth
        if not args.headless:
            window_size = create_window(width, height)
        started = previous = time.perf_counter()
        while True:
            captured = time.perf_counter()
            dt, now = captured - previous, captured - started
            previous = captured
            # Mirror first: model and interaction share displayed coordinates.
            frame = cv2.flip(frame, 1)
            observations = tracker.detect_all(frame, captured)
            inference_ms = (time.perf_counter() - captured) * 1000
            inference_total += inference_ms
            detected += bool(observations)
            game.update(observations, now, dt)
            step_physics(objects, dt, width, arena_height, gravity,
                         args.restitution, args.damping, args.friction)
            timestamps.append(time.perf_counter())
            fps = ((len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
                   if len(timestamps) > 1 else 0.0)
            draw_scene(frame, game, now, fps, debug, landmarks, inference_ms, dt)
            count += 1
            if now >= next_report:
                print(f"Live: {fps:.1f} FPS; {len(observations)} hands; "
                      f"interactions {game.stats}", flush=True)
                next_report = now + 5
            if not args.headless:
                cv2.imshow(WINDOW, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if key in (ord("f"), ord("F")):
                    fullscreen = not fullscreen
                    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                    if not fullscreen:
                        cv2.resizeWindow(WINDOW, *window_size)
                if key in (ord("d"), ord("D")):
                    debug = not debug
                if key in (ord("l"), ord("L")):
                    landmarks = not landmarks
                if key == ord("["):
                    game.smoothing = max(0, round(game.smoothing - 0.005, 3))
                if key == ord("]"):
                    game.smoothing = min(0.15, round(game.smoothing + 0.005, 3))
                if key in (ord("g"), ord("G")):
                    gravity = 0 if gravity else args.gravity
                    game.event(f"Gravity {'ON' if gravity else 'OFF'}", np.array([width / 2, 90]), now)
                if key in (ord("a"), ord("A")):
                    game.advanced = not game.advanced
                    game.event("Palm push / fist freeze " + ("ON" if game.advanced else "OFF"),
                               np.array([width / 2, 90]), now)
                if key in (ord("z"), ord("Z")):
                    game.toggle_depth()
                    game.event("Relative depth cue " + ("ON" if game.depth else "OFF"),
                               np.array([width / 2, 90]), now)
                if key in (ord("r"), ord("R")):
                    advanced, depth, smoothing = game.advanced, game.depth, game.smoothing
                    objects = spawn_objects(width, arena_height)[:args.objects]
                    game = Interaction(objects, width, arena_height, smoothing,
                                       args.pinch_threshold, args.release_threshold, args.throw_gain)
                    game.advanced, game.depth = advanced, depth
                    game.event("Crystals reset", np.array([width / 2, 90]), now)
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if args.seconds and time.perf_counter() - started >= args.seconds:
                break
            read_started = time.perf_counter()
            ok, frame = camera.read()
            capture_total += (time.perf_counter() - read_started) * 1000
            if not ok or frame is None or not frame.size:
                raise RuntimeError("Webcam stopped delivering frames. Reconnect it and restart.")
    finally:
        finished = time.perf_counter()
        if camera is not None:
            camera.release()
        if tracker is not None:
            tracker.close()
        cv2.destroyAllWindows()
        if started is not None and count:
            elapsed = finished - started
            print(f"Run: {count} frames / {elapsed:.2f}s = {count / elapsed:.1f} FPS; "
                  f"hand detected in {detected} frames.", flush=True)
            print(f"Mean inference: {inference_total / count:.1f} ms; "
                  f"mean camera read: {capture_total / max(count - 1, 1):.1f} ms.", flush=True)
            print(f"Interactions since last reset: {game.stats}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--backend", choices=("auto", "dshow", "msmf"), default="auto")
    parser.add_argument("--hands", type=int, choices=(1, 2), default=2)
    parser.add_argument("--objects", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--advanced", action="store_true", help="Enable palm push / fist freeze")
    parser.add_argument("--depth", action="store_true", help="Experimental apparent-size depth cue")
    parser.add_argument("--smoothing-ms", type=float, default=35.0)
    parser.add_argument("--pinch-threshold", type=float, default=0.30)
    parser.add_argument("--release-threshold", type=float, default=0.45)
    parser.add_argument("--throw-gain", type=float, default=1.25)
    parser.add_argument("--gravity", type=float, default=650.0, help="Pixels/s^2")
    parser.add_argument("--restitution", type=float, default=0.72)
    parser.add_argument("--damping", type=float, default=0.18, help="Air drag per second")
    parser.add_argument("--friction", type=float, default=4.0, help="Floor drag per second")
    parser.add_argument("--seconds", type=float, default=0, help="Optional timed run")
    parser.add_argument("--headless", action="store_true", help="Camera/inference smoke test")
    args = parser.parse_args()
    numeric = (args.smoothing_ms, args.pinch_threshold, args.release_threshold,
               args.throw_gain, args.gravity, args.restitution, args.damping, args.friction, args.seconds)
    if not all(math.isfinite(v) for v in numeric):
        parser.error("All numeric settings must be finite.")
    if not 0 < args.pinch_threshold < args.release_threshold:
        parser.error("Thresholds must satisfy 0 < pinch < release.")
    if not 0 <= args.smoothing_ms <= 150:
        parser.error("--smoothing-ms must be between 0 and 150.")
    if not 0 <= args.restitution <= 1 or min(args.throw_gain, args.gravity, args.damping, args.friction) < 0:
        parser.error("Restitution must be 0-1; physics settings must be nonnegative.")
    if args.seconds < 0 or (args.headless and args.seconds <= 0):
        parser.error("Use a positive --seconds duration for --headless.")
    return args


if __name__ == "__main__":
    try:
        run(parse_args())
    except KeyboardInterrupt:
        pass
    except (RuntimeError, OSError, ValueError, cv2.error) as error:
        print(f"Telekinesis CV: {error}", file=sys.stderr)
        sys.exit(1)
