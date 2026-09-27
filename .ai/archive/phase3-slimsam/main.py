"""Reality Manipulation milestone 1: touch point -> real-object silhouette."""
import argparse
from collections import deque
import math
import sys
import time

import cv2
import numpy as np

from hand_tracker import HandTracker, draw_landmarks, draw_pointer, smooth_pointer, update_pinch
from segmentation import MODEL_DIR, SegmentationWorker
from selection import Selection, hand_exclusion, compare_scene

WINDOW = "Telekinesis CV | Real-object selection"


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


def draw_selection(frame, selection, pointer, points, ratio, fps, worker, debug, now):
    mask = selection.mask
    if mask is not None:
        color = (100, 255, 130) if selection.confirmed else (255, 210, 80)
        # Binary mask: 1 selects object pixels, 0 leaves camera pixels untouched.
        # The 0.18 tint is only a preview overlay, not extraction or removal.
        frame[mask] = (frame[mask] * .82 + np.array(color) * .18).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, color, 2 if selection.confirmed else 1, cv2.LINE_AA)
    if points is not None:
        if debug:
            draw_landmarks(frame, points, True)
        draw_pointer(frame, pointer, points[8], debug)
    if selection.anchor is not None:
        cv2.drawMarker(frame, pixel(selection.anchor), (80, 180, 255),
                       cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)
    text(frame, f"REALITY MANIPULATION   {fps:.0f} FPS", 24, scale=.52)
    text(frame, selection.message, 48, color=(150, 240, 250), scale=.40)
    text(frame, "Keep camera and objects still | This milestone selects silhouettes", 70, scale=.38)
    if worker.error:
        text(frame, "Segmentation error; see console. R retries.", 92, color=(80, 100, 255))
    elif worker.busy:
        text(frame, "Preparing scene / segmentation in background...", 92, scale=.40)
    if debug:
        ratio_text = "--" if ratio is None else f"{ratio:.2f}"
        text(frame, f"Pinch {ratio_text} | {selection.gesture.state} | "
             f"reference {selection.reference_id} prompt {selection.prompt_id}", 115, scale=.38)
        if selection.result and selection.result.masks:
            result = selection.result
            text(frame, f"Mask {selection.choice + 1}/{len(result.masks)} | estimated IoU "
                 f"{result.scores[selection.choice]:.2f} | encoder {result.encoder_ms:.0f}ms "
                 f"decoder {result.decoder_ms:.0f}ms", 134, scale=.36)
        if selection.reference is not None:
            thumb = cv2.resize(selection.reference, (160, 120))
            frame[-164:-44, 12:172] = thumb
            text(frame, "REFERENCE (not live)", frame.shape[0] - 169, x=12, scale=.32)
        if mask is not None:
            thumb = cv2.resize(mask.astype(np.uint8) * 255, (160, 120), interpolation=cv2.INTER_NEAREST)
            frame[-164:-44, 184:344] = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
            text(frame, "BINARY MASK", frame.shape[0] - 169, x=184, scale=.32)
    text(frame, "B recapture scene  M alternate mask  R / Esc clear selection", frame.shape[0] - 27, scale=.38)
    text(frame, "F fullscreen  D debug  [ / ] smoothing  Q quit", frame.shape[0] - 9, scale=.38)


def run(args):
    tracker = camera = worker = None
    selection = Selection(args.pinch_threshold, args.release_threshold)
    pointer = None
    started = None
    count = detected = confirmed_count = 0
    timestamps = deque(maxlen=31)
    capture_requested = True
    stable_since = changed_since = None
    previous_frame = None
    debug, fullscreen = args.debug, False
    next_report = 5.0
    smoothing = args.smoothing_ms / 1000
    logged_error = None
    try:
        tracker = HandTracker(num_hands=1)
        camera, frame = open_camera(args.camera, args.backend)
        height, width = frame.shape[:2]
        if height < 240 or width < 360:
            raise RuntimeError("Camera must deliver at least 360x240 pixels.")
        worker = SegmentationWorker(args.model_dir, args.threads)
        if not args.headless:
            window_size = create_window(width, height)
        started = previous = time.perf_counter()
        while True:
            captured = time.perf_counter()
            dt, now = captured - previous, captured - started
            previous = captured
            if dt > .25:
                selection.reset_observation_dwell(now)
                stable_since = changed_since = None
            frame = cv2.flip(frame, 1)
            observations = tracker.detect_all(frame, captured)
            points = observations[0]["points"] if observations else None
            detected += points is not None
            pointer = smooth_pointer(pointer, points[8] if points is not None else None, dt, smoothing)
            _, ratio = update_pinch(points, False, args.pinch_threshold, args.release_threshold)
            excluded = hand_exclusion((height, width), points)

            if capture_requested and points is None:
                calm = previous_frame is None or not compare_scene(previous_frame, frame, excluded)[0]
                if not calm:
                    stable_since = None
                elif stable_since is None:
                    stable_since = now
                elif now - stable_since >= .5:
                    selection.set_reference(frame)
                    capture_requested = False
                    stable_since = None
                    # Warm the image embedding before a fingertip prompt arrives.
                    # The -1 prompt result is never presented as a chosen target.
                    worker.request(selection.reference, (width / 2, height / 2), selection.reference_id, -1)
            elif capture_requested:
                stable_since = None

            target_changed, exposed = False, 1.0
            if selection.reference is not None:
                scene_changed, exposed, target_changed = compare_scene(
                    selection.reference, frame, excluded, selection.mask)
                invalid = scene_changed or (target_changed and exposed >= .65)
                if invalid:
                    if changed_since is None:
                        changed_since = now
                    elif now - changed_since >= .30:
                        selection.invalidate()
                        worker.discard_pending()
                        changed_since = None
                else:
                    changed_since = None

            # Advance the prompt generation before polling; old worker completions
            # must never revive a preview after the user aimed somewhere else.
            point = selection.aim(pointer, now)
            if point is not None:
                worker.request(selection.reference, point, selection.reference_id, selection.prompt_id)
            result = worker.poll()
            if result is not None and selection.accept(result):
                print(f"Segmentation: {len(result.masks)} plausible masks; "
                      f"encoder {result.encoder_ms:.0f}ms, decoder {result.decoder_ms:.0f}ms", flush=True)
                # A new mask has not yet been compared with the live target region.
                if selection.mask is not None:
                    _, exposed, target_changed = compare_scene(selection.reference, frame, excluded, selection.mask)
            if selection.update_confirmation(ratio, now, exposed, target_changed or changed_since is not None):
                confirmed_count += 1
                print("Static reference silhouette confirmed", flush=True)
            if worker.error and worker.error != logged_error:
                print(worker.error, file=sys.stderr, flush=True)
                logged_error = worker.error
            previous_frame = frame.copy()  # Keep raw imagery, before UI/mask drawing.

            timestamps.append(time.perf_counter())
            fps = ((len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
                   if len(timestamps) > 1 else 0.0)
            draw_selection(frame, selection, pointer, points, ratio, fps, worker, debug, now)
            count += 1
            if now >= next_report:
                print(f"Live: {fps:.1f} FPS; hand={points is not None}; "
                      f"reference={selection.reference is not None}; busy={worker.busy}; "
                      f"confirmed={confirmed_count}", flush=True)
                next_report = now + 5
            if not args.headless:
                cv2.imshow(WINDOW, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                if key in (27, ord("r"), ord("R")):
                    selection.clear()
                    worker.discard_pending()
                    selection.message = "Aim again at a real object" if selection.reference is not None else selection.message
                if key in (ord("b"), ord("B")):
                    selection.invalidate()
                    worker.discard_pending()
                    capture_requested, stable_since, changed_since = True, None, None
                    selection.message = "Lower hand briefly; keep the target objects visible"
                if key in (ord("m"), ord("M")):
                    selection.cycle()
                if key in (ord("d"), ord("D")):
                    debug = not debug
                if key in (ord("f"), ord("F")):
                    fullscreen = not fullscreen
                    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                    if not fullscreen:
                        cv2.resizeWindow(WINDOW, *window_size)
                if key == ord("["):
                    smoothing = max(0, round(smoothing - .005, 3))
                if key == ord("]"):
                    smoothing = min(.15, round(smoothing + .005, 3))
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if args.seconds and time.perf_counter() - started >= args.seconds:
                break
            ok, frame = camera.read()
            if not ok or frame is None or not frame.size:
                raise RuntimeError("Webcam stopped delivering frames. Reconnect it and restart.")
    finally:
        finished = time.perf_counter()
        if camera is not None:
            camera.release()
        if tracker is not None:
            tracker.close()
        cv2.destroyAllWindows()
        if worker is not None:
            worker.close()
        if started is not None and count:
            print(f"Run: {count} frames / {finished - started:.2f}s = "
                  f"{count / (finished - started):.1f} FPS; hand in {detected} frames; "
                  f"confirmed masks {confirmed_count}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--backend", choices=("auto", "dshow", "msmf"), default="auto")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--smoothing-ms", type=float, default=35)
    parser.add_argument("--pinch-threshold", type=float, default=.30)
    parser.add_argument("--release-threshold", type=float, default=.45)
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument("--threads", type=int, choices=range(1, 7), default=2)
    parser.add_argument("--seconds", type=float, default=0)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if not all(math.isfinite(v) for v in (args.smoothing_ms, args.pinch_threshold, args.release_threshold, args.seconds)):
        parser.error("Numeric settings must be finite.")
    if not 0 <= args.smoothing_ms <= 150 or not 0 < args.pinch_threshold < args.release_threshold:
        parser.error("Smoothing must be 0-150 ms and thresholds must satisfy 0 < pinch < release.")
    if args.seconds < 0 or (args.headless and args.seconds <= 0):
        parser.error("Headless runs require a positive --seconds duration.")
    return args


if __name__ == "__main__":
    try:
        run(parse_args())
    except KeyboardInterrupt:
        pass
    except (RuntimeError, OSError, ValueError, cv2.error) as error:
        print(f"Telekinesis CV: {error}", file=sys.stderr)
        sys.exit(1)
